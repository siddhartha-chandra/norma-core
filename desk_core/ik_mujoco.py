"""MuJoCo preflight for unloaded ElRobot flange motions."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .interfaces import SceneObstacle
from .ik_tracik import MAX_POSITION_ERROR_M, MAX_ROTATION_ERROR_RAD, TOOL_FRAME
from .urdf_fk import DEFAULT_ARM_JOINTS

MAX_WAYPOINT_STEP_RAD = 0.02
CONTACT_MARGIN_M = 0.002
TUCK_OFFSETS_RAD = (0.15, -0.15, 0.30, -0.30, 0.50, -0.50)


class PlanningError(ValueError):
    """The model could not certify a move."""


@dataclass
class JointPlan:
    waypoints: np.ndarray
    tuck_end: int | None
    peak_shoulder_torque_nm: float
    cartesian_checks: dict[int, Callable[[np.ndarray], None]] = field(default_factory=dict)


class MuJoCoPlanner:
    def __init__(self, urdf_path: Path, table_z: float, obstacles: list[SceneObstacle],
                 shoulder_torque_limit_nm: float, joint_bounds_rad: list[tuple[float, float]],
                 gripper_open_fraction: float = 1.0):
        if not np.isfinite(table_z):
            raise PlanningError("Table height must be finite")
        if not math.isfinite(shoulder_torque_limit_nm) or shoulder_torque_limit_nm <= 0:
            raise PlanningError("Shoulder torque limit must be positive and finite")
        self.torque_limit = shoulder_torque_limit_nm

        root = ET.parse(urdf_path).getroot()
        if gripper_open_fraction != 1.0:
            self._pose_gripper(root, gripper_open_fraction)
        ET.SubElement(ET.SubElement(root, "mujoco"), "compiler",
                      strippath="false", fusestatic="false", discardvisual="true")
        for mesh in root.iter("mesh"):
            mesh.set("filename", str((urdf_path.parent / mesh.attrib["filename"]).resolve()))
        for joint in root.findall("joint"):
            if joint.attrib["name"] not in DEFAULT_ARM_JOINTS:
                joint.set("type", "fixed")
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        for i, geom in enumerate(spec.geoms):
            geom.name = f"{geom.parent.name}/{i}"
        spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                pos=[0, 0, table_z], size=[0.7, 0.7, 0.05])
        for i, obstacle in enumerate(obstacles):
            self._add_obstacle(spec, obstacle, i)
        for geom in spec.geoms:
            geom.margin = CONTACT_MARGIN_M
        self.spec = spec
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(name).id for name in DEFAULT_ARM_JOINTS]
        self.qadr = self.model.jnt_qposadr[joints]
        self.shoulder = joints[1]
        self.shoulder_body = self.model.jnt_bodyid[self.shoulder]
        self.flange = self.model.body(TOOL_FRAME).id
        # The CAD hulls of neighboring links overlap at the servo housings.
        self.link_groups = np.zeros(self.model.nbody, dtype=int)
        for body in range(1, self.model.nbody):
            self.link_groups[body] = (self.link_groups[self.model.body_parentid[body]]
                                      + int(self.model.body_jntnum[body] > 0))

        bounds = np.asarray(joint_bounds_rad, dtype=float)
        if bounds.shape != (len(joints), 2) or not np.all(np.isfinite(bounds)):
            raise PlanningError("Expected finite limits for all seven calibrated arm joints")
        limits = self.model.jnt_range[joints]
        self.lower = np.maximum(bounds[:, 0], limits[:, 0])
        self.upper = np.minimum(bounds[:, 1], limits[:, 1])
        if np.any(self.lower >= self.upper):
            raise PlanningError("Calibrated motor ranges do not overlap the MuJoCo joint limits")

    @staticmethod
    def _pose_gripper(root: ET.Element, open_fraction: float) -> None:
        gear = root.find("./joint[@name='rev_motor_08']")
        if gear is None or not 0 <= open_fraction <= 1:
            raise PlanningError("Cannot model the measured gripper opening")
        angle_limits = gear.find("limit").attrib
        lo, hi = float(angle_limits["lower"]), float(angle_limits["upper"])
        gear_angle = hi - open_fraction * (hi - lo)
        for joint in root.findall("joint"):
            mimic = joint.find("mimic")
            if joint is gear:
                q = gear_angle
            elif mimic is not None and mimic.get("joint") == gear.get("name"):
                q = gear_angle * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0"))
            else:
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz"), sep=" ")
            rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" "))
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            axis /= np.linalg.norm(axis)
            if joint.get("type") == "prismatic":
                xyz += rotation.apply(axis * q)
            else:
                rotation *= Rotation.from_rotvec(axis * q)
            origin.set("xyz", " ".join(map(str, xyz)))
            origin.set("rpy", " ".join(map(str, rotation.as_euler("xyz"))))

    @staticmethod
    def _add_obstacle(spec: mujoco.MjSpec, obstacle: SceneObstacle, index: int) -> None:
        if obstacle.shape == "cylinder":
            kind = mujoco.mjtGeom.mjGEOM_CYLINDER
            size = [obstacle.radius_m, obstacle.height_m / 2, 0]
            dimensions = size[:2]
        elif obstacle.shape == "box" and obstacle.size_xyz is not None and len(obstacle.size_xyz) == 3:
            kind = mujoco.mjtGeom.mjGEOM_BOX
            size = dimensions = np.asarray(obstacle.size_xyz, dtype=float) / 2
        else:
            raise PlanningError(f"Unsupported or incomplete obstacle {obstacle.name!r}")
        pos = [obstacle.x, obstacle.y, obstacle.z]
        if (not np.all(np.isfinite(pos)) or not np.all(np.isfinite(dimensions))
                or np.any(np.asarray(dimensions) <= 0)):
            raise PlanningError(f"Invalid obstacle {obstacle.name!r}")
        spec.worldbody.add_geom(name=f"obstacle/{index}/{obstacle.name}", type=kind, size=size, pos=pos)

    def _forward(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def flange_pose(self, q: np.ndarray) -> tuple[np.ndarray, Rotation]:
        self._forward(q)
        return self.data.xpos[self.flange].copy(), Rotation.from_matrix(self.data.xmat[self.flange].reshape(3, 3))

    def jaw_tip_offset(self, q: np.ndarray, rotation: Rotation, nominal_tip: np.ndarray) -> np.ndarray:
        """Use the nominal tip XY with the lowest point of either jaw at the requested orientation."""
        flange, actual = self.flange_pose(q)
        lowest = np.inf
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            if not self.model.body(body_id).name.startswith("Gripper_Jaw_"):
                continue
            mesh = self.model.geom_dataid[geom_id]
            first, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            verts = self.model.mesh_vert[first:first + count]
            geom_rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            world = verts @ geom_rotation.T + self.data.geom_xpos[geom_id]
            lowest = min(lowest, (rotation * actual.inv()).apply(world - flange)[:, 2].min())
        if not np.isfinite(lowest):
            raise PlanningError("MuJoCo model is missing gripper jaw geometry")
        offset = rotation.apply(nominal_tip)
        offset[2] = lowest
        return rotation.inv().apply(offset)

    def _shoulder_torque(self) -> float:
        # Holding torque = -axis · (lever arm × total downstream gravity force).
        lever = self.data.subtree_com[self.shoulder_body] - self.data.xanchor[self.shoulder]
        force = self.model.body_subtreemass[self.shoulder_body] * self.model.opt.gravity
        return -float(self.data.xaxis[self.shoulder] @ np.cross(lever, force))

    def shoulder_torque(self, q: np.ndarray) -> float:
        """Signed gravity holding torque about j2 in N·m."""
        self._forward(q)
        return self._shoulder_torque()

    def _check(self, q: np.ndarray) -> float:
        if (q.shape != self.lower.shape or not np.all(np.isfinite(q))
                or np.any(q < self.lower - 1e-6) or np.any(q > self.upper + 1e-6)):
            raise PlanningError("Joint waypoint outside calibrated limits")
        self._forward(q)
        torque = abs(self._shoulder_torque())
        if torque > self.torque_limit:
            raise PlanningError(f"Shoulder torque {torque:.3f} N·m exceeds {self.torque_limit:.3f} N·m")
        mujoco.mj_collision(self.model, self.data)
        for contact in self.data.contact:
            geom1, geom2 = (self.model.geom(g) for g in contact.geom)
            body1, body2 = self.model.geom_bodyid[contact.geom]
            group1, group2 = self.link_groups[[body1, body2]]
            if body1 and body2 and abs(group1 - group2) <= 1:
                continue
            if (geom1.name == "table" and body2 and not group2
                    or geom2.name == "table" and body1 and not group1):
                continue  # The fixed mounting base is allowed to touch the desk.
            raise PlanningError(f"Collision clearance violated: {geom1.name} / {geom2.name}")
        return torque

    def solve_ik(self, position, quat_xyzw, seed_joints_rad) -> np.ndarray:
        """Solve a flange pose in base_link, with a gravity penalty and hard safety checks."""
        position, quat = np.asarray(position, dtype=float), np.asarray(quat_xyzw, dtype=float)
        seed = np.asarray(seed_joints_rad, dtype=float)
        if (position.shape != (3,) or quat.shape != (4,) or np.linalg.norm(quat) == 0
                or seed.shape != self.lower.shape or not np.all(np.isfinite(np.r_[position, quat, seed]))):
            raise PlanningError("Expected a finite XYZ, nonzero xyzw quaternion and seven-joint seed")
        target_rotation = Rotation.from_quat(quat)

        def residual(q):
            xyz, rotation = self.flange_pose(q)
            torque = self._shoulder_torque() / self.torque_limit
            overload = max(0.0, abs(torque) - 1.0)
            return np.r_[xyz - position, 0.1 * (target_rotation * rotation.inv()).as_rotvec(),
                         0.0003 * torque, 0.01 * overload]

        failure = "Flange pose is unreachable within pose tolerances and joint limits"
        for start in (seed, (self.lower + self.upper) / 2):
            result = least_squares(residual, np.clip(start, self.lower, self.upper),
                                   bounds=(self.lower, self.upper), max_nfev=120)
            xyz, rotation = self.flange_pose(result.x)
            if (np.linalg.norm(xyz - position) <= MAX_POSITION_ERROR_M
                    and (target_rotation * rotation.inv()).magnitude() <= MAX_ROTATION_ERROR_RAD):
                try:
                    self._check(result.x)
                except PlanningError as exc:
                    failure = str(exc)
                else:
                    return result.x
        raise PlanningError(f"MuJoCo IK rejected: {failure}")

    def _validate_path(self, points: list[np.ndarray],
                       quantize: Callable[[np.ndarray], np.ndarray], sequential: bool = False,
                       check: Callable[[np.ndarray], None] | None = None) -> JointPlan:
        path = [quantize(points[0])]
        peak = self._check(path[0])
        if check is not None:
            check(path[0])
        tuck_end = None
        for i, (start, end) in enumerate(zip(points, points[1:])):
            count = max(1, math.ceil(float(np.max(np.abs(end - start))) / MAX_WAYPOINT_STEP_RAD))
            for q in np.linspace(start, end, count + 1)[1:]:
                commanded = quantize(q)
                try:
                    if sequential:
                        for axis in np.flatnonzero(commanded != path[-1]):
                            single = path[-1].copy()
                            single[axis] = commanded[axis]
                            segment = self._validate_path([path[-1], single], np.copy, check=check)
                            peak = max(peak, segment.peak_shoulder_torque_nm)
                            path.append(single)
                    else:
                        peak = max(peak, self._check(commanded))
                        if check is not None:
                            check(commanded)
                        if not np.array_equal(commanded, q):
                            self._check(q)
                        if not np.array_equal(commanded, path[-1]):
                            path.append(commanded)
                except PlanningError as exc:
                    raise PlanningError(f"Waypoint {len(path)}: {exc}") from exc
            if len(points) == 4 and i == 0:
                tuck_end = len(path) - 1
        return JointPlan(np.asarray(path), tuck_end, peak)

    def validate_segment(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Check a direct segment from the measured pose to a preflighted target."""
        return self._validate_path([start, goal], np.copy).peak_shoulder_torque_nm

    def validate_cartesian_segment(self, start: np.ndarray, goal: np.ndarray, *,
                                   check: Callable[[np.ndarray], None]) -> float:
        return self._validate_path([start, goal], np.copy, check=check).peak_shoulder_torque_nm

    def plan_cartesian_segment(self, start: np.ndarray, from_xyz: np.ndarray, to_xyz: np.ndarray,
                               rotation: Rotation, tip_in_flange: np.ndarray, *,
                               quantize: Callable[[np.ndarray], np.ndarray], sequential: bool) -> JointPlan:
        """Solve and check the motion of a fixed tool point along a Cartesian line."""
        delta = to_xyz - from_xyz
        length = float(np.linalg.norm(delta))
        direction = delta / length
        flange_offset = rotation.apply(tip_in_flange)
        points = [start]
        for distance in np.linspace(0, length, math.ceil(length / 0.005) + 1)[1:]:
            xyz = from_xyz + distance * direction - flange_offset
            points.append(self.solve_ik(xyz, rotation.as_quat(), points[-1]))

        def check(q):
            flange, actual = self.flange_pose(q)
            tip = flange + actual.apply(tip_in_flange)
            along = np.clip((tip - from_xyz) @ direction, 0, length)
            if (np.linalg.norm(tip - from_xyz - along * direction) > 0.004
                    or (actual * rotation.inv()).magnitude() > math.radians(3)):
                raise PlanningError("Joint motion deviates from the Cartesian line")

        plan = self._validate_path(points, quantize, sequential, check)
        plan.cartesian_checks = dict.fromkeys(range(1, len(plan.waypoints)), check)
        return plan

    def plan_trajectory(self, start, goal, quantize: Callable[[np.ndarray], np.ndarray] | None = None, *,
                        sequential: bool = False) -> JointPlan:
        """Preflight every waypoint, tucking j3/j4 before j2 changes on a lift."""
        quantize = quantize or np.copy
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        self._check(start)
        self._check(goal)
        shoulder_only = start.copy()
        shoulder_only[1] = goal[1]
        start_z = self.flange_pose(start)[0][2]
        rise = max(self.flange_pose(goal)[0][2], self.flange_pose(shoulder_only)[0][2]) - start_z
        if rise <= MAX_POSITION_ERROR_M or abs(goal[1] - start[1]) < 1e-6:
            return self._validate_path([start, goal], quantize, sequential)

        initial_torque = abs(self.shoulder_torque(start))
        candidates = []
        for elbow, wrist in [(goal[2], goal[3]), *(
            (start[2] + e, start[3] + w) for e in TUCK_OFFSETS_RAD for w in TUCK_OFFSETS_RAD
        )]:
            tucked = start.copy()
            tucked[2:4] = np.clip([elbow, wrist], self.lower[2:4], self.upper[2:4])
            tucked = quantize(tucked)
            if np.any(np.abs(tucked[2:4] - start[2:4]) < 1e-6):
                continue
            reduction = initial_torque - abs(self.shoulder_torque(tucked))
            if reduction > 1e-6:
                candidates.append((float(np.linalg.norm(tucked - start)), -reduction, tucked))

        failure = "No j3/j4 tuck reduces the starting shoulder torque"
        for _, _, tucked in sorted(candidates, key=lambda candidate: candidate[:2]):
            lifted = goal.copy()
            lifted[2:4] = tucked[2:4]
            try:
                return self._validate_path([start, tucked, lifted, goal], quantize, sequential)
            except PlanningError as exc:
                failure = str(exc)
        raise PlanningError(f"Tuck-first lift blocked: {failure}")
