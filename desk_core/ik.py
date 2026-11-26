"""Single-shot ElRobot pose IK with PINK and Pinocchio."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

SIMULATION_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = SIMULATION_DIR / "elrobot_follower.urdf"
ARM_JOINTS = tuple(f"rev_motor_{index:02d}" for index in range(1, 8))
TOOL_FRAME = "Gripper_Base_v1_1"
SELF_COLLISION_CLEARANCE_M = 0.001
TABLE_COLLISION_SIZE_XY_M = 2.0
TABLE_COLLISION_THICKNESS_M = 0.02
TABLE_GEOMETRY_NAME = "ik_table"


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    lower: float | None
    upper: float | None
    parent: str
    child: str


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    minimum_self_distance_m: float | None
    barrier_warnings: tuple[str, ...]
    minimum_table_distance_m: float | None = None
    termination_reason: str = "max_iterations"
    closest_self_collision_pair: tuple[str, str] | None = None
    closest_table_link: str | None = None


def inspect_urdf(
    urdf_path: Path = DEFAULT_URDF,
) -> tuple[list[JointInfo], list[str], list[str]]:
    """Return joints, links, and links that define collision geometry."""
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        joints.append(
            JointInfo(
                name=joint.attrib["name"],
                joint_type=joint.attrib["type"],
                lower=(
                    float(limit.attrib["lower"])
                    if limit is not None and "lower" in limit.attrib
                    else None
                ),
                upper=(
                    float(limit.attrib["upper"])
                    if limit is not None and "upper" in limit.attrib
                    else None
                ),
                parent=joint.find("parent").attrib["link"],
                child=joint.find("child").attrib["link"],
            )
        )
    links = [link.attrib["name"] for link in root.findall("link")]
    collidable = [
        link.attrib["name"]
        for link in root.findall("link")
        if link.find("collision") is not None
    ]
    return joints, links, collidable


def _mesh_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * triangle_count == len(data):
            triangles = np.frombuffer(
                data,
                dtype=np.dtype(
                    [
                        ("normal", "<f4", (3,)),
                        ("vertices", "<f4", (3, 3)),
                        ("attribute", "<u2"),
                    ]
                ),
                count=triangle_count,
                offset=84,
            )
            return triangles["vertices"].reshape(-1, 3).astype(float)

    text = data.decode("ascii")
    vertices = re.findall(
        r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text
    )
    if not vertices:
        raise ValueError(f"Could not read STL vertices from {path}")
    return np.asarray(vertices, dtype=float)


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized quaternion in xyzw order."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def _arm_only_urdf(
    urdf_path: Path, output_path: Path
) -> tuple[
    dict[str, tuple[float, float, int]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Write arm-only URDF and return mesh-bounds capsule proxy dimensions."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    proxy_dimensions = {}
    table_proxy_dimensions = {}
    parent_joint = {
        joint.find("child").attrib["link"]: joint for joint in root.findall("joint")
    }

    keep_links = {TOOL_FRAME}
    keep_joints = set()
    current_link = TOOL_FRAME
    while current_link != "base_link":
        joint = parent_joint.get(current_link)
        if joint is None:
            raise ValueError(f"Cannot trace {TOOL_FRAME!r} back to base_link")
        keep_joints.add(joint)
        parent_link = joint.find("parent").attrib["link"]
        keep_links.add(parent_link)
        current_link = parent_link

    for link in list(root.findall("link")):
        if link.attrib["name"] not in keep_links:
            root.remove(link)
            continue
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None:
                continue
            filename = mesh.attrib["filename"]
            if filename.startswith("package://"):
                raise ValueError(
                    f"package:// mesh URIs are not supported by the local proxy builder: {filename}"
                )
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh_path = urdf_path.parent / mesh_path
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Collision mesh does not exist: {mesh_path}")

            vertices = _mesh_vertices(mesh_path)
            scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
            if scale.size != 3:
                raise ValueError(f"Expected a 3D mesh scale in {filename!r}")
            vertices *= scale
            minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
            mesh_center = (minimum + maximum) / 2.0
            half_extents = (maximum - minimum) / 2.0
            major_axis = int(np.argmax(half_extents))
            transverse_axes = [axis for axis in range(3) if axis != major_axis]
            transverse_offsets = (
                vertices[:, transverse_axes] - mesh_center[transverse_axes]
            )
            transverse_distances = np.linalg.norm(transverse_offsets, axis=1)
            radius = float(transverse_distances.max())
            if radius <= 0.0:
                radius = float(np.max(half_extents))
            axial_offsets = np.abs(vertices[:, major_axis] - mesh_center[major_axis])
            cap_reach = np.sqrt(np.maximum(radius**2 - transverse_distances**2, 0.0))
            half_length = float(np.maximum(axial_offsets - cap_reach, 0.0).max())
            if radius <= 0.0:
                raise ValueError(f"Collision mesh has no volume: {mesh_path}")
            proxy_dimensions[link.attrib["name"]] = (radius, half_length, major_axis)

            origin = collision.find("origin")
            if origin is None:
                origin = ET.Element("origin", xyz="0 0 0", rpy="0 0 0")
                collision.insert(0, origin)
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            if xyz.size != 3 or rpy.size != 3:
                raise ValueError(
                    f"Invalid collision origin on link {link.attrib['name']}"
                )
            center = xyz + _rpy_rotation(rpy) @ mesh_center
            table_proxy_dimensions[link.attrib["name"]] = (
                2.0 * half_extents,
                center,
                rpy,
            )
            origin.set("xyz", " ".join(f"{value:.9g}" for value in center))
            geometry.clear()
            ET.SubElement(
                geometry,
                "sphere",
                radius=f"{radius + half_length:.9g}",
            )

    for joint in list(root.findall("joint")):
        if joint not in keep_joints:
            root.remove(joint)

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return proxy_dimensions, table_proxy_dimensions


def _axis_to_z_rotation(axis: int) -> np.ndarray:
    if axis == 0:
        return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    if axis == 1:
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return np.eye(3)


def _link_distances(urdf_path: Path) -> dict[tuple[str, str], int]:
    root = ET.parse(urdf_path).getroot()
    graph: dict[str, set[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        graph.setdefault(parent, set()).add(child)
        graph.setdefault(child, set()).add(parent)

    distances = {}
    for start in graph:
        visited = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in graph[node] - visited.keys():
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)
        for end, distance in visited.items():
            distances[(start, end)] = distance
    return distances


def _load_models(urdf_path: Path, table_z: float):
    try:
        import pinocchio as pin
    except ImportError as error:
        raise ImportError(
            "IK dependencies are missing; install hardware/elrobot/simulation/requirements-ik.txt"
        ) from error

    temp_dir = tempfile.TemporaryDirectory(prefix="elrobot-ik-")
    derived_urdf = Path(temp_dir.name) / "elrobot_arm_collision_proxy.urdf"
    proxy_dimensions, table_proxy_dimensions = _arm_only_urdf(
        urdf_path, derived_urdf
    )
    model, collision_model, _ = pin.buildModelsFromUrdf(
        str(derived_urdf), package_dirs=[str(urdf_path.parent)]
    )
    try:
        import coal
    except ImportError as error:
        temp_dir.cleanup()
        raise ImportError(
            "Pinocchio's coal collision primitives are required"
        ) from error
    for geometry_object in collision_model.geometryObjects:
        frame_name = model.frames[geometry_object.parentFrame].name
        radius, half_length, major_axis = proxy_dimensions[frame_name]
        geometry_object.geometry = coal.Capsule(radius, 2.0 * half_length)
        rotation = geometry_object.placement.rotation @ _axis_to_z_rotation(major_axis)
        geometry_object.placement = pin.SE3(
            rotation, geometry_object.placement.translation
        )
    collision_model.removeAllCollisionPairs()
    collision_model.addAllCollisionPairs()

    distances = _link_distances(derived_urdf)
    kept_pairs = []
    for pair in collision_model.collisionPairs:
        first = collision_model.geometryObjects[pair.first]
        second = collision_model.geometryObjects[pair.second]
        first_frame = model.frames[first.parentFrame].name
        second_frame = model.frames[second.parentFrame].name
        separation = distances.get((first_frame, second_frame))
        if separation is None:
            raise ValueError(
                f"Cannot map collision geometry frames {first_frame!r}, {second_frame!r}"
            )
        if separation > 2:
            kept_pairs.append(pair)
    collision_model.removeAllCollisionPairs()
    for pair in kept_pairs:
        collision_model.addCollisionPair(pair)

    neutral_data = model.createData()
    neutral_collision_data = pin.GeometryData(collision_model)
    pin.computeDistances(
        model,
        neutral_data,
        collision_model,
        neutral_collision_data,
        pin.neutral(model),
    )
    resting_pairs = []
    usable_pairs = []
    for pair, distance in zip(
        collision_model.collisionPairs, neutral_collision_data.distanceResults
    ):
        if distance.min_distance <= SELF_COLLISION_CLEARANCE_M:
            first = collision_model.geometryObjects[pair.first]
            second = collision_model.geometryObjects[pair.second]
            first_frame = model.frames[first.parentFrame].name
            second_frame = model.frames[second.parentFrame].name
            resting_pairs.append(f"{first_frame} / {second_frame}")
        else:
            usable_pairs.append(pair)
    collision_model.removeAllCollisionPairs()
    for pair in usable_pairs:
        collision_model.addCollisionPair(pair)
    if not kept_pairs:
        temp_dir.cleanup()
        raise ValueError(
            "No non-adjacent self-collision pairs remain for the ElRobot arm"
        )
    if not usable_pairs:
        temp_dir.cleanup()
        raise ValueError(
            "All non-adjacent collision pairs overlap in the neutral arm pose"
        )

    table_frame_id = model.getFrameId("base_link")
    table_frame = model.frames[table_frame_id]
    table_placement = table_frame.placement * pin.SE3(
        np.eye(3),
        np.array(
            [
                0.0,
                0.0,
                table_z - TABLE_COLLISION_THICKNESS_M / 2.0,
            ]
        ),
    )
    table_geometry = pin.GeometryObject(
        TABLE_GEOMETRY_NAME,
        table_frame.parentJoint,
        table_frame_id,
        table_placement,
        coal.Box(
            TABLE_COLLISION_SIZE_XY_M,
            TABLE_COLLISION_SIZE_XY_M,
            TABLE_COLLISION_THICKNESS_M,
        ),
    )
    table_geometry_id = collision_model.addGeometryObject(table_geometry, model)

    for link_name, (size, center, rpy) in table_proxy_dimensions.items():
        if link_name == "base_link":
            continue
        frame_id = model.getFrameId(link_name)
        frame = model.frames[frame_id]
        mesh_placement = pin.SE3(_rpy_rotation(rpy), center)
        proxy = pin.GeometryObject(
            f"table_proxy_{link_name}",
            frame.parentJoint,
            frame_id,
            frame.placement * mesh_placement,
            coal.Box(*size.tolist()),
        )
        proxy_id = collision_model.addGeometryObject(proxy, model)
        collision_model.addCollisionPair(
            pin.CollisionPair(table_geometry_id, proxy_id)
        )

    return (
        pin,
        model,
        collision_model,
        temp_dir,
        tuple(resting_pairs),
        table_geometry_id,
    )


class ElRobotIK:
    """PINK differential IK loop for the 7-joint arm, excluding its gripper DOF."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF, table_z: float = 0.0):
        self.table_z = float(table_z)
        if not math.isfinite(self.table_z):
            raise ValueError("table_z must be finite")
        try:
            import pink
            from pink.barriers import PositionBarrier, SelfCollisionBarrier
            from pink.exceptions import NoSolutionFound
            from pink.tasks import DampingTask, FrameTask, PostureTask
        except ImportError as error:
            raise ImportError(
                "PINK is not installed; install hardware/elrobot/simulation/requirements-ik.txt"
            ) from error

        self.urdf_path = Path(urdf_path).resolve()
        (
            self.pin,
            self.model,
            self.collision_model,
            self._temp_dir,
            self._resting_pairs_excluded,
            self._table_geometry_id,
        ) = _load_models(self.urdf_path, self.table_z)
        self._pink = pink
        self._NoSolutionFound = NoSolutionFound
        self._FrameTask = FrameTask
        self._PostureTask = PostureTask
        self._DampingTask = DampingTask
        self._PositionBarrier = PositionBarrier
        self._SelfCollisionBarrier = SelfCollisionBarrier

        if not self.model.existFrame(TOOL_FRAME):
            self.close()
            raise ValueError(f"Tool frame {TOOL_FRAME!r} was not loaded from the URDF")
        missing_joints = [
            name for name in ARM_JOINTS if not self.model.existJointName(name)
        ]
        if missing_joints:
            self.close()
            raise ValueError(
                f"Arm joints missing from Pinocchio model: {missing_joints}"
            )
        if self.model.nq != len(ARM_JOINTS) or self.model.nv != len(ARM_JOINTS):
            self.close()
            raise ValueError(
                f"Expected a 7-DOF arm-only model, got nq={self.model.nq}, nv={self.model.nv}"
            )

        self._q_indices = [
            self.model.idx_qs[self.model.getJointId(name)] for name in ARM_JOINTS
        ]
        self._collision_data = self.pin.GeometryData(self.collision_model)
        self._barriers = [
            self._SelfCollisionBarrier(
                n_collision_pairs=len(self.collision_model.collisionPairs),
                d_min=SELF_COLLISION_CLEARANCE_M,
                safe_displacement_gain=1.0,
            ),
            self._PositionBarrier(
                frame=TOOL_FRAME,
                indices=[2],
                p_min=np.array([self.table_z + 0.015]),
                gain=1.0,
            ),
        ]

    def close(self) -> None:
        """Release temporary derived URDF files."""
        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self) -> "ElRobotIK":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _checked_q_init(self, q_init: np.ndarray | None) -> np.ndarray:
        q = (
            self.pin.neutral(self.model)[self._q_indices]
            if q_init is None
            else np.asarray(q_init, dtype=float)
        )
        if q.shape != (len(ARM_JOINTS),) or not np.isfinite(q).all():
            raise ValueError(
                f"q_init must be a finite vector of shape (7,), got {q.shape}"
            )
        lower = self.model.lowerPositionLimit[self._q_indices]
        upper = self.model.upperPositionLimit[self._q_indices]
        if np.any(q < lower) or np.any(q > upper):
            raise ValueError("q_init is outside the ElRobot arm joint limits")
        return q.copy()

    def solve_ik_with_diagnostics(
        self,
        pos_xyz,
        quat_xyzw,
        q_init=None,
        *,
        max_iters: int = 800,
        position_tolerance_m: float = 0.03,
        orientation_tolerance_rad: float = 0.015,
        dt: float = 0.02,
    ) -> IKResult:
        """Solve one target pose; report q and convergence/barrier diagnostics."""
        position = np.asarray(pos_xyz, dtype=float)
        quaternion = np.asarray(quat_xyzw, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("pos_xyz must be a finite vector [x, y, z]")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("quat_xyzw must be a finite vector [qx, qy, qz, qw]")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-12:
            raise ValueError("quat_xyzw must not be the zero quaternion")
        if abs(quaternion_norm - 1.0) > 1e-3:
            raise ValueError(
                f"quat_xyzw must be normalized (norm={quaternion_norm:.6g})"
            )
        if (
            not isinstance(max_iters, int)
            or max_iters < 1
            or not math.isfinite(dt)
            or dt <= 0.0
            or not math.isfinite(position_tolerance_m)
            or position_tolerance_m < 0.0
            or not math.isfinite(orientation_tolerance_rad)
            or orientation_tolerance_rad < 0.0
        ):
            raise ValueError(
                "max_iters and dt must be positive; tolerances must be finite and nonnegative"
            )
        q = self._checked_q_init(q_init)

        qx, qy, qz, qw = quaternion / quaternion_norm
        rotation = np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ]
        )
        target = self.pin.SE3(rotation, position)
        frame_task = self._FrameTask(
            frame=TOOL_FRAME, position_cost=1.0, orientation_cost=0.3
        )
        frame_task.set_target(target)
        posture_task = self._PostureTask(cost=1e-2)
        posture_task.set_target(self._configuration_q(q))
        damping_task = self._DampingTask(cost=1e-4)
        tasks = [frame_task, posture_task, damping_task]

        data = self.model.createData()
        configuration = self._pink.Configuration(
            self.model,
            data,
            self._configuration_q(q),
            collision_model=self.collision_model,
            collision_data=self._collision_data,
        )

        position_error = math.inf
        orientation_error = math.inf
        iterations = 0
        converged = False
        qp_warning = None
        termination_reason = "max_iterations"
        for iteration in range(max_iters + 1):
            current = configuration.get_transform_frame_to_world(TOOL_FRAME)
            position_error = float(np.linalg.norm(current.translation - position))
            orientation_error = float(
                np.linalg.norm(self.pin.log3(current.rotation.T @ rotation))
            )
            if (
                position_error <= position_tolerance_m
                and orientation_error <= orientation_tolerance_rad
            ):
                converged = True
                termination_reason = "converged"
                break
            if iteration == max_iters:
                break
            try:
                velocity = self._pink.solve_ik(
                    configuration,
                    tasks,
                    dt=dt,
                    solver="quadprog",
                    damping=1e-6,
                    barriers=self._barriers,
                )
            except self._NoSolutionFound as error:
                qp_warning = f"PINK QP failed at iteration {iteration + 1}: {error}"
                termination_reason = "qp_failure"
                break
            configuration.integrate_inplace(velocity, dt)
            iterations += 1

        q_result = configuration.q[self._q_indices].copy()
        minimum_distance = None
        closest_self_pair = None
        minimum_table_distance = None
        closest_table_link = None
        for pair, distance in zip(
            self.collision_model.collisionPairs,
            configuration.collision_data.distanceResults,
        ):
            if self._table_geometry_id in (pair.first, pair.second):
                other_index = (
                    pair.second if pair.first == self._table_geometry_id else pair.first
                )
                link_name = self.model.frames[
                    self.collision_model.geometryObjects[other_index].parentFrame
                ].name
                if (
                    minimum_table_distance is None
                    or distance.min_distance < minimum_table_distance
                ):
                    minimum_table_distance = float(distance.min_distance)
                    closest_table_link = link_name
            else:
                first_name = self.model.frames[
                    self.collision_model.geometryObjects[pair.first].parentFrame
                ].name
                second_name = self.model.frames[
                    self.collision_model.geometryObjects[pair.second].parentFrame
                ].name
                if minimum_distance is None or distance.min_distance < minimum_distance:
                    minimum_distance = float(distance.min_distance)
                    closest_self_pair = (first_name, second_name)
        warnings = [qp_warning] if qp_warning is not None else []
        if self._resting_pairs_excluded:
            warnings.append(
                "Neutral-pose proxy pairs at or below the "
                f"{SELF_COLLISION_CLEARANCE_M:.3f} m clearance were excluded: "
                + ", ".join(self._resting_pairs_excluded)
                + "; these pairs are excluded from the PINK barrier"
            )
        if position[2] < self.table_z + 0.015:
            warnings.append(
                f"Requested TCP is below the table clearance ({self.table_z + 0.015:.3f} m)"
            )
        if (
            minimum_distance is not None
            and minimum_distance < SELF_COLLISION_CLEARANCE_M
        ):
            warnings.append(
                f"Final non-adjacent self-clearance is {minimum_distance:.4f} m "
                f"(barrier threshold {SELF_COLLISION_CLEARANCE_M:.3f} m)"
            )
        if (
            minimum_table_distance is not None
            and minimum_table_distance < SELF_COLLISION_CLEARANCE_M
        ):
            warnings.append(
                f"Final arm-to-table clearance is {minimum_table_distance:.4f} m "
                f"(barrier threshold {SELF_COLLISION_CLEARANCE_M:.3f} m)"
            )
        if np.any(
            self.model.upperPositionLimit[self._q_indices] - q_result < 0.02
        ) or np.any(q_result - self.model.lowerPositionLimit[self._q_indices] < 0.02):
            warnings.append("At least one arm joint is within 0.02 rad of a limit")

        return IKResult(
            q=q_result,
            converged=converged,
            iterations=iterations,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            minimum_self_distance_m=minimum_distance,
            barrier_warnings=tuple(warnings),
            minimum_table_distance_m=minimum_table_distance,
            termination_reason=termination_reason,
            closest_self_collision_pair=closest_self_pair,
            closest_table_link=closest_table_link,
        )

    def _configuration_q(self, q_arm: np.ndarray) -> np.ndarray:
        q = self.pin.neutral(self.model)
        q[self._q_indices] = q_arm
        return q

    def solve_ik(self, pos_xyz, quat_xyzw, q_init=None, **kwargs) -> np.ndarray:
        """Return the seven arm joint angles in radians (gripper excluded)."""
        return self.solve_ik_with_diagnostics(
            pos_xyz, quat_xyzw, q_init=q_init, **kwargs
        ).q


def solve_ik(pos_xyz, quat_xyzw, q_init=None, **kwargs) -> np.ndarray:
    """Convenience one-shot solver; use ElRobotIK to reuse a loaded model."""
    with ElRobotIK() as solver:
        return solver.solve_ik(pos_xyz, quat_xyzw, q_init=q_init, **kwargs)
