"""MuJoCo preflight for the station mock's keys-to-tray transfer."""
from dataclasses import dataclass

import numpy as np

from .ik_mujoco import CONTACT_MARGIN_M, MuJoCoPlanner, PlanningError
from .interfaces import SceneObstacle

MOCK_TCP_DROP_M = 0.12
MOCK_KEYS_MASS_KG = 0.03
STACK_CLEARANCE_M = 0.005


@dataclass
class Transfer:
    approach: np.ndarray
    carry: np.ndarray
    detour: str | None
    flange_path: np.ndarray


def _check_payload(path, radius, height, obstacles, support_z):
    for x, y, flange_z in path:
        z = flange_z - MOCK_TCP_DROP_M
        if z < support_z - CONTACT_MARGIN_M:
            raise PlanningError("Keys would move through the table")
        for ob in obstacles:
            if ob.shape == "cylinder":
                distance = np.hypot(x - ob.x, y - ob.y) - ob.radius_m
                half_height = ob.height_m / 2
            else:
                sx, sy, sz = ob.size_xyz
                distance = np.hypot(max(0, abs(x - ob.x) - sx / 2), max(0, abs(y - ob.y) - sy / 2))
                half_height = sz / 2
            if (distance < radius + CONTACT_MARGIN_M and
                    z < ob.z + half_height + CONTACT_MARGIN_M and z + height > ob.z - half_height):
                raise PlanningError(f"Keys would collide with {ob.name}")


def plan_transfer(planner: MuJoCoPlanner, start, source, destination, radius, height,
                  obstacles: list[SceneObstacle], quantize, table_z: float) -> Transfer:
    """Certify approach and carry before the mock is allowed to move."""
    source = np.asarray(source, dtype=float) + [0.0, 0.0, MOCK_TCP_DROP_M]
    destination = np.asarray(destination, dtype=float) + [0.0, 0.0, MOCK_TCP_DROP_M]
    _, rotation = planner.flange_pose(start)
    quaternion = rotation.as_quat()

    def leg(initial, xyz):
        goal = planner.solve_ik(xyz, quaternion, initial)
        return planner.plan_trajectory(initial, goal, quantize).waypoints

    approach = leg(start, source)
    delta = destination[:2] - source[:2]
    alternatives = [(None, [])]
    if np.linalg.norm(delta) > 1e-6:
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        midpoint = (source + destination) / 2
        for clearance in (0.065, 0.10):
            for side in (-1, 1):
                via = midpoint + [*(normal * clearance * side), 0.03]
                alternatives.append(("around", [via]))
    # Keep the entire crossover above the top of the mug, not just its midpoint.
    top = max((ob.z + (ob.height_m if ob.shape == "cylinder" else ob.size_xyz[2]) / 2
               for ob in obstacles), default=table_z)
    for extra in (0.02, 0.045):
        z = max(top + MOCK_TCP_DROP_M + extra, source[2], destination[2])
        alternatives.append(("over", [[*source[:2], z], [*destination[:2], z]]))
    failure = None
    for detour, vias in alternatives:
        try:
            segments = []
            current = approach[-1]
            for xyz in [*vias, destination]:
                segment = leg(current, xyz)
                segments.append(segment if not segments else segment[1:])
                current = segment[-1]
            carry = np.vstack(segments)
            flange_path = np.array([planner.flange_pose(q)[0] for q in carry])
            _check_payload(flange_path, radius, height, obstacles, table_z)
            peak = max(abs(planner.shoulder_torque(q, payload_mass_kg=MOCK_KEYS_MASS_KG)) for q in carry)
            if peak > planner.torque_limit:
                raise PlanningError(f"Loaded shoulder torque {peak:.3f} N·m exceeds {planner.torque_limit:.3f} N·m")
            return Transfer(approach, carry, detour, flange_path)
        except PlanningError as exc:
            failure = str(exc)
    raise PlanningError(f"No safe transfer to the departure tray: {failure}")
