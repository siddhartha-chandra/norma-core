"""TRAC-IK (pytracik) inverse kinematics for the ElRobot arm.

KINEMATICS CONTRACT (same as desk_core/ik_pink.py):
  - URDF: hardware/elrobot/simulation/elrobot_follower.urdf
  - World frame: base_link
  - EE frame: Gripper_Base_v1_1
  - Arm joints: rev_motor_01 .. rev_motor_07 (radians), q=0 aligned with joint_zero_ticks in rig_calib.json

Solutions are bounded by the given joint limits (normally the Station motor ranges converted to radians)
and verified with forward kinematics, so a returned solution always reaches the target within tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import os
from pathlib import Path
import sys
import threading

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .station_bridge import REPO_ROOT

DEFAULT_URDF = REPO_ROOT / "hardware" / "elrobot" / "simulation" / "elrobot_follower.urdf"
BASE_FRAME = "base_link"
TOOL_FRAME = "Gripper_Base_v1_1"
MAX_POSITION_ERROR_M = 0.0005
MAX_ROTATION_ERROR_RAD = math.radians(0.3)


@dataclass
class IKSolution:
    """Result of one IK query."""

    q: np.ndarray | None
    ok: bool
    position_error_m: float = math.inf
    rotation_error_rad: float = math.inf
    reason: str = ""


@dataclass
class CartesianPlan:
    """Joint waypoints for a straight-line flange motion."""

    path: np.ndarray
    problems: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


class TracIKSolver:
    """TRAC-IK chain from base_link to the flange, with FK verification and limit diagnostics.

    Parameters
    ----------
    urdf_path : pathlib.Path, optional
        Robot URDF; defaults to the ElRobot follower model.
    timeout : float, optional
        Per-query time budget in seconds.
    solve_type : str, optional
        TRAC-IK mode; "Distance" returns the solution closest to the seed, which keeps paths smooth.
    """

    def __init__(self, urdf_path: Path | str = DEFAULT_URDF, timeout: float = float(os.environ.get("DESK_IK_TIMEOUT_S", "").strip() or "0.02"), solve_type: str = "Distance"):
        from trac_ik import TracIK

        self.urdf_path = Path(urdf_path)
        # KDL prints URDF warnings to C stdout, which would corrupt the MCP stdio transport.
        sys.stdout.flush()
        saved_stdout = os.dup(1)
        os.dup2(2, 1)
        try:
            self._ik = TracIK(BASE_FRAME, TOOL_FRAME, str(self.urdf_path), timeout=timeout, epsilon=1e-5,
                              solver_type=solve_type)
        finally:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
        self.urdf_limits = tuple(np.array(v, dtype=float) for v in self._ik.joint_limits)
        # get_solver() shares one instance across threads, and every solve rewrites its joint limits.
        self._lock = threading.Lock()

    @property
    def dof(self) -> int:
        return int(self._ik.dof)

    def fk(self, q) -> np.ndarray:
        """Return the 4x4 base_link -> flange transform for joint angles q (radians)."""
        with self._lock:
            position, rotation = self._ik.fk(np.asarray(q, dtype=float))
        T = np.eye(4)
        T[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
        T[:3, 3] = np.ravel(position)
        return T

    def _raw_solve(self, position, rotation, seed, lower, upper):
        with self._lock:
            self._ik.joint_limits = (lower, upper)
            q = self._ik.ik(np.asarray(position, dtype=float), np.asarray(rotation, dtype=float),
                            seed_jnt_values=np.asarray(seed, dtype=float))
        if q is None or len(q) != self.dof or not np.all(np.isfinite(q)):
            return None
        return np.asarray(q, dtype=float)

    def solve(self, position, rotation, seed, lower=None, upper=None) -> IKSolution:
        """Solve one flange pose within the joint limits.

        Parameters
        ----------
        position : array_like
            Target flange position in base_link, metres.
        rotation : array_like
            Target 3x3 rotation matrix of the flange in base_link.
        seed : array_like
            Starting joint angles, radians; the closest solution to it is returned.
        lower, upper : array_like, optional
            Joint limits in radians; default to the URDF limits.

        Returns
        -------
        IKSolution
            On failure, ``reason`` names the joints that would leave the limits when that can be determined.
        """
        lower = self.urdf_limits[0] if lower is None else np.asarray(lower, dtype=float)
        upper = self.urdf_limits[1] if upper is None else np.asarray(upper, dtype=float)
        seed = np.clip(np.asarray(seed, dtype=float), lower, upper)
        q = self._raw_solve(position, rotation, seed, lower, upper)
        if q is None:
            return IKSolution(None, False, reason=self._blocking_joints(position, rotation, seed, lower, upper))
        T = self.fk(q)
        pos_err = float(np.linalg.norm(T[:3, 3] - np.asarray(position, dtype=float)))
        rot_err = float((Rotation.from_matrix(np.asarray(rotation)).inv() * Rotation.from_matrix(T[:3, :3])).magnitude())
        ok = pos_err <= MAX_POSITION_ERROR_M and rot_err <= MAX_ROTATION_ERROR_RAD
        reason = "" if ok else f"IK error {pos_err * 1000:.2f} mm / {math.degrees(rot_err):.2f} deg"
        return IKSolution(q, ok, pos_err, rot_err, reason)

    def _blocking_joints(self, position, rotation, seed, lower, upper) -> str:
        q = self._raw_solve(position, rotation, seed, *self.urdf_limits)
        if q is None:
            return "pose unreachable even within the URDF joint limits"
        blocked = [f"joint {j + 1} needs {q[j]:+.3f} rad, allowed [{lower[j]:+.3f}, {upper[j]:+.3f}]"
                   for j in range(self.dof) if q[j] < lower[j] - 1e-6 or q[j] > upper[j] + 1e-6]
        return "blocked by limits: " + "; ".join(blocked) if blocked else "no solution within the joint limits"

    def plan_cartesian(self, q0, shift, rotation=None, rot_frame="flange", lower=None, upper=None,
                       waypoint_m=0.002, waypoint_rad=math.radians(1.0),
                       max_jump_rad=0.15) -> CartesianPlan:
        """Solve a straight flange line with slerped orientation, each waypoint seeded from the previous one.

        Parameters
        ----------
        q0 : array_like
            Current joint angles, radians.
        shift : array_like
            Flange translation in base_link, metres.
        rotation : scipy.spatial.transform.Rotation, optional
            Flange rotation increment.
        rot_frame : {"flange", "base"}
            Axes the rotation increment is expressed in; it is applied about the flange origin.
        lower, upper : array_like, optional
            Joint limits in radians.
        waypoint_m, waypoint_rad : float
            Maximum translation and rotation between waypoints.
        max_jump_rad : float
            Refuse a plan whose joints jump more than this between waypoints (configuration flips).

        Returns
        -------
        CartesianPlan
            ``path`` starts with q0 and holds every solved waypoint up to the first problem.
        """
        q0 = np.asarray(q0, dtype=float)
        shift = np.asarray(shift, dtype=float)
        rotation = Rotation.identity() if rotation is None else rotation
        T0 = self.fk(q0)
        R0 = Rotation.from_matrix(T0[:3, :3])
        R1 = R0 * rotation if rot_frame == "flange" else rotation * R0
        angle = (R0.inv() * R1).magnitude()
        n = max(2, math.ceil(max(np.linalg.norm(shift) / waypoint_m, angle / waypoint_rad)))
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([R0, R1]))

        path, problems, q = [q0], [], q0
        worst_pos = worst_rot = 0.0
        for k in range(1, n + 1):
            s = k / n
            sol = self.solve(T0[:3, 3] + s * shift, slerp([s])[0].as_matrix(), q, lower, upper)
            if not sol.ok:
                problems.append(f"waypoint {k}/{n} ({s * 100:.0f}% of the way): {sol.reason}")
                break
            jump = float(np.max(np.abs(sol.q - q)))
            if jump > max_jump_rad:
                problems.append(f"waypoint {k}/{n}: joint jump {math.degrees(jump):.1f} deg (configuration flip)")
                break
            worst_pos, worst_rot = max(worst_pos, sol.position_error_m), max(worst_rot, sol.rotation_error_rad)
            path.append(sol.q)
            q = sol.q
        summary = {
            "flange_start_cm": (T0[:3, 3] * 100).round(2).tolist(),
            "flange_target_cm": ((T0[:3, 3] + shift) * 100).round(2).tolist(),
            "rotation_deg": round(math.degrees(angle), 2),
            "waypoints": n,
            "worst_ik_position_mm": round(worst_pos * 1000, 3),
            "worst_ik_orientation_deg": round(math.degrees(worst_rot), 3),
        }
        return CartesianPlan(np.array(path), problems, summary)


@lru_cache(maxsize=4)
def get_solver(urdf_path: str = str(DEFAULT_URDF)) -> TracIKSolver:
    """Shared solver per URDF; loading the KDL chain is the expensive part."""
    return TracIKSolver(urdf_path)


def solve_ik(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
    seed_joints_rad: list[float] | None = None,
    obstacles=None,
    joint_bounds_rad: list[tuple[float, float]] | None = None,
) -> list[float]:
    """IK plug-in entry point used by desk_core.ik_adapter.IKAdapter.

    Parameters
    ----------
    x, y, z : float
        Target flange position in base_link, metres.
    qx, qy, qz, qw : float
        Target flange orientation quaternion (xyzw).
    seed_joints_rad : list of float, optional
        Current joint angles; the closest solution is returned.
    obstacles : list, optional
        Ignored; TRAC-IK does not model collisions.
    joint_bounds_rad : list of (float, float), optional
        Per-joint limits in radians, e.g. CalibrationStore.bounds_rad(motors); URDF limits otherwise.

    Returns
    -------
    list of float
        Seven joint angles in radians.

    Raises
    ------
    ValueError
        When no verified solution exists within the limits.
    """
    solver = get_solver()
    seed = np.zeros(solver.dof) if seed_joints_rad is None else np.asarray(seed_joints_rad, dtype=float)
    lower = upper = None
    if joint_bounds_rad is not None:
        lower = np.array([b[0] for b in joint_bounds_rad], dtype=float)
        upper = np.array([b[1] for b in joint_bounds_rad], dtype=float)
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    sol = solver.solve([x, y, z], rotation, seed, lower, upper)
    if not sol.ok:
        raise ValueError(f"TRAC-IK: {sol.reason}")
    return [float(v) for v in sol.q]
