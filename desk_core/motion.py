"""Smooth Cartesian flange motions: TRAC-IK planning, minimum-jerk timing and streamed joint targets."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .calib_store import TICKS_PER_RAD, CalibrationStore
from .ik_tracik import TOOL_FRAME, get_solver
from .station_bridge import MotorState, StationBridge
from .urdf_fk import load_urdf

log = logging.getLogger(__name__)

# Midpoint of the two jaw tips in the flange frame, from the jaw meshes in elrobot_follower.urdf.
TCP_IN_FLANGE = np.array([0.0018, 0.1030, -0.0007])
HEIGHT_CHECK_LINKS = ("ST3215_3_v1_1", "ST3215_5_v1_1", "ST3215_7_v1_1", TOOL_FRAME)
RANGE_MARGIN_TICKS = 50
GRAVITY_FF_PATH = Path(__file__).resolve().parents[1] / "calib" / "gravity_ff.json"
GRAVITY_FF_MAX_RAD = 0.15


@dataclass
class MotionLimits:
    """Pace and safety settings for a flange move."""

    speed_cm_s: float = 1.0
    rot_speed_deg_s: float = 10.0
    joint_speed_deg_s: float = 20.0
    min_duration_s: float = 2.0
    floor_cm: float = 8.0
    rate_hz: float = 10.0
    lag_limit_ticks: int = 80
    max_step_ticks: int = 150
    correct_iterations: int = 4
    correct_tol_mm: float = 2.0
    correct_max_offset_ticks: int = 120


def min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau * tau)


def _sample(path: np.ndarray, s: float) -> np.ndarray:
    x = s * (len(path) - 1)
    i = min(int(x), len(path) - 2)
    return path[i] + (x - i) * (path[i + 1] - path[i])


@lru_cache(maxsize=1)
def _dynamics(urdf_path: str):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(urdf_path)
    return pin, model, model.createData()


def gravity_torque(q, urdf_path: str | None = None) -> np.ndarray:
    """Holding torque (N m) of the 7 arm joints at joint angles ``q`` (radians), from the URDF masses."""
    pin, model, data = _dynamics(urdf_path or str(CalibrationStore().urdf_path()))
    q_full = np.zeros(model.nq)
    q_full[:7] = np.asarray(q, dtype=float)[:7]
    return np.array(pin.computeGeneralizedGravity(model, data, q_full)[:7])


@lru_cache(maxsize=1)
def gravity_gains() -> tuple[np.ndarray, np.ndarray]:
    """Per-joint gravity gains (rad per N m) and friction bands (rad) from scripts/fit_gravity_ff.py; zeros if absent."""
    try:
        data = json.loads(GRAVITY_FF_PATH.read_text())
        return (np.array(data["gain_rad_per_Nm"], dtype=float),
                np.array(data.get("friction_rad", [0.0] * 7), dtype=float))
    except (OSError, ValueError, KeyError) as exc:
        log.info("no gravity feed-forward gains (%s)", exc)
        return np.zeros(7), np.zeros(7)


def gravity_offset_rad(q, urdf_path: str | None = None, direction=None) -> np.ndarray:
    """Target offset that cancels the servos' gravity sag at ``q`` and, given per-joint move directions
    (-1, 0, +1), their friction band; bounded by GRAVITY_FF_MAX_RAD."""
    gains, friction = gravity_gains()
    if not gains.any():
        return np.zeros(7)
    offset = gains * gravity_torque(q, urdf_path)
    if direction is not None:
        offset = offset + friction * np.asarray(direction, dtype=float)
    return np.clip(offset, -GRAVITY_FF_MAX_RAD, GRAVITY_FF_MAX_RAD)


def move_direction(path: np.ndarray, min_travel_rad: float = 0.005) -> np.ndarray:
    """Per-joint direction of a move (-1, 0, +1); joints travelling less than ``min_travel_rad`` count as still."""
    travel = path[-1] - path[0]
    return np.where(np.abs(travel) >= min_travel_rad, np.sign(travel), 0.0)


def lowest_point_m(calib: CalibrationStore, q) -> float:
    """Lowest z in base_link over the wrist links and the jaw-tip midpoint."""
    robot = load_urdf(calib.urdf_path())
    flange = robot.fk(q, target_link=TOOL_FRAME, base_link="base_link")
    tip_z = (flange[:3, :3] @ TCP_IN_FLANGE + flange[:3, 3])[2]
    return float(min([robot.fk(q, target_link=link, base_link="base_link")[2, 3] for link in HEIGHT_CHECK_LINKS]
                     + [tip_z]))


def allowed_bounds_rad(calib: CalibrationStore, motors: list[MotorState], margin: int = RANGE_MARGIN_TICKS):
    """Motor ranges shrunk by a margin, widened to include the current pose so joints may move away from a limit."""
    n_arm = calib.num_arm_joints(len(motors))
    arm = motors[:n_arm]
    shrunk = [MotorState(m.id, m.position, m.current_ma, m.range_min + margin, m.range_max - margin) for m in arm]
    bounds = calib.bounds_rad(shrunk)
    q_now = calib.ticks_to_rad(arm)
    lower = np.minimum([b[0] for b in bounds], q_now)
    upper = np.maximum([b[1] for b in bounds], q_now)
    return lower, upper


def plan_flange_move(calib: CalibrationStore, motors: list[MotorState], shift_m, rotation: Rotation,
                     rot_frame: str = "flange", limits: MotionLimits | None = None) -> dict:
    """Plan a straight-line flange move from the live pose and time it with a minimum-jerk profile.

    Parameters
    ----------
    calib : CalibrationStore
        Joint zeros, signs and URDF.
    motors : list of MotorState
        Live motor states (positions and calibrated ranges).
    shift_m : array_like
        Flange translation in base_link, metres.
    rotation : scipy.spatial.transform.Rotation
        Flange rotation increment.
    rot_frame : {"flange", "base"}
        Axes of the rotation increment.
    limits : MotionLimits, optional
        Pace and safety settings.

    Returns
    -------
    dict
        ``ok``, ``problems``, a JSON-friendly ``summary`` and, for execution, ``path`` (radians) and ``duration_s``.
    """
    limits = limits or MotionLimits()
    n_arm = calib.num_arm_joints(len(motors))
    q0 = np.array(calib.ticks_to_rad(motors))
    lower, upper = allowed_bounds_rad(calib, motors)
    shift_m = np.asarray(shift_m, dtype=float)
    plan = get_solver(str(calib.urdf_path())).plan_cartesian(q0, shift_m, rotation, rot_frame, lower, upper)
    problems = list(plan.problems)

    lowest = min(lowest_point_m(calib, q) for q in plan.path)
    # Starting below the floor (e.g. after sagging) is allowed as long as the move never goes lower.
    if lowest < min(limits.floor_cm / 100, lowest_point_m(calib, q0) - 0.002):
        problems.append(f"arm comes within {lowest * 100:.1f} cm of the table (floor {limits.floor_cm} cm)")

    joint_travel = np.abs(np.diff(plan.path, axis=0)).sum(axis=0) if len(plan.path) > 1 else np.zeros(n_arm)
    duration = max(limits.min_duration_s,
                   1.875 * float(joint_travel.max()) / math.radians(limits.joint_speed_deg_s),
                   float(np.linalg.norm(shift_m)) * 100 / limits.speed_cm_s,
                   math.degrees(rotation.magnitude()) / limits.rot_speed_deg_s)
    start_ticks = [m.position for m in motors[:n_arm]]
    goal_ticks = calib.rad_to_ticks(list(plan.path[-1]), motors)
    summary = dict(plan.summary)
    summary.update({
        "lowest_point_cm": round(lowest * 100, 1),
        "start_ticks": start_ticks,
        "goal_ticks": [goal_ticks[m.id] for m in motors[:n_arm]],
        "delta_ticks": [goal_ticks[m.id] - m.position for m in motors[:n_arm]],
        "duration_s": round(duration, 1),
        "peak_joint_speed_deg_s": round(math.degrees(1.875 * float(joint_travel.max()) / duration), 1),
    })
    return {"ok": not problems, "problems": problems, "summary": summary, "path": plan.path, "duration_s": duration}


async def stream_joint_path(bridge: StationBridge, calib: CalibrationStore, path: np.ndarray, duration: float,
                            limits: MotionLimits | None = None, gravity_ff: bool = True) -> dict:
    """Stream a joint path along a minimum-jerk profile; hold position on excessive tracking lag or error.

    Parameters
    ----------
    bridge : StationBridge
        Connected Station bridge.
    calib : CalibrationStore
        Radian/tick conversion.
    path : numpy.ndarray
        (N, 7) joint waypoints in radians, starting at the current pose.
    duration : float
        Trajectory duration in seconds.
    limits : MotionLimits, optional
        Streaming rate, lag limit and per-command step clamp.
    gravity_ff : bool
        Add the fitted gravity offsets (see `gravity_offset_rad`) to the commanded targets; lag is still
        measured against the uncompensated path.

    Returns
    -------
    dict
        ``ok``, cycles streamed, worst lag in ticks and the final ticks; ``aborted`` with a reason on failure.
    """
    limits = limits or MotionLimits()
    period = 1.0 / limits.rate_hz
    motors = (await bridge.get_state()).motors
    n_arm = calib.num_arm_joints(len(motors))
    arm_ids = [m.id for m in motors[:n_arm]]
    urdf = str(calib.urdf_path())
    direction = move_direction(path)
    commanded = np.array([q + gravity_offset_rad(q, urdf, direction) for q in path]) if gravity_ff else path

    def ticks_at(s: float, along=path) -> dict[int, int]:
        return calib.rad_to_ticks(list(_sample(along, s)), motors)

    start, cycles, worst_lag = time.monotonic(), 0, 0
    try:
        while (t := time.monotonic() - start) <= duration + period:
            await bridge.send_joint_targets(ticks_at(min_jerk((t + period) / duration), commanded),
                                            max_step=limits.max_step_ticks)
            present = (await bridge.get_state()).positions()
            expected = ticks_at(min_jerk(t / duration))
            lag = {mid: present[mid] - expected[mid] for mid in arm_ids}
            worst_lag = max(worst_lag, max(abs(v) for v in lag.values()))
            if max(abs(v) for v in lag.values()) > limits.lag_limit_ticks:
                hold = await bridge.abort()
                return {"ok": False, "aborted": f"lag {lag} at t={t:.2f}s", "hold": hold, "cycles": cycles}
            cycles += 1
            await asyncio.sleep(max(0.0, start + cycles * period - time.monotonic()))
        await bridge.send_joint_targets(ticks_at(1.0, commanded), max_step=limits.max_step_ticks)
        await bridge.wait_settle(ticks_at(1.0), tol_ticks=15, timeout_s=1.0)
    except BaseException as exc:
        log.warning("flange stream aborted: %r", exc)
        await bridge.abort()
        raise
    final = (await bridge.get_state()).positions()
    return {"ok": True, "cycles": cycles, "elapsed_s": round(time.monotonic() - start, 2), "worst_lag_ticks": worst_lag,
            "final_ticks": [final[mid] for mid in arm_ids]}


async def correct_final_pose(bridge: StationBridge, calib: CalibrationStore, path: np.ndarray,
                             limits: MotionLimits | None = None) -> dict:
    """Close the loop on the path's end pose: offset joint targets by their remaining error until the flange arrives.

    The servos are proportional position loops, so joints carrying gravity load settle short of their targets.
    Each iteration adds the measured joint error to the commanded target (bounded offset, inside the motor
    range) until the flange is within ``correct_tol_mm`` of the planned end pose.

    Parameters
    ----------
    bridge : StationBridge
        Connected Station bridge.
    calib : CalibrationStore
        Radian/tick conversion.
    path : numpy.ndarray
        Joint path in radians; its last row is the goal.
    limits : MotionLimits, optional
        Iteration count, tolerance and offset bound.

    Returns
    -------
    dict
        ``iterations``, flange error after each iteration and the commanded offsets in ticks.
    """
    limits = limits or MotionLimits()
    motors = (await bridge.get_state()).motors
    n_arm = calib.num_arm_joints(len(motors))
    arm = motors[:n_arm]
    goal = calib.rad_to_ticks(list(path[-1]), motors)
    offset = gravity_offset_rad(path[-1], str(calib.urdf_path()), move_direction(path))
    start = calib.rad_to_ticks(list(path[-1] + offset), motors)
    command = {m.id: start[m.id] for m in arm}
    history = []
    for iteration in range(limits.correct_iterations + 1):
        state = await bridge.get_state()
        err = flange_error(calib, path, calib.ticks_to_rad(state.motors))
        history.append(err["off_target_mm"])
        stalled = len(history) > 1 and history[-2] - history[-1] < 0.3
        if err["off_target_mm"] <= limits.correct_tol_mm or stalled or iteration == limits.correct_iterations:
            break
        present = state.positions()
        for m in arm:
            offset = command[m.id] - present[m.id]
            offset = int(np.clip(offset, -limits.correct_max_offset_ticks, limits.correct_max_offset_ticks))
            command[m.id] = int(np.clip(goal[m.id] + offset, m.range_min, m.range_max))
        await bridge.send_joint_targets(command, max_step=limits.max_step_ticks)
        await asyncio.sleep(0.5)
    return {"iterations": len(history) - 1, "off_target_mm": history,
            "offset_ticks": [command[m.id] - goal[m.id] for m in arm]}


def flange_error(calib: CalibrationStore, path: np.ndarray, final_rad) -> dict:
    """Moved distance and remaining error of the flange, from the reached joint angles."""
    solver = get_solver(str(calib.urdf_path()))
    T_start, T_goal, T_final = solver.fk(path[0]), solver.fk(path[-1]), solver.fk(final_rad)
    rot_off = (Rotation.from_matrix(T_goal[:3, :3]).inv() * Rotation.from_matrix(T_final[:3, :3])).magnitude()
    return {
        "flange_moved_cm": ((T_final[:3, 3] - T_start[:3, 3]) * 100).round(2).tolist(),
        "off_target_mm": round(float(np.linalg.norm(T_final[:3, 3] - T_goal[:3, 3]) * 1000), 1),
        "rotation_off_target_deg": round(math.degrees(rot_off), 2),
    }
