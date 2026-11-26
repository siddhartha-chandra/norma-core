"""DESK-CORE FastMCP Server — connects Claude Code and Codex to NormaCore Station.

Exposes:
  1. Bare-bone hardware primitives: observe, get_state, set_torque, move_joints, grasp, release, go_home,
     move_to_home, estop
  2. Core robotics primitives: move_joint, get_frame, solve_ik, get_joint_angles, get_metric_depth
  3. Calibration & scene tools: pixel_to_world, update_scene_obstacle, get_calib_status, reload_config
  4. Composite execution tools: move_to_pose, move_flange, pick_at, place_at, pick_pixel, execute_sequence
  5. Numbered wrist stereo workflow: detect_requested_object, aim_wrist, capture_wrist_stereo, estimate_wrist_depth,
     measure_graspability, estimate_grasp_pose, plan_tolerant_grasp, retract_to_hover, move_to_drop,
     master_pick_place
  6. Prompt skills (@mcp.prompt): desk_operator_skill, pick_and_place_skill

Every tool call is recorded under runs/<UTC timestamp>/ (calls.jsonl + frames/*.jpg)
so sessions are replayable; set DESK_RUNLOG=0 to disable. See desk_core/runlog.py.
"""
from __future__ import annotations

import asyncio
import atexit
import functools
import io
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from dotenv import dotenv_values
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from PIL import Image as PILImage
from pydantic import TypeAdapter

from .calib_store import CalibrationStore, TICKS_PER_RAD
from .hive import HIVE_SHAPE, POSITION_TOL_M, Layout, Obstacle, Tile, plan_hive
from .ik_adapter import IKAdapter
from .interfaces import SceneObstacle
from .robo_jev import ReflexMonitor, RoboJevGatekeeper, check_reflex_and_abort, guarded_recover
from .runlog import RunLogger
from .station_bridge import (DEFAULT_MAX_STEP, MAX_ANGLE_STEP, REPO_ROOT, LiveStationBridge, MockStationBridge,
                             PreflightWriteUncertain, RobotState, StationBridge, create_bridge)
from .tts import SpeakerTTS
from .command_planner import plan_command as _plan_command
from .vision_3d import localize_object_3d

if TYPE_CHECKING:
    from .main_control import MainControl

if os.environ.get("DESK_NO_DOTENV") != "1":
    for _k, _v in dotenv_values(REPO_ROOT / ".env").items():
        if _v is not None and not os.environ.get(_k, "").strip():
            os.environ[_k] = _v

_bridge: StationBridge | None = None
_connect_lock = asyncio.Lock()
_flange_lock = asyncio.Lock()
# Motion tools may call other motion tools in the same task. Independent calls share the lock.
## added by Abhi
_motion_owner: asyncio.Task | None = None
_estop_lock = asyncio.Lock()
_main_control: MainControl | None = None
_connect_cooldown_until = 0.0

_calib = CalibrationStore()
_ik = IKAdapter()
_obstacles: dict[str, SceneObstacle] = {}
_hive_tiles = TypeAdapter(list[Tile])
_robo_jev = RoboJevGatekeeper()
_tts = SpeakerTTS()

RUNS_DIR = Path(os.environ.get("DESK_RUNS_DIR") or (REPO_ROOT / "runs"))
FIXED_POSITIONS_PATH = REPO_ROOT / "calib" / "fixed_positions.json"
_logger: RunLogger | None = None
_frame_seq = 0


def _traced(fn):
    """Log every tool call (args, result, frames, duration) to the run's calls.jsonl."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        if _logger is None:
            return await fn(*args, **kwargs)
        t0 = time.perf_counter()
        call_args = dict(kwargs)
        if args:
            call_args["_positional"] = [repr(a) for a in args]
        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:
            _logger.record_call(fn.__name__, call_args, error=exc, duration_ms=1000 * (time.perf_counter() - t0))
            raise
        _logger.record_call(fn.__name__, call_args, result=result, duration_ms=1000 * (time.perf_counter() - t0))
        return result

    return wrapper


## added by Abhi
def _motion_tool(fn):
    """Serialize MCP motion operations while allowing nested calls in the same task."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        global _motion_owner
        task = asyncio.current_task()
        if _estop_lock.locked():
            raise RuntimeError("Emergency stop is in progress.")
        if _motion_owner is task and task is not None:
            return await fn(*args, **kwargs)
        if _flange_lock.locked():
            if fn.__name__ == "move_flange":
                return {"ok": False, "executed": False, "problems": ["another move_flange is still running"]}
            raise RuntimeError("Another desk-core motion operation is running.")
        async with _flange_lock:
            _motion_owner = task
            try:
                return await fn(*args, **kwargs)
            finally:
                _motion_owner = None

    return wrapper


async def _get_bridge() -> StationBridge:
    """Lazy bridge singleton that retries failed cold connects."""
    global _bridge, _connect_cooldown_until
    async with _connect_lock:
        if _bridge is not None:
            return _bridge
        if time.monotonic() < _connect_cooldown_until:
            raise RuntimeError("desk-core: Station still unreachable, retry in a few seconds.")
        ## added by Abhi
        bridge = create_bridge(follow_motor_rx=True)
        try:
            await bridge.connect()
        except Exception as exc:
            _connect_cooldown_until = time.monotonic() + 5.0
            mode = "mock" if (os.environ.get("DESK_MOCK") == "1" or not os.environ.get("STATION_HOST", "").strip()) else "live"
            raise RuntimeError(f"desk-core: {mode} bridge connect failed: {exc}") from exc
        _bridge = bridge
        return _bridge


def _decode_rgb_matrix(jpeg: bytes) -> np.ndarray:
    return np.asarray(PILImage.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.uint8)


async def get_frame_matrix(camera: str = "top", denoise: bool = True) -> np.ndarray:
    """Direct Python helper returning the raw (H, W, 3) uint8 RGB image matrix."""
    bridge = await _get_bridge()
    jpeg = await bridge.get_frame(camera, denoise=denoise)
    return _decode_rgb_matrix(jpeg)


mcp = FastMCP("desk-core")


# =============================================================================
# 1. Bare-Bone Hardware & Telemetry Primitives
# =============================================================================


### 1. Observe the robot and both cameras
# Existing MCP tool
@mcp.tool()
@_traced
async def observe(views: list[str] | None = None, denoise: bool = True) -> list:
    """Inspect camera feed(s) ('top', or 'wrist' = gripper eye-in-hand; there is no separate wrist camera) and live ST3215 joint telemetry in one round-trip."""
    bridge = await _get_bridge()
    if bridge.abort_source != "reflex":
        bridge.rearm()
    want_views = views or ["top", "wrist"]
    st = await bridge.get_state()
    summary = {
        "bus_serial": st.bus_serial,
        "torque_enabled": st.torque_enabled,
        "link": getattr(bridge, "link_status", lambda: {"bridge": "unknown"})(),
        "ik_backend": _ik.backend_name(),
        "cameras": await bridge.list_cameras(),
        "angles_rad": _calib.ticks_to_rad(st.motors),
        "joints": [
            {
                "id": m.id,
                "pos": m.position,
                "norm": round(m.to_norm(), 3),
                "range": [m.range_min, m.range_max],
                "current_ma": m.current_ma,
                "err": m.error_status,
            }
            for m in st.motors
        ],
    }
    out: list = [json.dumps(summary, indent=2)]
    for cam in want_views:
        try:
            jpeg = await bridge.get_frame(cam, denoise=denoise)
            out.append(Image(data=jpeg, format="jpeg"))
        except Exception as exc:
            out.append(f"[camera {cam!r} unavailable: {exc}]")
    return out


### 1. Read motor telemetry
# Existing MCP tool
@mcp.tool()
@_traced
async def get_state() -> dict:
    """Read live motor positions (raw ticks + normalized + radians), limits, and currents."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    return {
        "bus_serial": st.bus_serial,
        "torque_enabled": st.torque_enabled,
        "link_age_ms": st.link_age_ms,
        "angles_rad": _calib.ticks_to_rad(st.motors),
        "motors": [
            {
                "id": m.id,
                "position": m.position,
                "norm": round(m.to_norm(), 3),
                "range_min": m.range_min,
                "range_max": m.range_max,
                "current_ma": m.current_ma,
                "error_status": m.error_status,
            }
            for m in st.motors
        ],
    }


@mcp.tool()
@_traced
@_motion_tool
async def set_torque(enable: bool) -> dict:
    """Safely enable or disable motor torque (writes Goal=Present + speed/accel caps first)."""
    bridge = await _get_bridge()
    ok = await bridge.set_torque(enable)
    return {"ok": ok, "torque_enabled": enable}


@mcp.tool()
@_traced
@_motion_tool
async def move_joints(targets: dict[str, int], max_step: int = DEFAULT_MAX_STEP, wait: bool = True) -> dict:
    """Move joints to target raw encoder ticks (clamped to calibrated range & max_step per call)."""
    bridge = await _get_bridge()
    parsed = {int(k): int(v) for k, v in targets.items()}
    applied = await bridge.send_joint_targets(parsed, max_step=max_step)
    settled = await bridge.wait_settle(applied) if (wait and applied) else False
    return {"applied": applied, "settled": settled}


### 7. Close and verify the grasp
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def grasp(close_ticks: int | None = None) -> dict:
    """Close gripper jaws and verify grasp via jaw encoder slack + motor current."""
    bridge = await _get_bridge()
    from_default = False
    if close_ticks is None and _calib.data.get("calibrated"):
        grp = _calib.data.get("gripper") or {}
        if "close_ticks" in grp:
            close_ticks = int(grp["close_ticks"])
            from_default = True
    try:
        return await bridge.grasp_with_verify(close_ticks=close_ticks, clamp_to_range=from_default)
    except TypeError:
        return await bridge.grasp_with_verify(close_ticks=close_ticks)


### 7 and 8. Open the gripper before the pick and during placement
# Existing MCP tool
GRIPPER_OPEN_TOL_TICKS = 25


@mcp.tool()
@_traced
@_motion_tool
async def release() -> dict:
    """Open gripper jaws to calibrated open limit."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    ## added by Abhi
    grp = _calib.data.get("gripper") or {}
    gripper_id = int(grp.get("motor_id", _calib.num_arm_joints(len(st.motors)) + 1))
    gripper = next((m for m in st.motors if m.id == gripper_id), None)
    if gripper is None or gripper.error_status:
        return {"ok": False, "released": False, "gripper_id": gripper_id, "reason": "Missing or faulted gripper"}
    target_open = int(grp.get("open_ticks", gripper.range_max)) if _calib.data.get("calibrated") else gripper.range_max
    result = {"ok": False, "released": False, "released_object": None, "gripper_id": gripper_id, "target": target_open}
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        state = await bridge.get_state()
        motor = next((m for m in state.motors if m.id == gripper_id), None)
        if state.bus_serial != st.bus_serial or motor is None or motor.error_status:
            return {**result, "reason": "Gripper telemetry changed while opening"}
        if abs(motor.position - target_open) <= GRIPPER_OPEN_TOL_TICKS:
            released = bridge.release_mock_object() if hasattr(bridge, "release_mock_object") else None
            return {**result, "ok": True, "released": True, "released_object": released}
        before = motor.position
        applied = await bridge.send_joint_targets({gripper_id: target_open}, max_step=DEFAULT_MAX_STEP)
        step = applied.get(gripper_id)
        if step is None or step == before or not await bridge.wait_settle({gripper_id: step}, timeout_s=2.0):
            return {**result, "reason": "Gripper did not settle at the commanded open step"}
    return {**result, "reason": "Gripper did not reach the open target"}


### 9. Return to the legacy midpoint home
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def go_home(look_at_table: bool = True) -> dict:
    """Move arm safely toward calibrated home / table-viewing pose (creep-stepped if far)."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    n_arm = _calib.num_arm_joints(len(st.motors))
    arm_motors = st.motors[:n_arm]

    targets: dict[int, int] = {}
    if _calib.data.get("calibrated"):
        home_q = _calib.data.get("home_joints_rad")
        table_q = _calib.data.get("table_view_joints_rad") or [0.0, 0.4, 0.0, -1.1, 0.0, 0.9, 0.0]
        if look_at_table:
            # Prefer explicit table_view_joints_rad, or use table-viewing pose if home_joints_rad is all zeros
            if _calib.data.get("table_view_joints_rad"):
                q_list = list(table_q)
            elif isinstance(home_q, list) and len(home_q) == n_arm and any(abs(float(v)) > 0.05 for v in home_q):
                q_list = [float(v) for v in home_q]
            else:
                q_list = list(table_q[:n_arm])
        else:
            q_list = [float(v) for v in home_q] if isinstance(home_q, list) and len(home_q) == n_arm else [0.0] * n_arm

        for m, angle in zip(arm_motors, q_list):
            raw = _calib._zero_tick(m) + _calib._sign(m.id) * float(angle) * TICKS_PER_RAD
            tick = int(round(raw))
            valid_bounds = m.range_max > m.range_min and m.range_min <= m.position <= m.range_max
            lo, hi = (m.range_min, m.range_max) if valid_bounds else (0, 4095)
            targets[m.id] = max(lo, min(hi, tick))
    else:
        for m in arm_motors:
            valid_bounds = m.range_max > m.range_min and m.range_min <= m.position <= m.range_max
            targets[m.id] = (m.range_min + m.range_max) // 2 if valid_bounds else 2048

    applied = targets
    settled = False
    for _ in range(10):
        applied = await bridge.send_joint_targets(targets, max_step=300)
        settled = await bridge.wait_settle(applied, tol_ticks=40, timeout_s=2.0)
        cur = {m.id: m.position for m in (await bridge.get_state()).motors[:n_arm]}
        if all(abs(cur.get(mid, tgt) - tgt) <= 40 for mid, tgt in targets.items()):
            settled = True
            break

    if hasattr(bridge, "set_mock_ee_pose"):
        bridge.set_mock_ee_pose(0.0, 0.27, 0.25 if look_at_table else 0.22)
    return {"home": targets, "applied": applied, "look_at_table": look_at_table, "settled": settled}


### 9. Return to the calibrated saved home
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def move_to_home(max_step: int = DEFAULT_MAX_STEP, timeout_s: float = 30.0) -> dict:
    """Move the arm straight to the saved "home" joint ticks in calib/fixed_positions.json.

    No IK and no trajectory planning: arm joints are stepped toward the stored ticks, at most ``max_step``
    ticks per command, until every joint is within 40 ticks of home or ``timeout_s`` runs out. Joints move
    independently, so the path between poses is not collision-checked. The gripper is left as it is.
    """
    home = json.loads(FIXED_POSITIONS_PATH.read_text())["home"]
    bridge = await _get_bridge()
    ## added by Abhi
    if max_step <= 0 or not np.isfinite(timeout_s) or timeout_s <= 0:
        return {"ok": False, "settled": False, "reason": "Home motion needs a positive step and timeout"}
    max_step = min(max_step, DEFAULT_MAX_STEP)
    start = await bridge.get_state()
    motors = {m.id: m for m in start.motors}
    requested = {int(k): int(v) for k, v in home["joint_ticks"].items()}
    if not requested or not all(mid in motors and motors[mid].error_status == 0 for mid in requested):
        return {"ok": False, "settled": False, "reason": "Missing or faulted home motor", "home_ticks": requested}
    targets, clamped = {}, {}
    for mid, ticks in requested.items():
        m = motors[mid]
        lo, hi = m.range_min, m.range_max
        if hi <= lo:
            return {"ok": False, "settled": False, "reason": f"Invalid range for home motor {mid}"}
        targets[mid] = max(lo, min(hi, ticks))
        if targets[mid] != ticks:
            clamped[mid] = ticks
    deadline = time.monotonic() + timeout_s
    settled = False
    final = start.positions()
    while time.monotonic() < deadline:
        state = await bridge.get_state()
        final = state.positions()
        healthy = {m.id for m in state.motors if m.error_status == 0}
        if state.bus_serial != start.bus_serial or not targets.keys() <= healthy:
            break
        if all(abs(final[mid] - tgt) <= 40 for mid, tgt in targets.items()):
            settled = True
            break
        applied = await bridge.send_joint_targets(targets, max_step=max_step)
        if applied.keys() != targets.keys() or not await bridge.wait_settle(
            applied, tol_ticks=40, timeout_s=min(3.0, max(0.1, deadline - time.monotonic()))
        ):
            break
    return {"ok": settled, "settled": settled, "home_ticks": targets, "clamped_from": clamped,
            "final_ticks": {mid: final.get(mid) for mid in targets}}


### 1–9. Emergency stop on a hazard
# Existing MCP tool
@mcp.tool()
@_traced
async def estop() -> dict:
    """Emergency stop: freeze every motor immediately at its current encoder reading (or cut torque if stale)."""
    ## added by Abhi
    async with _estop_lock:
        owner = _motion_owner
        if owner is not None and owner is not asyncio.current_task():
            owner.cancel()
        bridge = await _get_bridge()
        return await bridge.abort()


# =============================================================================
# 2. Core Robotics Primitives (Angles, Frame Matrix, IK, Metric Depth)
# =============================================================================


### 6. Read the live IK seed
# Existing MCP tool
@mcp.tool()
@_traced
async def get_joint_angles(include_gripper: bool = False) -> list[float]:
    """Return current joint angles [j_1, ..., j_7] in radians from live ST3215 telemetry."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    angles = _calib.ticks_to_rad(st.motors)
    if include_gripper and len(st.motors) > len(angles):
        g = st.motors[-1]
        angles.append(round((g.position - 0.5 * (g.range_min + g.range_max)) / TICKS_PER_RAD, 6))
    return angles


MOVE_JOINT_SPEED_DEG_S = 25.0
MOVE_JOINT_MIN_DURATION_S = 1.0
MOVE_JOINT_LAG_LIMIT_TICKS = 120


### 7 and 8. Execute a validated joint solution
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def move_joint(
    joints: list[float] | None = None,
    j_1: float | None = None,
    j_2: float | None = None,
    j_3: float | None = None,
    j_4: float | None = None,
    j_5: float | None = None,
    j_6: float | None = None,
    j_7: float | None = None,
    wait: bool = True,
) -> bool:
    """Move arm to target joint angles [j_1, ..., j_7] (radians) in one smooth minimum-jerk stream. Returns True on arrival."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    n_arm = _calib.num_arm_joints(len(st.motors))

    if joints is not None:
        target_rad = [float(v) for v in joints]
    else:
        named = [j_1, j_2, j_3, j_4, j_5, j_6, j_7][:n_arm]
        if any(v is None for v in named):
            return False
        target_rad = [float(v) for v in named if v is not None]

    ok_lim, _reason = _calib.validate_joint_limits(target_rad, st.motors)
    if not ok_lim:
        return False

    from .motion import MotionLimits, move_joints_smooth

    limits = MotionLimits(joint_speed_deg_s=MOVE_JOINT_SPEED_DEG_S, min_duration_s=MOVE_JOINT_MIN_DURATION_S,
                          lag_limit_ticks=MOVE_JOINT_LAG_LIMIT_TICKS)
    if limits.out_of_bounds():
        return False
    try:
        run = await move_joints_smooth(bridge, _calib, target_rad, limits, reflex=_reach_reflex(bridge))
        return bool(run["ok"])
    except Exception:
        return False


### 2. Read a scene frame for target localization
# Existing MCP tool
@mcp.tool()
@_traced
async def get_frame(
    camera: str = "top",
    denoise: bool = True,
    as_matrix: bool = False,
    max_dim: int = 0,
    save_npy: bool = True,
) -> list:
    """Fetch camera frame as (H, W, 3) uint8 RGB matrix reference (.npy path + optional inline matrix) and Image."""
    global _frame_seq
    _frame_seq += 1
    bridge = await _get_bridge()
    jpeg = await bridge.get_frame(camera, denoise=denoise)
    arr = _decode_rgb_matrix(jpeg)

    frame_dir = (_logger.run_dir / "frames") if _logger is not None else (RUNS_DIR / "latest_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    npy_path = frame_dir / f"matrix_{_frame_seq:04d}_{camera}.npy"
    if save_npy:
        np.save(npy_path, arr)

    meta: dict[str, Any] = {
        "camera": camera,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "npy_path": str(npy_path) if save_npy else None,
    }
    if as_matrix:
        mat = arr
        if max_dim > 0 and max(arr.shape[0], arr.shape[1]) > max_dim:
            scale = max_dim / float(max(arr.shape[0], arr.shape[1]))
            new_w = max(1, int(round(arr.shape[1] * scale)))
            new_h = max(1, int(round(arr.shape[0] * scale)))
            mat = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            meta["matrix_shape"] = list(mat.shape)
        meta["matrix"] = mat.tolist()

    return [json.dumps(meta), Image(data=jpeg, format="jpeg")]


def _mujoco_planner(st: RobotState, obstacles: list[SceneObstacle], shoulder_torque_limit_nm: float,
                    gripper_open_fraction: float = 1.0):
    from .ik_mujoco import MuJoCoPlanner, PlanningError
    from .urdf_fk import DEFAULT_ARM_JOINTS

    if (_calib.world_frame() != "base_link" or tuple(_calib.joint_names()) != DEFAULT_ARM_JOINTS
            or _calib.num_arm_joints(len(st.motors)) != len(DEFAULT_ARM_JOINTS)
            or len(st.motors) != 8 or [m.id for m in st.motors] != list(range(1, 9))):
        raise PlanningError("MuJoCo requires the calibrated seven-joint ElRobot model")
    if any(not m.range_min <= m.position <= m.range_max or m.range_max <= m.range_min for m in st.motors):
        raise PlanningError("Invalid calibrated motor ranges or current ticks")
    return MuJoCoPlanner(_calib.urdf_path(), _calib.z_table(), obstacles, shoulder_torque_limit_nm,
                         _calib.bounds_rad(st.motors), gripper_open_fraction)


### 6. Solve the flange pose with the configured IK backend
# Existing MCP tool
@mcp.tool()
@_traced
async def solve_ik(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
    seed_joints_rad: list[float] | None = None,
    shoulder_torque_limit_nm: float | None = None,
) -> list[float]:
    """Solve a metric flange pose (xyzw). Supply a shoulder torque limit to use MuJoCo IK."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    ## added by Abhi
    if seed_joints_rad is None:
        seed_joints_rad = _calib.ticks_to_rad(st.motors)
    if shoulder_torque_limit_nm is not None:
        obstacles = list(_obstacles.values())

        def solve_in_sim() -> list[float]:
            planner = _mujoco_planner(st, obstacles, shoulder_torque_limit_nm)
            return planner.solve_ik([x, y, z], [qx, qy, qz, qw], seed_joints_rad).tolist()

        return await asyncio.to_thread(solve_in_sim)
    n_arm = _calib.num_arm_joints(len(st.motors))
    valid_bounds = all(m.range_max > m.range_min and m.range_min <= m.position <= m.range_max for m in st.motors[:n_arm])
    return await asyncio.to_thread(
        _ik.solve_ik,
        x=float(x),
        y=float(y),
        z=float(z),
        qx=float(qx),
        qy=float(qy),
        qz=float(qz),
        qw=float(qw),
        seed_joints_rad=seed_joints_rad,
        obstacles=list(_obstacles.values()),
        n_arm=n_arm,
        joint_bounds_rad=_calib.bounds_rad(st.motors) if valid_bounds else None,
    )


async def _live_arm_q(q_rad: list[float] | None, camera: str) -> list[float] | None:
    if q_rad is not None:
        return [float(v) for v in q_rad]
    cam_check = "wrist" if camera in ("both", "dual") else camera
    if f"T_world_cam_{cam_check}" in _calib.data or f"T_ee_cam_{cam_check}" not in _calib.data:
        return None
    bridge = await _get_bridge()
    st = await bridge.get_state()
    return _calib.ticks_to_rad(st.motors)


### 2. Read the existing single-frame depth approximation
# Existing MCP tool
@mcp.tool()
@_traced
async def get_metric_depth(x: float, y: float, camera: str = "top", q_rad: list[float] | None = None) -> float:
    """Estimate metric depth z (meters) at a 2D pixel coordinate in the specified camera."""
    q = await _live_arm_q(q_rad, camera)
    return _calib.get_metric_depth(float(x), float(y), camera=camera, q_rad=q)


# =============================================================================
# 3. Calibration, Geometry & Scene Obstacle Management
# =============================================================================


### 2. Project a scene pixel to the coarse targeting plane
# Existing MCP tool
@mcp.tool()
@_traced
async def pixel_to_world(
    x: float,
    y: float,
    z_m: float | None = None,
    camera: str = "top",
    q_rad: list[float] | None = None,
) -> dict:
    """Project 2D image pixel (x, y) into 3D metric (x, y, z) meters in robot base frame."""
    q = await _live_arm_q(q_rad, camera)
    wx, wy, wz = _calib.pixel_to_world(float(x), float(y), z_m=z_m, camera=camera, q_rad=q)
    depth_cam_z = _calib.get_metric_depth(float(x), float(y), camera=camera, q_rad=q)
    return {"x": wx, "y": wy, "z": wz, "cam_depth_z": depth_cam_z, "camera": camera}


### 2. Find the requested object in a camera frame
# Existing MCP tool
@mcp.tool()
@_traced
async def localize_3d(
    query: str,
    camera: str = "top",
    z_m: float | None = None,
    use_gemini: bool = True,
    denoise: bool = True,
    timeout_s: float = 6.0,
    provider: str = "auto",
    model: str | None = None,
) -> dict:
    """Zero-shot 2D object localization (Gemini/Anthropic/OpenAI/Qwen/DeepSeek with OpenCV fallback) on one camera. The 3D point uses a placeholder depth: the table plane (z_table) unless z_m is given; epipolar 2D->3D is not wired yet."""
    if camera in ("both", "dual"):
        raise ValueError("camera='both' is disabled: dual-camera 3D localization is pending the epipolar step; use 'top' or 'wrist'.")
    bridge = await _get_bridge()
    jpeg = await bridge.get_frame(camera, denoise=denoise)
    q = await _live_arm_q(None, camera)
    return await localize_object_3d(
        jpeg_bytes=jpeg,
        query=query,
        calib=_calib,
        camera=camera,
        q_rad=q,
        z_m=z_m,
        use_gemini=use_gemini,
        timeout_s=float(timeout_s),
        model=model,
        provider=provider,
    )


@mcp.tool()
@_traced
async def plan_command(command: str) -> dict:
    """Route a natural-language command with Jev: returns an execute_sequence plan for pick/place/home templates, or a non-motion route. Dry run; never moves the arm."""

    async def _localize(query: str) -> dict:
        return await localize_3d(query=query, camera="top")

    return await _plan_command(command, localize=_localize)


@mcp.tool()
@_traced
@_motion_tool
async def robo_jev_arm_watch(
    x: float,
    y: float,
    query: str = "target",
    camera: str = "top",
    depth_m: float | None = None,
) -> dict:
    """Arm a camera crop around the localized target before a reach. Use the same query for both cameras."""
    bridge = await _get_bridge()
    if camera not in ("top", "wrist"):
        raise ValueError("Robo-Jev camera must be 'top' or 'wrist'.")
    if bridge.abort_source == "reflex":
        raise ValueError("Recover the interrupted reach with robo_jev_guard before arming another watch.")
    if any(watch.query != query for watch in _robo_jev.watches.values()):
        _robo_jev.clear_watch()
    q_rad = await _live_arm_q(None, camera)
    jpeg = await bridge.get_frame(camera, denoise=False)
    pixel = float(x), float(y)
    d_m = float(depth_m) if depth_m is not None else _calib.get_metric_depth(*pixel, camera=camera, q_rad=q_rad)
    world = None
    if camera == "wrist" and bridge.link_status().get("bridge") == "live":
        if _calib.T_world_cam(camera, q_rad=q_rad) is None:
            raise ValueError("Wrist reflex tracking requires wrist camera extrinsics.")
        world = _calib.pixel_to_world(*pixel, camera=camera, q_rad=q_rad)
    return _robo_jev.arm_watch(jpeg, pixel, query=query, depth_m=d_m, camera=camera, world=world)


def _reach_reflex(bridge, target_xy=None) -> ReflexMonitor | None:
    if not _robo_jev.watches:
        return None
    if target_xy is None and _robo_jev.watch is not None:
        target_xy = _calib.pixel_to_world(*_robo_jev.watch.ref_pixel, camera="top")[:2]
    return ReflexMonitor(_robo_jev, bridge, _calib, lambda: list(_obstacles.values()), target_xy)


def _reflex_failure(bridge) -> dict:
    if bridge.abort_source == "reflex":
        return {"reflex": _robo_jev.interruption, "recovery_tool": "robo_jev_guard"}
    return {}


@mcp.tool()
@_traced
async def robo_jev_check(
    camera: str = "top",
    target_x: float | None = None,
    target_y: float | None = None,
    abort_on_hazard: bool = True,
) -> dict:
    """Run Robo-Jev <150 ms reflex checks (`target_shifted`, `path_clear`, `will_tip`, `grasp_secure`)."""
    bridge = await _get_bridge()
    target_xy = (float(target_x), float(target_y)) if (target_x is not None and target_y is not None) else None
    if abort_on_hazard:
        return await check_reflex_and_abort(
            gatekeeper=_robo_jev,
            bridge=bridge,
            calib=_calib,
            obstacles=list(_obstacles.values()),
            target_xy=target_xy,
            camera=camera,
        )
    jpeg = await bridge.get_frame(camera, denoise=False)
    return _robo_jev.evaluate(
        curr_jpeg=jpeg,
        calib=_calib,
        obstacles=list(_obstacles.values()),
        target_xy=target_xy,
        camera=camera,
    )


@mcp.tool()
@_traced
@_motion_tool
async def robo_jev_guard(
    query: str = "keys",
    target_x: float | None = None,
    target_y: float | None = None,
    camera: str = "top",
    obstacle_wait_timeout_s: float = 8.0,
    use_gemini: bool = False,
) -> dict:
    """Pause on obstacle until clear & resume; recalibrate on target shift if inside safe bounds."""
    bridge = await _get_bridge()
    target_xy = (float(target_x), float(target_y)) if (target_x is not None and target_y is not None) else None
    return await guarded_recover(
        gatekeeper=_robo_jev,
        bridge=bridge,
        calib=_calib,
        query=query,
        obstacles_fn=lambda: list(_obstacles.values()),
        target_xy=target_xy,
        camera=camera,
        obstacle_wait_timeout_s=float(obstacle_wait_timeout_s),
        use_gemini=use_gemini,
    )


@mcp.tool()
@_traced
async def speak_status(text: str) -> dict:
    """Speak a short status cue over the portable speaker (non-blocking TTS)."""
    queued = _tts.speak(text)
    return {"queued": queued, "tts": _tts.status()}


### 1. Register an obstacle before planning movement
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def update_scene_obstacle(
    name: str,
    x: float,
    y: float,
    z: float = 0.05,
    shape: str = "cylinder",
    radius_m: float = 0.045,
    height_m: float = 0.10,
    remove: bool = False,
    size_xyz: tuple[float, float, float] | None = None,
) -> dict:
    """Register or remove a 3D obstacle (e.g. mug, departure_tray) for IK/planner collision checks."""
    if remove:
        _obstacles.pop(name, None)
    else:
        _obstacles[name] = SceneObstacle(
            name=name, shape=shape, x=float(x), y=float(y), z=float(z),
            radius_m=float(radius_m), height_m=float(height_m), size_xyz=size_xyz,
        )
    return {"obstacles": sorted(_obstacles.keys())}


### 1. Check active calibration and IK backend
# Existing MCP tool
@mcp.tool()
@_traced
async def get_calib_status() -> dict:
    """Inspect active calibration files, Pink IK backend availability, and registered obstacles."""
    return {
        "calibration": _calib.status(),
        "ik_backend": _ik.backend_name(),
        "obstacles": [vars(o) for o in _obstacles.values()],
    }


@mcp.tool()
@_traced
@_motion_tool
async def reload_config() -> dict:
    """Hot-reload calib/rig_calib.json and the Pink IK module without restarting the MCP server."""
    ## added by Abhi
    global _main_control
    cal_status = _calib.reload()
    _main_control = None  # Captured poses use the calibration active when their images were taken.
    _ik.reload()
    if _bridge is not None and hasattr(_bridge, "apply_hardware_config"):
        hw = _calib.hardware_config()
        _bridge.apply_hardware_config(
            bus_serial=hw.get("bus_serial"),
            cam_top=hw.get("camera_top"),
            cam_wrist=hw.get("camera_wrist"),
        )
    return {"calibration": cal_status, "ik_backend": _ik.backend_name()}


@mcp.tool()
@_traced
@_motion_tool
async def reanchor_plane(
    pixel_points: list[list[float]] | None = None,
    world_points: list[list[float]] | None = None,
    z_table: float | None = None,
    max_error_cm: float = 1.0,
    allow_high_error: bool = False,
    clear: bool = False,
) -> dict:
    """Fast <60s tabletop re-anchor: fit 4+ point planar homography H_table, save to rig_calib.json, and hot-reload.

    Pass 4+ overhead camera pixels [[u, v], ...] and tabletop world coordinates [[x, y], ...] (meters),
    or omit world_points to reuse previously saved table_anchors. Pass clear=True to remove H_table.
    """
    from .homography import clear_plane_homography, reanchor_plane as _reanchor_plane

    global _main_control
    if clear:
        res = clear_plane_homography(calib_path=_calib.path)
        cal_status = _calib.reload()
        _main_control = None
        return {**res, "calibration": cal_status}

    if pixel_points is None:
        raise ValueError("pixel_points (4+ [u, v] coordinates) is required unless clear=True.")

    res = _reanchor_plane(
        pixel_points=pixel_points,
        world_points=world_points,
        z_table=z_table,
        calib_path=_calib.path,
        max_error_m=float(max_error_cm) / 100.0,
        allow_high_error=bool(allow_high_error),
    )
    cal_status = _calib.reload()
    _main_control = None
    return {**res, "calibration": cal_status}


# =============================================================================
# 4. Composite Execution Tools & Declarative Pipeline Runner
# =============================================================================


def _live_preflight_state(bridge: LiveStationBridge, state: RobotState) -> RobotState:
    from .ik_mujoco import PlanningError

    hardware = _calib.data.get("hardware") or {}
    bus = hardware.get("bus_serial")
    if not (_calib.data.get("calibrated") and _calib.data.get("arm_type") == "elrobot"
            and bus and bus == bridge.target_bus == state.bus_serial):
        raise PlanningError("Live preflight requires the bus named in the calibrated ElRobot rig")
    if not bridge._follow_motor_rx:
        raise PlanningError("Live preflight requires fresh raw motor telemetry")
    for name, low, high in (("joint_signs", -1, 1), ("joint_zero_ticks", 0, MAX_ANGLE_STEP)):
        values = _calib.data.get(name)
        if isinstance(values, dict):
            values = [values.get(str(mid)) for mid in range(1, 8)]
        if not isinstance(values, list) or len(values) < 7 or any(
            not isinstance(v, (int, float)) or not low <= v <= high or (name == "joint_signs" and abs(v) != 1)
            for v in values[:7]
        ):
            raise PlanningError(f"Explicit calibrated {name} required for all arm joints")
    if "z_table" not in _calib.data:
        raise PlanningError("Live preflight requires a calibrated table height")
    motors = []
    for m in state.motors:
        lo, hi = bridge._effective_motor_range(state.bus_serial, m)
        if m.error_status or (lo, hi) == (0, MAX_ANGLE_STEP):
            raise PlanningError(f"Motor {m.id} is faulted or lacks a calibrated range")
        motors.append(replace(m, range_min=lo, range_max=hi))
    gripper = _calib.data.get("gripper") or {}
    if (not motors or gripper.get("motor_id") != 8 or motors[-1].id != 8
            or gripper.get("open_ticks") != motors[-1].range_max
            or abs(motors[-1].position - motors[-1].range_max) > GRIPPER_OPEN_TOL_TICKS):
        raise PlanningError("Live preflight requires an open calibrated gripper; payloads are not modeled")
    bridge._require_fresh_motor_rx(state, state.positions())
    return replace(state, motors=motors)


@mcp.tool()
@_traced
@_motion_tool
async def plan_and_move(
    x: float,
    y: float,
    z: float,
    shoulder_torque_limit_nm: float,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
    execute: bool = False,
) -> dict:
    """Preflight a flange pose in MuJoCo. Live execution requires aligned, unloaded ElRobot hardware."""
    from .ik_mujoco import PlanningError
    from .motion import FLANGE_SETTLE_TOL_TICKS

    bridge = await _get_bridge()
    live = isinstance(bridge, LiveStationBridge)
    if execute and not (live or isinstance(bridge, MockStationBridge)):
        return {"ok": False, "executed": False, "problems": ["Unsupported Station bridge"]}
    if isinstance(bridge, MockStationBridge) and bridge.get_mock_scene_state()["held_object"]:
        return {"ok": False, "executed": False, "problems": ["Payloads are not modeled by the shoulder torque guard"]}
    st = await bridge.get_state()
    if live:
        try:
            st = _live_preflight_state(bridge, st)
        except (PlanningError, RuntimeError) as exc:
            return {"ok": False, "executed": False, "problems": [str(exc)]}
    initial_ticks = st.positions()
    motors = [replace(m) for m in st.motors]
    obstacles = list(_obstacles.values())

    def preflight():
        fraction = motors[-1].to_norm() if live else 1.0
        planner = _mujoco_planner(replace(st, motors=motors), obstacles, shoulder_torque_limit_nm, fraction)
        start = np.asarray(_calib.ticks_to_rad(motors))
        goal = planner.solve_ik([x, y, z], [qx, qy, qz, qw], start)

        def quantize(q):
            ticks = _calib.rad_to_ticks(q.tolist(), motors)
            return np.asarray(_calib.ticks_to_rad([replace(m, position=ticks.get(m.id, m.position)) for m in motors]))

        plan = planner.plan_trajectory(start, goal, quantize, sequential=live)
        targets = [_calib.rad_to_ticks(q.tolist(), motors) for q in plan.waypoints]
        if any(abs(second[mid] - tick) > DEFAULT_MAX_STEP for first, second in zip(targets, targets[1:])
               for mid, tick in first.items()):
            raise PlanningError("Preflight path exceeds Station's per-command step limit")
        return planner, plan, targets

    try:
        planner, plan, targets = await asyncio.to_thread(preflight)
    except PlanningError as exc:
        return {"ok": False, "executed": False, "problems": [str(exc)]}
    result = {"ok": True, "executed": False, "waypoints": len(targets),
              "tuck_end_waypoint": plan.tuck_end, "peak_shoulder_torque_nm": plan.peak_shoulder_torque_nm,
              "joints_rad": plan.waypoints[-1].tolist()}
    if not execute:
        return result
    current = await bridge.get_state()
    if current.bus_serial != st.bus_serial or current.positions() != initial_ticks or list(_obstacles.values()) != obstacles:
        return {"ok": False, "executed": False, "problems": ["Start pose or obstacles changed during preflight"]}

    async def stop_motion(problem: str, *, write_uncertain: bool = False) -> dict:
        result.update(ok=False, problems=[problem])
        if result["executed"] or write_uncertain:
            try:
                result["hold"] = await bridge.abort()
            except Exception as exc:
                result["problems"].append(f"Station hold failed: {exc}")
        return result

    expected = replace(st, motors=motors)
    measured = plan.waypoints[0]
    for index, planned_target in enumerate(targets[1:], start=1):
        target = planned_target
        try:
            if live:
                if plan.tuck_end is not None and index == plan.tuck_end + 1:
                    initial = plan.waypoints[0]
                    if (np.any(np.abs(measured[2:4] - initial[2:4]) < 1e-6)
                            or abs(planner.shoulder_torque(measured)) >= abs(planner.shoulder_torque(initial))):
                        raise PlanningError("Measured elbow/wrist tuck did not reduce shoulder torque")
                axis, = np.flatnonzero(plan.waypoints[index] != plan.waypoints[index - 1])
                mid = expected.motors[axis].id
                peak = planner.validate_segment(measured, plan.waypoints[index])
                result["peak_shoulder_torque_nm"] = max(peak, result["peak_shoulder_torque_nm"])
                if index > 1:
                    target = {mid: planned_target[mid]}
                applied = await bridge.send_joint_targets(target, max_step=DEFAULT_MAX_STEP, preflight_start=expected)
            else:
                applied = await bridge.send_joint_targets(target, max_step=DEFAULT_MAX_STEP)
        except PreflightWriteUncertain as exc:
            return await stop_motion(str(exc), write_uncertain=True)
        except (PlanningError, RuntimeError) as exc:
            return await stop_motion(str(exc))
        result["executed"] = True
        if applied != target:
            return await stop_motion("Station did not apply the preflighted waypoint")
        if live:
            try:
                settled = await bridge.wait_preflight_settle(
                    planned_target, tol_ticks=FLANGE_SETTLE_TOL_TICKS, moving_from={mid: expected.motors[axis].position}
                )
            except RuntimeError as exc:
                return await stop_motion(str(exc))
            if settled is None:
                return await stop_motion("Station did not settle near the preflighted waypoint")
            positions = settled.positions()
            if (settled.bus_serial != expected.bus_serial or positions.keys() != expected.positions().keys()
                    or positions[8] != initial_ticks[8]):
                return await stop_motion("Station bus, motor inventory or gripper changed during execution")
            expected = replace(expected, motors=[replace(m, position=positions[m.id]) for m in expected.motors])
            measured = np.asarray(_calib.ticks_to_rad(expected.motors))
            try:
                torque = planner.validate_segment(measured, measured)
            except PlanningError as exc:
                return await stop_motion(str(exc))
            result["peak_shoulder_torque_nm"] = max(torque, result["peak_shoulder_torque_nm"])
    if live:
        result["measured_joints_rad"] = measured.tolist()
        final = expected.positions()
        if any(abs(final[mid] - tick) > FLANGE_SETTLE_TOL_TICKS for mid, tick in targets[-1].items()):
            return await stop_motion("Station did not finish within the planned joint tolerance")
    if isinstance(bridge, MockStationBridge):
        bridge.set_mock_ee_pose(*planner.flange_pose(plan.waypoints[-1])[0])
    return result


### 7 and 8. Move to a flange pose through the configured IK backend
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def move_to_pose(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
) -> dict:
    """Move to a metric target position and xyzw orientation quaternion."""
    bridge = await _get_bridge()
    try:
        angles = await solve_ik(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)
    except (RuntimeError, ValueError):
        if not hasattr(bridge, "set_mock_ee_pose"):
            raise
        import math as _math
        st = await bridge.get_state()
        n_arm = _calib.num_arm_joints(len(st.motors))
        yaw = _math.atan2(float(y), max(0.05, float(x)))
        reach = min(1.0, ((float(x) ** 2 + float(y) ** 2) ** 0.5) / 0.5)
        angles = [round(yaw, 4), round(0.4 * reach, 4), round(-0.6 * reach, 4)] + [0.0] * max(0, n_arm - 3)
    ok = await move_joint(joints=angles, wait=True)
    if ok and hasattr(bridge, "set_mock_ee_pose"):
        bridge.set_mock_ee_pose(x, y, z)
    return {"ok": ok, "pose": [x, y, z, qx, qy, qz, qw], "joints_rad": angles,
            **(_reflex_failure(bridge) if not ok else {})}


### 3 and 4. Plan or execute a calibrated Cartesian flange move
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def move_flange(
    dx_cm: float = 0.0,
    dy_cm: float = 0.0,
    dz_cm: float = 0.0,
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
    rot_frame: str = "flange",
    execute: bool = False,
    speed_cm_s: float = 1.0,
    rot_speed_deg_s: float = 10.0,
    joint_speed_deg_s: float = 20.0,
    floor_cm: float = 8.0,
    min_duration_s: float = 2.0,
    lag_limit_ticks: int = 80,
    correct: bool = True,
) -> dict:
    """Move the gripper flange relative to its current pose along a straight line, smoothly (TRAC-IK).

    Translation is in base_link (cm); rotation (deg, x-y-z order) is about the flange origin, using the
    flange's own axes (rot_frame="flange", flange +Y is the gripper axis) or base_link axes ("base").
    Dry run by default: returns the plan. execute=True streams a minimum-jerk trajectory and holds
    position on tracking lag. Refuses plans that leave the motor ranges, flip configuration or come
    within floor_cm of the table. correct=True closes the loop at the end, offsetting joint targets until
    gravity-loaded joints actually reach the planned pose. Pace settings are bounded (max 6 cm/s, 90 deg/s
    joint speed, lag limit 120 ticks, min duration 1 s) and only one move_flange runs at a time.
    """
    from scipy.spatial.transform import Rotation

    from .motion import MotionLimits, flange_error, plan_flange_move, stream_joint_path

    if rot_frame not in ("flange", "base"):
        return {"ok": False, "problems": [f"rot_frame must be 'flange' or 'base', got {rot_frame!r}"]}
    limits = MotionLimits(speed_cm_s=speed_cm_s, rot_speed_deg_s=rot_speed_deg_s,
                          joint_speed_deg_s=joint_speed_deg_s, floor_cm=floor_cm,
                          min_duration_s=min_duration_s, lag_limit_ticks=lag_limit_ticks)
    if problems := limits.out_of_bounds():
        return {"ok": False, "executed": False, "problems": problems}
    bridge = await _get_bridge()
    st = await bridge.get_state()
    rotation = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True)
    plan = await asyncio.to_thread(plan_flange_move, _calib, st.motors,
                                   np.array([dx_cm, dy_cm, dz_cm]) / 100.0, rotation, rot_frame, limits)
    result = {"ok": plan["ok"], "executed": False, "problems": plan["problems"], "plan": plan["summary"]}
    if not plan["ok"] or not execute:
        return result
    run = await stream_joint_path(bridge, _calib, plan["path"], plan["duration_s"], limits,
                                  reflex=_reach_reflex(bridge), correct=correct)
    result.update({"ok": run["ok"], "executed": True, "run": run})
    if run["ok"]:
        if correct:
            result["correction"] = run.pop("correction")
        final_rad = _calib.ticks_to_rad((await bridge.get_state()).motors)
        result["result"] = flange_error(_calib, plan["path"], final_rad)
    return result


@mcp.tool()
@_traced
async def mock_add_object(
    name: str,
    x: float,
    y: float,
    z: float = 0.0,
    color: str = "blue",
    shape: str = "cube",
    radius_m: float = 0.025,
    insect: str | None = None,
) -> dict:
    """Add a synthetic 3D object in meters. Use shape='hive_tile' and set insect for a Hive tile."""
    bridge = await _get_bridge()
    if not hasattr(bridge, "add_mock_object"):
        return {"ok": False, "error": "mock_add_object is only available in --mock mode"}
    return {"ok": True, **bridge.add_mock_object(
        name=name, x=x, y=y, z=z, color=color, shape=shape, radius_m=radius_m, insect=insect,
    )}


@mcp.tool()
@_traced
async def mock_scene_state() -> dict:
    """Return ground-truth 3D positions of synthetic objects and end-effector in --mock mode."""
    bridge = await _get_bridge()
    if not hasattr(bridge, "get_mock_scene_state"):
        return {"ok": False, "error": "mock_scene_state is only available in --mock mode"}
    return {"ok": True, **bridge.get_mock_scene_state()}


@mcp.tool()
@_traced
async def mock_reset_scene() -> dict:
    """Reset the synthetic 3D tabletop scene in --mock mode."""
    bridge = await _get_bridge()
    if not hasattr(bridge, "reset_mock_scene"):
        return {"ok": False, "error": "mock_reset_scene is only available in --mock mode"}
    return {"ok": True, **bridge.reset_mock_scene()}


@mcp.tool()
@_traced
@_motion_tool
async def hive_sort(execute: bool = False, tiles: list[Tile] | None = None) -> dict:
    """Plan cream/black Hive piles by insect. Execute only in mock mode, using its tile inventory.

    For a dry run from measured coordinates, pass tiles with x, y and base z in meters.
    Use insect=null for unreadable tiles. Without tiles, read the mock scene inventory.
    """
    if execute and tiles is not None:
        return {"ok": False, "error": "Hive simulation execution reads tiles from the mock scene"}
    obstacles = [Obstacle(obj.x, obj.y, obj.radius_m) for obj in _obstacles.values()]
    mock = None
    if tiles is None:
        bridge = await _get_bridge()
        if not isinstance(bridge, MockStationBridge):
            return {"ok": False, "error": "Hive sorting requires the Station mock; measured tiles support dry runs only"}
        mock = bridge
        scene = bridge.get_mock_scene_state()
        if scene["held_object"] is not None:
            return {"ok": False, "error": "Release the held object before planning a Hive sort"}
        tiles = []
        for name, obj in scene["objects"].items():
            if obj["shape"] == HIVE_SHAPE:
                tiles.append(Tile(name, obj["color"], obj["insect"], obj["x"], obj["y"], obj["z"]))
            else:
                obstacles.append(Obstacle(obj["x"], obj["y"], obj["radius_m"]))
    try:
        plan = plan_hive(_hive_tiles.validate_python(tiles), Layout.load(), tuple(obstacles))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if mock is not None:
        for step in plan.moves:
            for rider in mock.get_mock_riders(step.name):
                if scene["objects"][rider]["shape"] != HIVE_SHAPE:
                    return {"ok": False, "error": f"Hive tile {step.name!r} is blocked by {rider!r}"}
    report = {"ok": True, "executed": execute, **asdict(plan), "completed": 0}
    if not execute or not plan.moves:
        return report

    observation = await observe()
    if any(isinstance(frame, str) for frame in observation[1:]):
        return {**report, "ok": False, "executed": False, "error": "Hive sort could not observe both mock cameras"}
    hover_m = 0.06
    for step in plan.moves:
        if not (await release()).get("released"):
            return {**report, "ok": False, "error": f"could not open gripper before picking {step.name}"}
        x, y, z = step.source
        bridge.set_mock_ee_pose(x, y, z + hover_m)
        bridge.set_mock_ee_pose(x, y, z)
        grip = await grasp()
        if not grip.get("ok") or not grip.get("holding") or grip.get("held_object") != step.name:
            return {**report, "ok": False, "error": f"could not pick {step.name}"}
        bridge.set_mock_ee_pose(x, y, z + hover_m)

        x, y, z = step.destination
        bridge.set_mock_ee_pose(x, y, z + hover_m)
        bridge.set_mock_ee_pose(x, y, z)
        released = await release()
        placed = bridge.get_mock_scene_state()["objects"][step.name]
        if not released.get("released") or released.get("released_object") != step.name or any(
            abs(placed[axis] - target) > POSITION_TOL_M for axis, target in zip(("x", "y", "z"), step.destination)
        ):
            return {**report, "ok": False, "error": f"could not place {step.name} at its planned position"}
        bridge.set_mock_ee_pose(x, y, z + hover_m)
        report["completed"] += 1
    return report


async def _resolve_grasp_pose(
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    dz: float = 0.06,
) -> tuple[float, float, float, list[float] | None, list[float] | None]:
    """Find reachable (grasp, hover) joint solutions for pick/place, falling back to radial tabletop orientations."""
    try:
        q_hover = await solve_ik(x=x, y=y, z=z + dz, qx=qx, qy=qy, qz=qz, qw=qw)
        q_grasp = await solve_ik(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)
        return (x, y, z, q_hover, q_grasp)
    except (RuntimeError, ValueError):
        pass

    import math as _math
    from scipy.spatial.transform import Rotation as _Rot

    yaw = _math.atan2(float(y), float(x)) - _math.pi / 2.0
    z_candidates = [float(z)] + ([float(z) + 0.06, float(z) + 0.08] if float(z) < 0.06 else [])
    pitches = (60.0, 45.0, 30.0, 75.0, 15.0, 0.0, 90.0, -15.0, -30.0, -45.0, -60.0)
    for z_try in z_candidates:
        q_grasp = None
        best_rot = None
        for pitch in pitches:
            for roll in (0.0, 180.0):
                rot = (
                    _Rot.from_euler("z", yaw)
                    * _Rot.from_euler("x", -pitch, degrees=True)
                    * _Rot.from_euler("y", roll, degrees=True)
                )
                cand_q = rot.as_quat()
                try:
                    q_grasp = await solve_ik(
                        x=x, y=y, z=z_try,
                        qx=float(cand_q[0]), qy=float(cand_q[1]),
                        qz=float(cand_q[2]), qw=float(cand_q[3]),
                    )
                    best_rot = cand_q
                    break
                except (RuntimeError, ValueError):
                    continue
            if q_grasp is not None:
                break
        if q_grasp is None:
            continue
        # Solve hover pose (preferring the same orientation and requested dz, falling back to dz=0.04 at workspace edge)
        bridge = await _get_bridge()
        st = await bridge.get_state()
        n_arm = _calib.num_arm_joints(len(st.motors))
        valid_bounds = all(m.range_max > m.range_min and m.range_min <= m.position <= m.range_max for m in st.motors[:n_arm])
        bounds = _calib.bounds_rad(st.motors) if valid_bounds else None
        hover_quats = [best_rot] + [
            (
                _Rot.from_euler("z", yaw)
                * _Rot.from_euler("x", -hp, degrees=True)
                * _Rot.from_euler("y", hr, degrees=True)
            ).as_quat()
            for hp in pitches
            for hr in (0.0, 180.0)
        ]
        dz_candidates = [dz] if dz <= 0.04 else [dz, 0.04]
        for dz_try in dz_candidates:
            for hq in hover_quats:
                try:
                    q_hover = _ik.solve_ik(
                        x=x, y=y, z=z_try + dz_try,
                        qx=float(hq[0]), qy=float(hq[1]),
                        qz=float(hq[2]), qw=float(hq[3]),
                        seed_joints_rad=q_grasp,
                        obstacles=list(_obstacles.values()),
                        n_arm=n_arm,
                        joint_bounds_rad=bounds,
                    )
                    return (x, y, z_try, q_hover, q_grasp)
                except (RuntimeError, ValueError):
                    continue
    return (x, y, z, None, None)


### 7. Execute a grasp at an already validated pose
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def pick_at(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.7071067811865476,
    qz: float = 0.0,
    qw: float = 0.7071067811865476,
    approach_dz: float = 0.06,
) -> dict:
    """Composite pick: open jaws -> hover above -> descend -> grasp_with_verify -> retract cleanly."""
    bridge = await _get_bridge()
    rx, ry, rz, q_hover, q_grasp = await _resolve_grasp_pose(x, y, z, qx, qy, qz, qw, dz=approach_dz)
    ## added by Abhi
    if not (opened := await release()).get("released"):
        return {"ok": False, "stage": "open", "release": opened}
    if q_hover is not None and q_grasp is not None:
        if not await move_joint(joints=q_hover, wait=True):
            return {"ok": False, "stage": "hover", "error": "move_joint(hover) failed", **_reflex_failure(bridge)}
        if not await move_joint(joints=q_grasp, wait=True):
            return {"ok": False, "stage": "descend", "error": "move_joint(descend) failed", **_reflex_failure(bridge)}
        if hasattr(bridge, "set_mock_ee_pose"):
            bridge.set_mock_ee_pose(rx, ry, rz)
        grip = await grasp()
        _robo_jev.clear_watch()
        if not grip.get("ok"):
            return {"ok": False, "stage": "close", "grasp": grip, "retract_ok": False}
        retract_ok = await move_joint(joints=q_hover, wait=True)
        if retract_ok and hasattr(bridge, "set_mock_ee_pose"):
            bridge.set_mock_ee_pose(rx, ry, rz + approach_dz)
        return {"ok": bool(grip.get("holding")) and bool(retract_ok), "grasp": grip, "retract_ok": retract_ok}

    hover = await move_to_pose(x=rx, y=ry, z=rz + approach_dz, qx=qx, qy=qy, qz=qz, qw=qw)
    if not hover["ok"]:
        return {"ok": False, "stage": "hover", **hover}
    descend = await move_to_pose(x=rx, y=ry, z=rz, qx=qx, qy=qy, qz=qz, qw=qw)
    if not descend["ok"]:
        return {"ok": False, "stage": "descend", **descend}
    grip = await grasp()
    _robo_jev.clear_watch()
    ## added by Abhi
    if not grip.get("ok"):
        return {"ok": False, "stage": "close", "grasp": grip, "retract_ok": False}
    retract = await move_to_pose(x=rx, y=ry, z=rz + approach_dz, qx=qx, qy=qy, qz=qz, qw=qw)
    return {"ok": bool(grip.get("holding")) and bool(retract.get("ok")), "grasp": grip, "retract_ok": retract["ok"]}


### 8. Place at an already validated release pose
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def place_at(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.7071067811865476,
    qz: float = 0.0,
    qw: float = 0.7071067811865476,
    retract_dz: float = 0.06,
    retract_home: bool = True,
) -> dict:
    """Composite place: hover above tray/drop -> lower -> release jaws -> retract -> return to resting pose."""
    rx, ry, rz, q_hover, q_grasp = await _resolve_grasp_pose(x, y, z, qx, qy, qz, qw, dz=retract_dz)
    if q_hover is not None and q_grasp is not None:
        bridge = await _get_bridge()
        if not await move_joint(joints=q_hover, wait=True):
            return {"ok": False, "stage": "hover", "error": "move_joint(hover) failed"}
        if not await move_joint(joints=q_grasp, wait=True):
            return {"ok": False, "stage": "lower", "error": "move_joint(lower) failed"}
        if hasattr(bridge, "set_mock_ee_pose"):
            bridge.set_mock_ee_pose(rx, ry, rz)
        rel = await release()
        if not rel.get("released"):
            return {"ok": False, "stage": "release", "release": rel, "retract_ok": False}
        retract_ok = await move_joint(joints=q_hover, wait=True)
        if retract_ok and hasattr(bridge, "set_mock_ee_pose"):
            bridge.set_mock_ee_pose(rx, ry, rz + retract_dz)
        if retract_home and retract_ok:
            await go_home()
        return {"ok": bool(rel.get("released")) and bool(retract_ok), "release": rel, "retract_ok": retract_ok}

    hover = await move_to_pose(x=rx, y=ry, z=rz + retract_dz, qx=qx, qy=qy, qz=qz, qw=qw)
    if not hover["ok"]:
        return {"ok": False, "stage": "hover", **hover}
    lower = await move_to_pose(x=rx, y=ry, z=rz, qx=qx, qy=qy, qz=qz, qw=qw)
    if not lower["ok"]:
        return {"ok": False, "stage": "lower", **lower}
    rel = await release()
    ## added by Abhi
    if not rel.get("released"):
        return {"ok": False, "stage": "release", "release": rel, "retract_ok": False}
    retract = await move_to_pose(x=rx, y=ry, z=rz + retract_dz, qx=qx, qy=qy, qz=qz, qw=qw)
    if retract_home and retract["ok"]:
        await go_home()
    return {"ok": bool(rel.get("released")) and bool(retract.get("ok")), "release": rel, "retract_ok": retract["ok"]}


### 7. Legacy single-frame pick shortcut
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def pick_pixel(
    x: float,
    y: float,
    camera: str = "top",
    qx: float = 0.0,
    qy: float = 0.7071067811865476,
    qz: float = 0.0,
    qw: float = 0.7071067811865476,
) -> dict:
    """Composite vision-to-grasp: pixel (x, y) -> get_metric_depth -> pixel_to_world -> pick_at."""
    depth_z = await get_metric_depth(x=x, y=y, camera=camera)
    world = await pixel_to_world(x=x, y=y, camera=camera)
    await robo_jev_arm_watch(x=x, y=y, camera=camera, depth_m=depth_z)
    res = await pick_at(
        x=world["x"], y=world["y"], z=world["z"], qx=qx, qy=qy, qz=qz, qw=qw
    )
    return {"pixel": [x, y], "depth_z": depth_z, "world": world, **res}


def _resolve_refs(obj: Any, env: dict[str, Any]) -> Any:
    if isinstance(obj, str) and obj.startswith("$"):
        key, _, field = obj[1:].partition(".")
        val = env[key]
        return val[field] if field and isinstance(val, dict) else val
    if isinstance(obj, list):
        return [_resolve_refs(el, env) for el in obj]
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, env) for k, v in obj.items()}
    return obj


## added by Abhi
async def _get_main_control() -> MainControl:
    """Use the existing bridge and calibration for all numbered workflow stages."""
    from .main_control import MainControl

    global _main_control
    bridge = await _get_bridge()
    if _main_control is None or _main_control.bridge is not bridge:
        _main_control = MainControl(bridge, _calib, {
            "observe": observe,
            "localize_3d": localize_3d,
            "update_scene_obstacle": update_scene_obstacle,
            "solve_ik": solve_ik,
            "grasp": grasp,
            "release": release,
            "estop": estop,
            "robo_jev_arm_watch": robo_jev_arm_watch,
        }, _obstacles, _robo_jev)
    return _main_control


## added by Abhi
def _target_coordinates(target: list[float] | dict[str, float]) -> list[float]:
    """Use a localization result's world field or explicit XYZ in meters."""
    return [target[key] for key in ("x", "y", "z")] if isinstance(target, dict) else target


### 2. Detect the requested object in the scene camera
## added by Abhi
@mcp.tool()
@_traced
async def detect_requested_object(query: str, provider: str = "auto") -> dict:
    """Detect a named object in the top view and return a coarse target in base_link for aim_wrist."""
    controller = await _get_main_control()
    await controller.observe()
    return await controller.detect_requested_object(query, provider=provider)


### 3. Aim the wrist camera at the localized object
## added by Abhi
@mcp.tool()
@_traced
@_motion_tool
async def aim_wrist(target_base_m: list[float] | dict[str, float], execute: bool = False) -> dict:
    """Plan a wrist viewing pose from XYZ or localize_3d's world field. Pass execute=True to move."""
    controller = await _get_main_control()
    await controller.observe()
    return await controller.aim_wrist(_target_coordinates(target_base_m), execute=execute)


### 4. Capture both settled wrist frames for temporal stereo
## added by Abhi
@mcp.tool()
@_traced
@_motion_tool
async def capture_wrist_stereo(target_base_m: list[float] | dict[str, float], execute: bool = False) -> dict:
    """Plan one sweep at the target. Pass execute=True to capture both views and receive a capture_id."""
    controller = await _get_main_control()
    await controller.observe()
    return await controller.capture_wrist_stereo(_target_coordinates(target_base_m), execute=execute)


### 4. Estimate metric depth from the captured wrist images
## added by Abhi
@mcp.tool()
@_traced
async def estimate_wrist_depth(capture_id: str, roi: list[int] | None = None, build_cloud: bool = False) -> dict:
    """Triangulate a captured wrist sweep. ROI is [x, y, width, height] in image A. Dense output is optional."""
    controller = await _get_main_control()
    return await controller.estimate_wrist_depth(capture_id, roi=roi, build_cloud=build_cloud)


### 5. Measure the object's footprint and check the jaw gap
## added by Abhi
@mcp.tool()
@_traced
async def measure_graspability(capture_id: str, jaw_opening_mm: float, clearance_mm: float = 2.0) -> dict:
    """Check the captured object's footprint against the measured jaw opening in millimeters."""
    controller = await _get_main_control()
    return controller.measure_graspability(capture_id, jaw_opening_mm=jaw_opening_mm, clearance_mm=clearance_mm)


### 6. Estimate the calibrated grasp pose for IK
## added by Abhi
@mcp.tool()
@_traced
async def estimate_grasp_pose(
    capture_id: str,
    tool: dict[str, Any] | None = None,
    jaw_offset_deg: float = 90.0,
    grasp_drop_m: float | None = None,
) -> dict:
    """Convert the estimated object surface into an IK target using measured grasp tool calibration."""
    controller = await _get_main_control()
    selected_tool = _calib.data.get("grasp_tool") if tool is None else tool
    if not isinstance(selected_tool, dict):
        return {"ok": False, "reason": "Provide measured tool calibration or configure grasp_tool in the active rig."}
    return controller.estimate_grasp_pose(
        capture_id, tool=selected_tool, jaw_offset_deg=jaw_offset_deg, grasp_drop_m=grasp_drop_m
    )


### 6. Plan a tolerant top-down grasp from the measured 3D pose and planar yaw
@mcp.tool()
@_traced
async def plan_tolerant_grasp(capture_id: str, tool: dict[str, Any] | None = None) -> dict:
    """Plan grasp and hover joints for a captured object after estimate_wrist_depth. Used by master_pick_place.

    The wide jaws self-centre the object, so the jaw tips may sit off-centre along the closing axis and the jaw
    heading may deviate within the slack the measured footprint allows. Returns ``q_hover`` and ``q_contact``
    joint angles plus the flange ``ik_target``. Planning only, the arm does not move.
    """
    controller = await _get_main_control()
    selected_tool = _calib.data.get("grasp_tool") if tool is None else tool
    if not isinstance(selected_tool, dict):
        return {"ok": False, "reason": "Provide measured tool calibration or configure grasp_tool in the active rig."}
    return await controller.plan_tolerant_grasp(capture_id, tool=selected_tool)


### 7. Retract the closed gripper to the planned pickup hover
@mcp.tool()
@_traced
@_motion_tool
async def retract_to_hover() -> dict:
    """After closing the jaws on the object, move back to the hover from the last plan_tolerant_grasp.

    Runs through the controller's checked joint path (table clearance for the jaw geometry) and then verifies
    the gripper still holds. Call before move_to_drop.
    """
    controller = await _get_main_control()
    return await controller.retract_to_hover()


### 8. Move to the saved drop X with stored joint ticks
@mcp.tool()
@_traced
@_motion_tool
async def move_to_drop() -> dict:
    """Move to the drop pose above the saved X using joint_ticks in calib/drop_location.json (no IK).

    The joint path is checked against the table for the jaw geometry before streaming. The gripper is not
    opened; call release() afterwards to drop the object.
    """
    controller = await _get_main_control()
    return await controller.move_to_drop()


### 10. Run the complete request-to-pick-and-place workflow
## added by Abhi
@mcp.tool()
@_traced
@_motion_tool
async def master_pick_place(
    query: str,
    execute: bool = False,
    tool: dict[str, Any] | None = None,
    obstacles: list[dict[str, Any]] | None = None,
    provider: str = "auto",
    build_cloud: bool = False,
    return_home: bool = True,
) -> dict:
    """Find an object, estimate its grasp from wrist stereo, and place at saved X. Dry run unless execute=True."""
    controller = await _get_main_control()
    return await controller.master_pick_place(
        query, execute=execute, tool=tool, obstacles=obstacles, provider=provider, build_cloud=build_cloud,
        return_home=return_home,
    )


_SEQ_TOOLS = {
    "observe": observe,
    "get_state": get_state,
    "set_torque": set_torque,
    "move_joints": move_joints,
    "move_joint": move_joint,
    "get_joint_angles": get_joint_angles,
    "get_frame": get_frame,
    "solve_ik": solve_ik,
    "get_metric_depth": get_metric_depth,
    "pixel_to_world": pixel_to_world,
    "localize_3d": localize_3d,
    "robo_jev_arm_watch": robo_jev_arm_watch,
    "robo_jev_check": robo_jev_check,
    "robo_jev_guard": robo_jev_guard,
    "speak_status": speak_status,
    "update_scene_obstacle": update_scene_obstacle,
    "get_calib_status": get_calib_status,
    "reload_config": reload_config,
    "reanchor_plane": reanchor_plane,
    "grasp": grasp,
    "release": release,
    "go_home": go_home,
    "move_to_home": move_to_home,
    "estop": estop,
    "move_to_pose": move_to_pose,
    "plan_and_move": plan_and_move,
    "move_flange": move_flange,
    "pick_at": pick_at,
    "place_at": place_at,
    "pick_pixel": pick_pixel,
    "hive_sort": hive_sort,
    ## added by Abhi
    "detect_requested_object": detect_requested_object,
    "aim_wrist": aim_wrist,
    "capture_wrist_stereo": capture_wrist_stereo,
    "estimate_wrist_depth": estimate_wrist_depth,
    "measure_graspability": measure_graspability,
    "estimate_grasp_pose": estimate_grasp_pose,
    "plan_tolerant_grasp": plan_tolerant_grasp,
    "retract_to_hover": retract_to_hover,
    "move_to_drop": move_to_drop,
    "master_pick_place": master_pick_place,
}


### 1–9. Pass values between MCP tools in an explicit sequence
# Existing MCP tool
@mcp.tool()
@_traced
@_motion_tool
async def execute_sequence(steps: list[dict[str, Any]], stop_on_false: bool = True) -> dict:
    """Execute a composite multi-tool pipeline in one MCP call with $var bindings across steps.

    Example step: {"tool": "solve_ik", "args": {"x": 0.25, "y": 0.0, "z": 0.05}, "save_as": "q"}
    Next step:    {"tool": "move_joint", "args": {"joints": "$q"}}
    """
    env: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        name = step.get("tool", "")
        fn = _SEQ_TOOLS.get(name)
        if fn is None:
            return {"ok": False, "failed_step": idx, "error": f"Unknown/disallowed tool: {name!r}", "trace": trace}
        raw_args = step.get("args") or {}
        args = _resolve_refs(raw_args, env)
        try:
            res = await fn(**args)
        except Exception as exc:
            return {"ok": False, "failed_step": idx, "tool": name, "error": repr(exc), "trace": trace}

        save_as = step.get("save_as")
        if save_as:
            env[str(save_as)] = res
        ok_step = not (res is False or (isinstance(res, dict) and res.get("ok") is False))
        # Omit large image objects from trace summary
        summary = f"list[{len(res)}]" if (name in {"observe", "get_frame"} and isinstance(res, list)) else res
        trace.append({"step": idx, "tool": name, "ok": ok_step, "result": summary})
        if stop_on_false and not ok_step:
            return {"ok": False, "failed_step": idx, "tool": name, "trace": trace}
    return {"ok": True, "steps": len(steps), "trace": trace}


# =============================================================================
# 5. Agent Prompt Skills (@mcp.prompt)
# =============================================================================


@mcp.prompt()
def desk_operator_skill() -> str:
    """Standard operating procedure for Claude Code / Codex driving DESK-CORE."""
    return (
        "1. Always start with `observe()` to inspect the top/wrist feeds and `link` freshness.\n"
        "2. Check `get_calib_status()` so you know if Pink IK and rig calibration are loaded.\n"
        "3. Register obstacles (e.g. mug, departure_tray) via `update_scene_obstacle(...)` before moving.\n"
        "4. Prefer composite tools (`move_to_pose`, `pick_at`, `place_at`, `execute_sequence`) or "
        "`solve_ik -> move_joint -> grasp` over raw `move_joints` tick guesses.\n"
        "5. Before a localized reach, call `robo_jev_arm_watch` on the top target pixel and the wrist target pixel "
        "when visible in both views, using the same query.\n"
        "6. If anything shifts or stalls, call `estop()` immediately and re-`observe()`. "
        "If Robo-Jev has already latched the emergency stop, re-`observe()` without replacing the reflex latch "
        "with a manual stop. Call `robo_jev_guard` and stop if it fails. "
        "Use its `world` result if it re-localized the target; otherwise obtain fresh world coordinates via "
        "`localize_3d` before planning. Use its coordinates only if it reports `found: true`; otherwise stay stopped. "
        "Plan from the stopped pose; never resume the interrupted trajectory."
    )


### 1–9. Direct Claude to the complete wrist stereo workflow
# Existing MCP prompt
## added by Abhi
@mcp.prompt()
def pick_and_place_skill(target_name: str, destination_name: str = "drop_x") -> str:
    """Guided camera-to-grasp workflow for picking a messy desk item and placing it cleanly."""
    return (
        f"Plan the pick of `{target_name}` for the requested destination `{destination_name}`.\n"
        "1. Call `observe(views=['top', 'wrist'])` and check follower telemetry.\n"
        "2. Register visible tabletop obstacles with `update_scene_obstacle` before moving.\n"
        f"3. Call `master_pick_place(query={target_name!r})` for a dry run. The master's destination is the saved X (`drop_x`).\n"
        "4. For an authorized live pick to X with verified tool calibration, call the same tool with `execute=True`. "
        "If Robo-Jev stops the reach, call `robo_jev_guard` and run a fresh localization and plan only if it succeeds. "
        "Use the returned status to report absence, uncertainty, an aborted stage, or successful placement."
    )


def _init_run_logger() -> None:
    global _logger
    if os.environ.get("DESK_RUNLOG", "1") == "0":
        return
    host = "" if os.environ.get("DESK_MOCK") == "1" else os.environ.get("STATION_HOST", "").strip()
    hw = _calib.hardware_config()
    _logger = RunLogger(
        RUNS_DIR,
        meta={
            "mode": "live" if host else "mock",
            "station": f"{host}:{os.environ.get('STATION_PORT', '8888')}" if host else None,
            "bus_serial": hw.get("bus_serial") or None,
            "camera_top": hw.get("camera_top") or None,
            "camera_wrist": hw.get("camera_wrist") or None,
            "pid": os.getpid(),
        },
    )
    atexit.register(_logger.close)


_init_run_logger()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
