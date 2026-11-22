"""DESK-CORE FastMCP Server — connects Claude Code and Codex to NormaCore Station.

Exposes:
  1. Bare-bone hardware primitives: observe, get_state, set_torque, move_joints, grasp, release, go_home, estop
  2. Core robotics primitives: move_joint, get_frame, solve_ik, get_joint_angles, get_metric_depth
  3. Calibration & scene tools: pixel_to_world, update_scene_obstacle, get_calib_status, reload_config
  4. Composite execution tools: move_to_pose, move_flange, pick_at, place_at, pick_pixel, execute_sequence
  5. Prompt skills (@mcp.prompt): desk_operator_skill, pick_and_place_skill, cleanup_skill

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
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import dotenv_values
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from PIL import Image as PILImage

from .calib_store import CalibrationStore, TICKS_PER_RAD
from .ik_adapter import IKAdapter
from .interfaces import SceneObstacle
from .robo_jev import RoboJevGatekeeper, check_reflex_and_abort, guarded_recover
from .runlog import RunLogger
from .station_bridge import DEFAULT_MAX_STEP, REPO_ROOT, StationBridge, create_bridge
from .tts import SpeakerTTS
from .vision_3d import localize_object_3d, localize_object_3d_dual

if os.environ.get("DESK_NO_DOTENV") != "1":
    for _k, _v in dotenv_values(REPO_ROOT / ".env").items():
        if _v is not None and not os.environ.get(_k, "").strip():
            os.environ[_k] = _v

_bridge: StationBridge | None = None
_connect_lock = asyncio.Lock()
_flange_lock = asyncio.Lock()
_connect_cooldown_until = 0.0

_calib = CalibrationStore()
_ik = IKAdapter()
_obstacles: dict[str, SceneObstacle] = {}
_robo_jev = RoboJevGatekeeper()
_tts = SpeakerTTS()

RUNS_DIR = Path(os.environ.get("DESK_RUNS_DIR") or (REPO_ROOT / "runs"))
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


async def _get_bridge() -> StationBridge:
    """Lazy bridge singleton that retries failed cold connects."""
    global _bridge, _connect_cooldown_until
    async with _connect_lock:
        if _bridge is not None:
            return _bridge
        if time.monotonic() < _connect_cooldown_until:
            raise RuntimeError("desk-core: Station still unreachable, retry in a few seconds.")
        bridge = create_bridge()
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


@mcp.tool()
@_traced
async def observe(views: list[str] | None = None, denoise: bool = True) -> list:
    """Inspect camera feed(s) ('top', or 'wrist' = gripper eye-in-hand; there is no separate wrist camera) and live ST3215 joint telemetry in one round-trip."""
    bridge = await _get_bridge()
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
async def set_torque(enable: bool) -> dict:
    """Safely enable or disable motor torque (writes Goal=Present + speed/accel caps first)."""
    bridge = await _get_bridge()
    ok = await bridge.set_torque(enable)
    return {"ok": ok, "torque_enabled": enable}


@mcp.tool()
@_traced
async def move_joints(targets: dict[str, int], max_step: int = DEFAULT_MAX_STEP, wait: bool = True) -> dict:
    """Move joints to target raw encoder ticks (clamped to calibrated range & max_step per call)."""
    bridge = await _get_bridge()
    parsed = {int(k): int(v) for k, v in targets.items()}
    applied = await bridge.send_joint_targets(parsed, max_step=max_step)
    settled = await bridge.wait_settle(applied) if (wait and applied) else False
    return {"applied": applied, "settled": settled}


@mcp.tool()
@_traced
async def grasp(close_ticks: int | None = None) -> dict:
    """Close gripper jaws and verify grasp via jaw encoder slack + motor current."""
    bridge = await _get_bridge()
    if close_ticks is None and _calib.data.get("calibrated"):
        grp = _calib.data.get("gripper") or {}
        if "close_ticks" in grp:
            close_ticks = int(grp["close_ticks"])
    return await bridge.grasp_with_verify(close_ticks=close_ticks)


@mcp.tool()
@_traced
async def release() -> dict:
    """Open gripper jaws to calibrated open limit."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    gripper = st.motors[-1]
    target_open = gripper.range_max
    if _calib.data.get("calibrated"):
        grp = _calib.data.get("gripper") or {}
        if "open_ticks" in grp:
            target_open = int(grp["open_ticks"])
    applied = await bridge.send_joint_targets({gripper.id: target_open}, max_step=1200)
    await bridge.wait_settle(applied, timeout_s=2.0)
    released_obj = bridge.release_mock_object() if hasattr(bridge, "release_mock_object") else None
    return {"released": True, "released_object": released_obj, "gripper_id": gripper.id, "target": applied.get(gripper.id)}


@mcp.tool()
@_traced
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


@mcp.tool()
@_traced
async def estop() -> dict:
    """Emergency stop: freeze every motor immediately at its current encoder reading (or cut torque if stale)."""
    bridge = await _get_bridge()
    return await bridge.abort()


# =============================================================================
# 2. Core Robotics Primitives (Angles, Frame Matrix, IK, Metric Depth)
# =============================================================================


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


@mcp.tool()
@_traced
async def move_joint(
    joints: list[float] | None = None,
    j_1: float | None = None,
    j_2: float | None = None,
    j_3: float | None = None,
    j_4: float | None = None,
    j_5: float | None = None,
    j_6: float | None = None,
    j_7: float | None = None,
    max_step_rad: float = 0.4,
    wait: bool = True,
) -> bool:
    """Move arm to target joint angles [j_1, ..., j_7] (radians), safely creep-stepped. Returns True on settle."""
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

    try:
        tgt_ticks = _calib.rad_to_ticks(target_rad, st.motors)
        step_ticks = max(40, min(int(round(max_step_rad * TICKS_PER_RAD)), DEFAULT_MAX_STEP * 2))
        # Creep smoothly in bounded steps so large IK moves never trigger shoulder overload
        for _ in range(12):
            curr = (await bridge.get_state()).positions()
            if all(abs(curr.get(mid, 0) - tgt) <= 25 for mid, tgt in tgt_ticks.items()):
                return True
            applied = await bridge.send_joint_targets(tgt_ticks, max_step=step_ticks)
            if not applied:
                return False
            if wait:
                settled = await bridge.wait_settle(applied, tol_ticks=25, timeout_s=3.0)
                if not settled:
                    return False
            if all(applied.get(mid) == tgt for mid, tgt in tgt_ticks.items()):
                return True
        return False
    except Exception:
        return False


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
) -> list[float]:
    """Solve IK for a metric target pose with an xyzw quaternion."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    seed_joints_rad = _calib.ticks_to_rad(st.motors)
    n_arm = _calib.num_arm_joints(len(st.motors))
    valid_bounds = all(m.range_max > m.range_min and m.range_min <= m.position <= m.range_max for m in st.motors[:n_arm])
    return _ik.solve_ik(
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


@mcp.tool()
@_traced
async def get_metric_depth(x: float, y: float, camera: str = "top", q_rad: list[float] | None = None) -> float:
    """Estimate metric depth z (meters) at a 2D pixel coordinate in the specified camera."""
    q = await _live_arm_q(q_rad, camera)
    return _calib.get_metric_depth(float(x), float(y), camera=camera, q_rad=q)


# =============================================================================
# 3. Calibration, Geometry & Scene Obstacle Management
# =============================================================================


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
    """Zero-shot 3D object localization via Gemini/Anthropic/OpenAI/Qwen/DeepSeek with automatic OpenCV fallback. Pass camera='both' to fuse top + gripper eye-in-hand ('wrist') views."""
    bridge = await _get_bridge()
    if camera in ("both", "dual"):
        top_jpeg, wrist_jpeg = await asyncio.gather(
            bridge.get_frame("top", denoise=denoise),
            bridge.get_frame("wrist", denoise=denoise),
        )
        q = await _live_arm_q(None, "wrist")
        return await localize_object_3d_dual(
            top_jpeg=top_jpeg,
            wrist_jpeg=wrist_jpeg,
            query=query,
            calib=_calib,
            q_rad=q,
            z_m=z_m,
            use_gemini=use_gemini,
            timeout_s=float(timeout_s),
            model=model,
            provider=provider,
        )
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
async def robo_jev_arm_watch(
    x: float,
    y: float,
    query: str = "target",
    camera: str = "top",
    depth_m: float | None = None,
) -> dict:
    """Arm Robo-Jev's <150 ms visual reflex watch on pixel (x, y) before a reach."""
    bridge = await _get_bridge()
    jpeg = await bridge.get_frame(camera, denoise=False)
    d_m = float(depth_m) if depth_m is not None else _calib.get_metric_depth(float(x), float(y), camera=camera)
    return _robo_jev.arm_watch(jpeg_bytes=jpeg, pixel=(float(x), float(y)), query=query, depth_m=d_m)


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


@mcp.tool()
@_traced
async def update_scene_obstacle(
    name: str,
    x: float,
    y: float,
    z: float = 0.05,
    shape: str = "cylinder",
    radius_m: float = 0.045,
    height_m: float = 0.10,
    remove: bool = False,
) -> dict:
    """Register or remove a 3D obstacle (e.g. mug, departure_tray) for IK/planner collision checks."""
    if remove:
        _obstacles.pop(name, None)
    else:
        _obstacles[name] = SceneObstacle(
            name=name, shape=shape, x=float(x), y=float(y), z=float(z),
            radius_m=float(radius_m), height_m=float(height_m),
        )
    return {"obstacles": sorted(_obstacles.keys())}


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
async def reload_config() -> dict:
    """Hot-reload calib/rig_calib.json and the Pink IK module without restarting the MCP server."""
    cal_status = _calib.reload()
    _ik.reload()
    if _bridge is not None and hasattr(_bridge, "apply_hardware_config"):
        hw = _calib.hardware_config()
        _bridge.apply_hardware_config(
            bus_serial=hw.get("bus_serial"),
            cam_top=hw.get("camera_top"),
            cam_wrist=hw.get("camera_wrist"),
        )
    return {"calibration": cal_status, "ik_backend": _ik.backend_name()}


# =============================================================================
# 4. Composite Execution Tools & Declarative Pipeline Runner
# =============================================================================


@mcp.tool()
@_traced
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
    return {"ok": ok, "pose": [x, y, z, qx, qy, qz, qw], "joints_rad": angles}


@mcp.tool()
@_traced
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

    from .motion import MotionLimits, correct_final_pose, flange_error, plan_flange_move, stream_joint_path

    if rot_frame not in ("flange", "base"):
        return {"ok": False, "problems": [f"rot_frame must be 'flange' or 'base', got {rot_frame!r}"]}
    limits = MotionLimits(speed_cm_s=speed_cm_s, rot_speed_deg_s=rot_speed_deg_s,
                          joint_speed_deg_s=joint_speed_deg_s, floor_cm=floor_cm,
                          min_duration_s=min_duration_s, lag_limit_ticks=lag_limit_ticks)
    if problems := limits.out_of_bounds():
        return {"ok": False, "executed": False, "problems": problems}
    if _flange_lock.locked():
        return {"ok": False, "executed": False, "problems": ["another move_flange is still running"]}
    async with _flange_lock:
        bridge = await _get_bridge()
        st = await bridge.get_state()
        rotation = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True)
        plan = await asyncio.to_thread(plan_flange_move, _calib, st.motors,
                                       np.array([dx_cm, dy_cm, dz_cm]) / 100.0, rotation, rot_frame, limits)
        result = {"ok": plan["ok"], "executed": False, "problems": plan["problems"], "plan": plan["summary"]}
        if not plan["ok"] or not execute:
            return result
        run = await stream_joint_path(bridge, _calib, plan["path"], plan["duration_s"], limits)
        result.update({"ok": run["ok"], "executed": True, "run": run})
        if run["ok"]:
            if correct:
                result["correction"] = await correct_final_pose(bridge, _calib, plan["path"], limits)
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
) -> dict:
    """Add or update a synthetic 3D object at known (x, y, z) meters in the mock tabletop scene."""
    bridge = await _get_bridge()
    if not hasattr(bridge, "add_mock_object"):
        return {"ok": False, "error": "mock_add_object is only available in --mock mode"}
    return {"ok": True, **bridge.add_mock_object(name=name, x=x, y=y, z=z, color=color, shape=shape, radius_m=radius_m)}


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


@mcp.tool()
@_traced
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
    rx, ry, rz, q_hover, q_grasp = await _resolve_grasp_pose(x, y, z, qx, qy, qz, qw, dz=approach_dz)
    await release()
    if q_hover is not None and q_grasp is not None:
        bridge = await _get_bridge()
        if not await move_joint(joints=q_hover, wait=True):
            return {"ok": False, "stage": "hover", "error": "move_joint(hover) failed"}
        if not await move_joint(joints=q_grasp, wait=True):
            return {"ok": False, "stage": "descend", "error": "move_joint(descend) failed"}
        if hasattr(bridge, "set_mock_ee_pose"):
            bridge.set_mock_ee_pose(rx, ry, rz)
        grip = await grasp()
        retract_ok = await move_joint(joints=q_hover, wait=True)
        if retract_ok and hasattr(bridge, "set_mock_ee_pose"):
            bridge.set_mock_ee_pose(rx, ry, rz + approach_dz)
        return {"ok": bool(grip.get("ok")) and bool(retract_ok), "grasp": grip, "retract_ok": retract_ok}

    hover = await move_to_pose(x=rx, y=ry, z=rz + approach_dz, qx=qx, qy=qy, qz=qz, qw=qw)
    if not hover["ok"]:
        return {"ok": False, "stage": "hover", **hover}
    descend = await move_to_pose(x=rx, y=ry, z=rz, qx=qx, qy=qy, qz=qz, qw=qw)
    if not descend["ok"]:
        return {"ok": False, "stage": "descend", **descend}
    grip = await grasp()
    retract = await move_to_pose(x=rx, y=ry, z=rz + approach_dz, qx=qx, qy=qy, qz=qz, qw=qw)
    return {"ok": bool(grip.get("ok")) and bool(retract.get("ok")), "grasp": grip, "retract_ok": retract["ok"]}


@mcp.tool()
@_traced
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
    retract = await move_to_pose(x=rx, y=ry, z=rz + retract_dz, qx=qx, qy=qy, qz=qz, qw=qw)
    if retract_home and retract["ok"]:
        await go_home()
    return {"ok": bool(rel.get("released")) and bool(retract.get("ok")), "release": rel, "retract_ok": retract["ok"]}


@mcp.tool()
@_traced
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
    "grasp": grasp,
    "release": release,
    "go_home": go_home,
    "estop": estop,
    "move_to_pose": move_to_pose,
    "move_flange": move_flange,
    "pick_at": pick_at,
    "place_at": place_at,
    "pick_pixel": pick_pixel,
}


@mcp.tool()
@_traced
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
        "5. On any unexpected slip, obstruction, or shift, call `estop()` immediately and re-`observe()`."
    )


@mcp.prompt()
def pick_and_place_skill(target_name: str, destination_name: str = "departure tray") -> str:
    """Guided camera-to-grasp workflow for picking a messy desk item and placing it cleanly."""
    return (
        f"Execute Pick & Place for `{target_name}` -> `{destination_name}`:\n"
        "1. Call `observe(views=['top', 'wrist'])` and locate pixel coordinates (x, y) of "
        f"`{target_name}`, `{destination_name}`, and any mug obstacle.\n"
        "2. Call `update_scene_obstacle` for any obstacle in the workspace.\n"
        "3. Call `get_metric_depth(x, y)` and `pixel_to_world(x, y)` (or composite `pick_pixel(x, y)`).\n"
        "4. Verify grasp outcome via `grasp` holding flag, then call `place_at(...)` and `go_home()`."
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
