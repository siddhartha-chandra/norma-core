"""DESK-CORE FastMCP Server — connects Claude Code and Codex to NormaCore Station.

Every tool call is recorded under runs/<UTC timestamp>/ (calls.jsonl +
frames/*.jpg) so sessions are replayable and benchmarkable (issue #3);
set DESK_RUNLOG=0 to disable. See desk_core/runlog.py.
"""
from __future__ import annotations

import asyncio
import atexit
import functools
import json
import os
import time
from pathlib import Path

from dotenv import dotenv_values
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from .runlog import RunLogger
from .station_bridge import REPO_ROOT, StationBridge, create_bridge

for _k, _v in dotenv_values(REPO_ROOT / ".env").items():
    if _v is not None and not os.environ.get(_k, "").strip():
        os.environ[_k] = _v

_bridge: StationBridge | None = None
_connect_lock = asyncio.Lock()
_connect_cooldown_until = 0.0

RUNS_DIR = Path(os.environ.get("DESK_RUNS_DIR") or (REPO_ROOT / "runs"))
_logger: RunLogger | None = None


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
    """Lazy bridge singleton that retries failed cold connects.

    A *connected* Live bridge heals Station restarts on its own
    (LiveStationBridge supervisor task); this only covers the cold case where
    Station was down when the agent session started — previously a failed
    connect cached the unconnected bridge forever.
    """
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


mcp = FastMCP("desk-core")


@mcp.tool()
@_traced
async def observe(views: list[str] | None = None, denoise: bool = True) -> list:
    """Inspect camera feed(s) ('top', 'wrist') and live ST3215 joint telemetry in one round-trip."""
    bridge = await _get_bridge()
    want_views = views or ["top", "wrist"]
    st = await bridge.get_state()
    summary = {
        "bus_serial": st.bus_serial,
        "torque_enabled": st.torque_enabled,
        "link": getattr(bridge, "link_status", lambda: {"bridge": "unknown"})(),
        "cameras": await bridge.list_cameras(),
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
    """Read live motor positions, calibrated min/max ticks, and currents."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    return {
        "bus_serial": st.bus_serial,
        "torque_enabled": st.torque_enabled,
        "link_age_ms": st.link_age_ms,
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
async def move_joints(targets: dict[str, int], max_step: int = 250, wait: bool = True) -> dict:
    """Move joints to target encoder ticks (clamped to calibrated range & max_step per call)."""
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
    return await bridge.grasp_with_verify(close_ticks=close_ticks)


@mcp.tool()
@_traced
async def release() -> dict:
    """Open gripper jaws to calibrated open limit."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    gripper = st.motors[-1]
    applied = await bridge.send_joint_targets({gripper.id: gripper.range_max}, max_step=1200)
    await bridge.wait_settle(applied, timeout_s=2.0)
    return {"released": True, "gripper_id": gripper.id, "target": applied.get(gripper.id)}


@mcp.tool()
@_traced
async def go_home() -> dict:
    """Move arm safely toward calibrated midpoint pose."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    mids = {m.id: (m.range_min + m.range_max) // 2 for m in st.motors[:-1]}
    applied = await bridge.send_joint_targets(mids, max_step=300)
    settled = await bridge.wait_settle(applied, timeout_s=4.0)
    return {"home": applied, "settled": settled}


@mcp.tool()
@_traced
async def estop() -> dict:
    """Emergency stop: freeze every motor immediately at its current encoder reading."""
    bridge = await _get_bridge()
    return await bridge.abort()


def _init_run_logger() -> None:
    global _logger
    if os.environ.get("DESK_RUNLOG", "1") == "0":
        return
    host = "" if os.environ.get("DESK_MOCK") == "1" else os.environ.get("STATION_HOST", "").strip()
    _logger = RunLogger(
        RUNS_DIR,
        meta={
            "mode": "live" if host else "mock",
            "station": f"{host}:{os.environ.get('STATION_PORT', '8888')}" if host else None,
            "bus_serial": os.environ.get("STATION_BUS_SERIAL") or None,
            "camera_top": os.environ.get("CAMERA_TOP") or None,
            "camera_wrist": os.environ.get("CAMERA_WRIST") or None,
            "pid": os.getpid(),
        },
    )
    atexit.register(_logger.close)


_init_run_logger()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
