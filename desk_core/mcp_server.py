"""DESK-CORE FastMCP Server — connects Claude Code and Codex to NormaCore Station."""
from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from .station_bridge import StationBridge, create_bridge

load_dotenv()

_bridge: StationBridge | None = None
_connect_lock = asyncio.Lock()


async def _get_bridge() -> StationBridge:
    global _bridge
    async with _connect_lock:
        if _bridge is None:
            _bridge = create_bridge()
            await _bridge.connect()
        return _bridge


mcp = FastMCP("desk-core")


@mcp.tool()
async def observe(views: list[str] | None = None, denoise: bool = True) -> list:
    """Inspect camera feed(s) ('top', 'wrist') and live ST3215 joint telemetry in one round-trip."""
    bridge = await _get_bridge()
    want_views = views or ["top", "wrist"]
    st = await bridge.get_state()
    summary = {
        "bus_serial": st.bus_serial,
        "torque_enabled": st.torque_enabled,
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
async def get_state() -> dict:
    """Read live motor positions, calibrated min/max ticks, and currents."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    return {
        "bus_serial": st.bus_serial,
        "torque_enabled": st.torque_enabled,
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
async def set_torque(enable: bool) -> dict:
    """Safely enable or disable motor torque (writes Goal=Present + speed/accel caps first)."""
    bridge = await _get_bridge()
    ok = await bridge.set_torque(enable)
    return {"ok": ok, "torque_enabled": enable}


@mcp.tool()
async def move_joints(targets: dict[str, int], max_step: int = 250, wait: bool = True) -> dict:
    """Move joints to target encoder ticks (clamped to calibrated range & max_step per call)."""
    bridge = await _get_bridge()
    parsed = {int(k): int(v) for k, v in targets.items()}
    applied = await bridge.send_joint_targets(parsed, max_step=max_step)
    settled = await bridge.wait_settle(applied) if (wait and applied) else False
    return {"applied": applied, "settled": settled}


@mcp.tool()
async def grasp(close_ticks: int | None = None) -> dict:
    """Close gripper jaws and verify grasp via jaw encoder slack + motor current."""
    bridge = await _get_bridge()
    return await bridge.grasp_with_verify(close_ticks=close_ticks)


@mcp.tool()
async def release() -> dict:
    """Open gripper jaws to calibrated open limit."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    gripper = st.motors[-1]
    applied = await bridge.send_joint_targets({gripper.id: gripper.range_max}, max_step=1200)
    await bridge.wait_settle(applied, timeout_s=2.0)
    return {"released": True, "gripper_id": gripper.id, "target": applied.get(gripper.id)}


@mcp.tool()
async def go_home() -> dict:
    """Move arm safely toward calibrated midpoint pose."""
    bridge = await _get_bridge()
    st = await bridge.get_state()
    mids = {m.id: (m.range_min + m.range_max) // 2 for m in st.motors[:-1]}
    applied = await bridge.send_joint_targets(mids, max_step=300)
    settled = await bridge.wait_settle(applied, timeout_s=4.0)
    return {"home": applied, "settled": settled}


@mcp.tool()
async def estop() -> dict:
    """Emergency stop: freeze every motor immediately at its current encoder reading."""
    bridge = await _get_bridge()
    return await bridge.abort()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
