#!/usr/bin/env python3
"""Test the desk-core MCP `move_flange` tool from the command line.

Starts desk_core.mcp_server over stdio (like an MCP client such as Claude Code would), plans the move with
a dry run, and on --execute asks for confirmation and calls the tool again with execute=True.
All IK and motion logic lives in desk_core (ik_tracik.py, motion.py); this is only a client.

Example
-------
    .venv/bin/python scripts/move_flange.py --dx 5                # plan +5 cm along base X
    .venv/bin/python scripts/move_flange.py --dz -2 --ry -15      # 2 cm down, -15 deg about the gripper axis
    .venv/bin/python scripts/move_flange.py --dx 5 --execute
    .venv/bin/python scripts/move_flange.py --dx 5 --mock         # against the mock bridge, no hardware
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]


def parse_result(res):
    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    text = next((b.text for b in res.content if getattr(b, "type", None) == "text"), "")
    try:
        return json.loads(text)
    except ValueError:
        return {"ok": False, "problems": [text]}


def show(result):
    for key, value in (result.get("plan") or {}).items():
        print(f"  {key:>24}: {value}")
    for problem in result.get("problems") or []:
        print("  REFUSED:", problem)
    for key in ("run", "correction", "result"):
        if key in result:
            print(f"  {key}: {json.dumps(result[key])}")


async def main(args) -> int:
    env = dict(os.environ)
    if args.mock:
        env["DESK_MOCK"] = "1"
    else:
        env.setdefault("STATION_HOST", args.host)
        env.setdefault("STATION_PORT", str(args.port))
    params = StdioServerParameters(command=sys.executable, args=["-m", "desk_core.mcp_server"], env=env, cwd=str(REPO))
    tool_args = {"dx_cm": args.dx, "dy_cm": args.dy, "dz_cm": args.dz, "rx_deg": args.rx, "ry_deg": args.ry,
                 "rz_deg": args.rz, "rot_frame": args.rot_frame, "speed_cm_s": args.speed_cm_s,
                 "rot_speed_deg_s": args.rot_speed_deg_s, "joint_speed_deg_s": args.joint_speed_deg_s,
                 "floor_cm": args.floor_cm, "min_duration_s": args.min_duration_s,
                 "lag_limit_ticks": args.lag_limit_ticks, "correct": not args.no_correct}

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            plan = parse_result(await session.call_tool("move_flange", {**tool_args, "execute": False}))
            print("plan:")
            show(plan)
            if not plan.get("ok"):
                return 1
            if not args.execute:
                print("dry run OK; add --execute to move")
                return 0
            if not args.yes and input("Clear the workspace, hand on power. Type 'yes' to move: ").strip() != "yes":
                print("cancelled")
                return 1
            result = parse_result(await session.call_tool("move_flange", {**tool_args, "execute": True}))
            print("execution:")
            show(result)
            return 0 if result.get("ok") else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dx", type=float, default=0.0, help="translation along base X, cm")
    parser.add_argument("--dy", type=float, default=0.0, help="translation along base Y, cm")
    parser.add_argument("--dz", type=float, default=0.0, help="translation along base Z, cm")
    parser.add_argument("--rx", type=float, default=0.0, help="rotation about X, deg")
    parser.add_argument("--ry", type=float, default=0.0, help="rotation about Y, deg (flange Y = gripper axis)")
    parser.add_argument("--rz", type=float, default=0.0, help="rotation about Z, deg")
    parser.add_argument("--rot-frame", choices=["flange", "base"], default="flange")
    parser.add_argument("--speed-cm-s", type=float, default=1.0)
    parser.add_argument("--rot-speed-deg-s", type=float, default=10.0)
    parser.add_argument("--joint-speed-deg-s", type=float, default=20.0)
    parser.add_argument("--floor-cm", type=float, default=8.0)
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--lag-limit-ticks", type=int, default=80, help="hold position if a joint lags this much")
    parser.add_argument("--no-correct", action="store_true", help="skip the closed-loop end-pose correction")
    parser.add_argument("--execute", action="store_true", help="move the arm (default: dry run)")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    parser.add_argument("--mock", action="store_true", help="use the mock bridge instead of Station")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    sys.exit(asyncio.run(main(parser.parse_args())))
