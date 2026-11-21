#!/usr/bin/env python3
"""MCP stdio smoke test for desk_core — Bear-Bonnes issue #3.

Spawns the desk-core MCP server exactly the way Claude Code / Codex spawn it
(stdio subprocess, `python -m desk_core.mcp_server`), then runs the bare-bone
loop:

    observe -> move_joints -> grasp -> release -> go_home -> estop

asserting on every step, and finally verifies the replay log
(runs/<ts>/calls.jsonl + frames/*.jpg) captured the whole sequence.

Offline (default, MockStationBridge):
    uv run --offline python scripts/mcp_smoke.py
On the rig, after pinning STATION_HOST in .env:
    uv run --offline python scripts/mcp_smoke.py --live
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("DESK_RUNS_DIR") or (REPO_ROOT / "runs"))
EXPECTED_TOOLS = {"observe", "get_state", "set_torque", "move_joints", "grasp", "release", "go_home", "estop"}
SEQUENCE = ["observe", "move_joints", "grasp", "release", "go_home", "estop"]

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


class SmokeFailure(Exception):
    pass


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{' — ' + detail if detail else ''}")
    if not cond:
        raise SmokeFailure(f"{name}: {detail}")


def server_params(live: bool) -> StdioServerParameters:
    """Mirror .mcp.json's launch line so the smoke test validates the real config."""
    env = dict(os.environ)
    if not live:
        env["DESK_MOCK"] = "1"  # force offline mock regardless of local .env
        env["STATION_HOST"] = ""
    if shutil.which("uv"):
        cmd, args = "uv", ["run", "--offline", "--managed-python", "python", "-m", "desk_core.mcp_server"]
    else:
        cmd, args = sys.executable, ["-m", "desk_core.mcp_server"]
    return StdioServerParameters(command=cmd, args=args, env=env, cwd=str(REPO_ROOT))


def result_json(res) -> dict:
    """First JSON payload in a CallToolResult (handles fastmcp text wrappers)."""
    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    for block in res.content:
        if getattr(block, "type", None) != "text":
            continue
        for candidate in (block.text, *re.findall(r"(\{.*\}|\[.*\])", block.text, flags=re.S)):
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            return parsed.get("result", parsed) if isinstance(parsed, dict) and set(parsed) == {"result"} else parsed
    raise SmokeFailure(f"no JSON in tool result: {res.content[:1]!r}")


async def call(session: ClientSession, tool: str, args: dict, timeout_s: float = 60.0):
    t0 = time.perf_counter()
    if getattr(res := await asyncio.wait_for(session.call_tool(tool, args), timeout_s), "isError", False):
        raise SmokeFailure(f"{tool} returned an error: {res.content[:1]!r}")
    return result_json(res), time.perf_counter() - t0


async def run_sequence(session: ClientSession, live: bool) -> None:
    t0 = time.perf_counter()
    res = await asyncio.wait_for(session.call_tool("observe", {}), 60)
    ms = time.perf_counter() - t0
    if getattr(res, "isError", False):
        raise SmokeFailure(f"observe returned an error: {res.content[:1]!r}")
    summary = result_json(res)
    joints = summary.get("joints", [])
    check("observe returns telemetry JSON", isinstance(summary, dict) and "joints" in summary, f"{ms*1000:.0f} ms")
    check("observe reports motors", len(joints) >= (4 if live else 6), f"{len(joints)} joints")
    link = summary.get("link", {})
    check("observe reports link status", link.get("bridge") == ("live" if live else "mock"), json.dumps(link))
    n_img = sum(1 for b in res.content if getattr(b, "type", None) == "image")
    check("observe returns camera frames", n_img >= (1 if live else 2), f"{n_img} images")

    j1 = int(joints[0]["pos"])
    target = j1 + 200
    data, ms = await call(session, "move_joints", {"targets": {"1": target}, "max_step": 250, "wait": True})
    applied = {str(k): v for k, v in data.get("applied", {}).items()}
    check("move_joints applies creep-clamped target", applied.get("1") == target,
          f"{j1} -> {applied.get('1')} ({ms*1000:.0f} ms)")

    data, ms = await call(session, "grasp", {})
    check("grasp verifies jaw contact", data.get("ok") is True and "gripper_id" in data,
          f"holding={data.get('holding')} ({ms*1000:.0f} ms)")

    data, ms = await call(session, "release", {})
    check("release opens jaws", data.get("released") is True, f"{ms*1000:.0f} ms")

    data, ms = await call(session, "go_home", {})
    check("go_home targets joint midpoints", bool(data.get("home")), f"{len(data.get('home', {}))} joints ({ms*1000:.0f} ms)")

    data, ms = await call(session, "estop", {})
    check("estop freezes arm", data.get("aborted") is True, f"{ms*1000:.0f} ms")


def verify_run_log(live: bool) -> Path:
    logs = sorted(RUNS_DIR.glob("*/calls.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SmokeFailure(f"no run log found under {RUNS_DIR}")
    path = logs[-1]
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events = [r["event"] for r in recs]
    check("run log starts with run_start meta", events[:1] == ["run_start"])
    calls = [r for r in recs if r["event"] == "tool_call"]
    tools = [c["tool"] for c in calls]
    check("run log replays full tool sequence", tools == SEQUENCE, " vs ".join(tools))
    check("every logged call succeeded", all(c["ok"] for c in calls))
    check("every call has duration_ms", all("duration_ms" in c for c in calls))
    images = calls[0].get("images", [])
    for rel in images:
        if not (path.parent / rel).exists():
            raise SmokeFailure(f"logged frame missing on disk: {rel}")
    check("observe frames saved under frames/", len(images) >= (1 if live else 2), f"{len(images)} files")
    return path.parent


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="run against the rig using STATION_HOST from env/.env")
    ap.add_argument("--timeout", type=float, default=180.0, help="whole-session timeout (s)")
    args = ap.parse_args()

    print(f"desk-core MCP smoke — mode={'LIVE' if args.live else 'MOCK'}")
    t0 = time.perf_counter()
    async with stdio_client(server_params(args.live)) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=60)
            tools = {t.name for t in (await session.list_tools()).tools}
            check("server exposes all 8 tools", EXPECTED_TOOLS <= tools, ", ".join(sorted(tools)))
            await asyncio.wait_for(run_sequence(session, args.live), timeout=args.timeout)

    run_dir = verify_run_log(args.live)
    print(f"\nSMOKE PASS in {time.perf_counter() - t0:.1f}s — replay log: {run_dir.relative_to(REPO_ROOT)}")
    print("Replay anytime with: uv run python scripts/replay_run.py " + str(run_dir.relative_to(REPO_ROOT)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except SmokeFailure as exc:
        print(f"\nSMOKE FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        print("\nSMOKE FAIL (unexpected error)", file=sys.stderr)
        sys.exit(1)
