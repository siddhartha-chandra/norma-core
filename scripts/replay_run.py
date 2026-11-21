#!/usr/bin/env python3
"""Replay a recorded desk_core run against the current bridge config.

    uv run --offline python scripts/replay_run.py --latest      # newest run
    uv run --offline python scripts/replay_run.py runs/20261121T153000Z-1234

Bridge mode comes from the environment / .env, so the same recorded run can be
replayed against MockStationBridge offline or the live rig. Targets are
absolute ticks, so replaying a motion sequence converges to the same pose
even after the mock/real state drifted (range clamping still applies).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcp_smoke import RUNS_DIR, SmokeFailure, result_json, server_params  # noqa: E402

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def load_calls(run_dir: Path) -> list[dict]:
    path = run_dir / "calls.jsonl"
    if not path.exists():
        raise SmokeFailure(f"{path} not found")
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in recs if r.get("event") == "tool_call"]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", help="run directory containing calls.jsonl")
    ap.add_argument("--latest", action="store_true", help="use the most recent run under runs/")
    args = ap.parse_args()

    if args.latest or not args.run_dir:
        cands = sorted(RUNS_DIR.glob("*/calls.jsonl"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SmokeFailure(f"no runs found under {RUNS_DIR}")
        run_dir = cands[-1].parent
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = RUNS_DIR.parent / run_dir

    calls = load_calls(run_dir)
    live = os.environ.get("DESK_MOCK") != "1" and bool(os.environ.get("STATION_HOST", "").strip())
    print(f"Replaying {len(calls)} calls from {run_dir.name} (mode={'LIVE' if live else 'MOCK'})")
    failures = 0
    async with stdio_client(server_params(live)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for rec in calls:
                t0 = time.perf_counter()
                try:
                    res = await asyncio.wait_for(session.call_tool(rec["tool"], rec["args"]), 60)
                    ok = not getattr(res, "isError", False)
                    detail = "" if ok else str(result_json(res))[:80]
                except Exception as exc:
                    ok, detail = False, repr(exc)[:120]
                ms = (time.perf_counter() - t0) * 1000
                if ok != rec.get("ok", True):
                    ok = False
                    detail = detail or "ok-status differs from recorded run"
                failures += 0 if ok else 1
                print(f"  [{'PASS' if ok else 'FAIL'}] #{rec['seq']:02d} {rec['tool']} ({ms:.0f} ms) {detail}")
    if failures:
        print(f"\nREPLAY FAIL: {failures}/{len(calls)} calls diverged", file=sys.stderr)
        return 1
    print(f"\nREPLAY PASS: {len(calls)}/{len(calls)} calls converged")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except SmokeFailure as exc:
        print(f"\nREPLAY FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        print("\nREPLAY FAIL (unexpected error)", file=sys.stderr)
        sys.exit(1)
