#!/usr/bin/env python3
"""Trace an infinity curve with the gripper tip and record the measured path.

The curve lies in the base X-Z plane, starts and ends at its crossing point (the current tip position) and
keeps the flange orientation fixed, so the tip and the flange trace the same shape. Each waypoint is solved
with TRAC-IK, streamed with desk_core.motion (minimum-jerk timing, gravity and friction feed-forward) and the
measured joint angles are saved for scripts/plot_trace.py.

Example
-------
    .venv/bin/python scripts/trace_infinity.py                      # plan only
    .venv/bin/python scripts/trace_infinity.py --execute            # move and save docs/motion/infinity_trace.npz
    .venv/bin/python scripts/plot_trace.py docs/motion/infinity_trace.npz
"""
import argparse
import asyncio
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from desk_core.calib_store import CalibrationStore  # noqa: E402
from desk_core.ik_tracik import get_solver  # noqa: E402
from desk_core.motion import (TCP_IN_FLANGE, MotionLimits, allowed_bounds_rad, correct_final_pose,  # noqa: E402
                              lowest_point_m, stream_joint_path)

STEP_M = 0.002
MAX_JUMP_RAD = 0.15


def infinity_offsets(width_m: float, height_m: float, loops: int, step_m: float = STEP_M):
    """Infinity curve offsets in the base X-Z plane from the crossing point, evenly spaced by arc length.

    Returns
    -------
    offsets : numpy.ndarray
        (N, 3) offsets in metres, first and last at the origin.
    length : float
        Curve length in metres.
    """
    t = np.linspace(0, 2 * np.pi * loops, 20000)
    pts = np.c_[0.5 * width_m * np.sin(t), np.zeros_like(t), 0.5 * height_m * np.sin(2 * t)]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0, arc[-1], int(arc[-1] / step_m) + 1)
    return np.c_[[np.interp(s, arc, pts[:, k]) for k in range(3)]].T, float(arc[-1])


def plan_tip_path(calib, motors, offsets):
    """Joint path that moves the flange (and so the tip) by ``offsets`` at fixed orientation; None on failure."""
    solver = get_solver(str(calib.urdf_path()))
    q0 = np.array(calib.ticks_to_rad(motors))
    lower, upper = allowed_bounds_rad(calib, motors)
    T0 = solver.fk(q0)
    path, seed = [q0], q0
    for i, d in enumerate(offsets[1:], 1):
        sol = solver.solve(T0[:3, 3] + d, T0[:3, :3], seed, lower, upper)
        if not sol.ok:
            return None, f"waypoint {i}/{len(offsets) - 1}: {sol.reason}"
        if np.abs(sol.q - seed).max() > MAX_JUMP_RAD:
            return None, f"waypoint {i}: joint jump {np.abs(sol.q - seed).max():.3f} rad"
        path.append(sol.q)
        seed = sol.q
    return np.array(path), ""


def tip_position(solver, q) -> np.ndarray:
    T = solver.fk(np.asarray(q))
    return T[:3, :3] @ TCP_IN_FLANGE + T[:3, 3]


async def main(args) -> int:
    from desk_core.station_bridge import LiveStationBridge

    calib = CalibrationStore()
    solver = get_solver(str(calib.urdf_path()))
    hw = calib.hardware_config()
    bridge = LiveStationBridge(host=args.host, port=args.port, target_bus=hw["bus_serial"],
                               cam_top=hw["camera_top"], cam_wrist=hw["camera_wrist"])
    await bridge.connect()
    await asyncio.sleep(2.0)
    motors = (await bridge.get_state()).motors
    offsets, length = infinity_offsets(args.width_cm / 100, args.height_cm / 100, args.loops)
    path, problem = plan_tip_path(calib, motors, offsets)
    if path is None:
        print("REFUSED:", problem)
        return 1
    lowest = min(lowest_point_m(calib, q) for q in path[::5])
    duration = length / (args.speed_cm_s / 100)
    print(f"tip start cm {np.round(tip_position(solver, path[0]) * 100, 1).tolist()}, "
          f"{args.width_cm:g} x {args.height_cm:g} cm, {args.loops} loops, path {length * 100:.0f} cm, "
          f"{len(path)} waypoints, lowest point {lowest * 100:.1f} cm, duration {duration:.0f} s")
    if lowest < args.floor_cm / 100:
        print(f"REFUSED: arm comes within {lowest * 100:.1f} cm of the table (floor {args.floor_cm} cm)")
        return 1
    if not args.execute:
        print("plan OK; add --execute to move")
        return 0
    limits = MotionLimits(floor_cm=args.floor_cm)
    trace = []
    run = await stream_joint_path(bridge, calib, path, duration, limits, trace=trace)
    print("run", run)
    if not run["ok"]:
        return 2
    correction = await correct_final_pose(bridge, calib, path, limits)
    print("correction", correction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             planned=np.array([tip_position(solver, q) for q in path]),
             t=np.array([r[0] for r in trace]),
             measured=np.array([tip_position(solver, r[1]) for r in trace]),
             expected=np.array([tip_position(solver, r[2]) for r in trace]),
             width_cm=args.width_cm, height_cm=args.height_cm, loops=args.loops)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width-cm", type=float, default=10.0)
    parser.add_argument("--height-cm", type=float, default=5.0)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--speed-cm-s", type=float, default=2.5, help="mean tip speed")
    parser.add_argument("--floor-cm", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "motion" / "infinity_trace.npz")
    parser.add_argument("--execute", action="store_true", help="move the arm (default: plan only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    sys.exit(asyncio.run(main(parser.parse_args())))
