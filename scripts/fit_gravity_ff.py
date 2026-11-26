#!/usr/bin/env python3
"""Fit per-joint gravity and friction feed-forward for desk_core.motion.

The ST3215 servos are proportional position loops, so a joint settles short of its goal by roughly
gravity torque / stiffness, plus a friction band that depends on the direction it arrived from. This visits
poses at different reaches (slow joint-space moves), holds each goal, measures the shortfall and fits
``shortfall_rad = gain * gravity_torque_Nm + friction_rad * sign(move direction)`` per joint. The result is
written to calib/gravity_ff.json, which desk_core.motion adds to every streamed target.

Example
-------
    .venv/bin/python scripts/fit_gravity_ff.py              # plan the poses only
    .venv/bin/python scripts/fit_gravity_ff.py --execute    # move, measure and write the gains
    .venv/bin/python scripts/fit_gravity_ff.py --around-current --append --execute   # add poses near the live pose
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from desk_core.calib_store import TICKS_PER_RAD, CalibrationStore  # noqa: E402
from desk_core.ik_tracik import get_solver  # noqa: E402
from desk_core.motion import (GRAVITY_FF_PATH, MotionLimits, allowed_bounds_rad, gravity_torque,  # noqa: E402
                              lowest_point_m, stream_joint_path)
from desk_core.station_bridge import LiveStationBridge, MotorState  # noqa: E402

NOMINAL_TICKS = np.array([2049, 2233, 1063, 1098, 1857, 1300, 1399])
SPREAD_TICKS = np.array([0, 250, 400, 350, 0, 200, 0])
WRIST_SPREAD_TICKS = np.array([300, 150, 200, 200, 300, 200, 300])
N_POSES = 8
HOLD_S = 2.0
FIT_JOINTS = (1, 2, 3, 5)
FRICTION_ONLY_JOINTS = (0, 4, 6)


def plan_poses(calib, motors, nominal_ticks, spread=SPREAD_TICKS, seed=0):
    """Joint-space poses around ``nominal_ticks`` with a wide spread of shoulder torque, all safely clear."""
    solver = get_solver(str(calib.urdf_path()))
    lower, upper = allowed_bounds_rad(calib, motors)
    nominal = [MotorState(m.id, int(t), m.current_ma, m.range_min, m.range_max)
               for m, t in zip(motors[:7], nominal_ticks)]
    q_nom = np.array(calib.ticks_to_rad(nominal))
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(4000):
        q = q_nom + rng.uniform(-1, 1, 7) * spread / TICKS_PER_RAD
        if np.any(q < lower) or np.any(q > upper):
            continue
        p = solver.fk(q)[:3, 3]
        if p[1] > -0.05 or not 0.15 < p[2] < 0.35 or lowest_point_m(calib, q) < 0.10:
            continue
        candidates.append((gravity_torque(q)[1], q, p))
    candidates.sort(key=lambda c: c[0])
    picks = [candidates[i] for i in np.linspace(0, len(candidates) - 1, N_POSES).astype(int)]
    return [(p[1] * 100, p[2] * 100, q) for _, q, p in picks]


def saved_samples():
    """Samples from the gains file, each with its move ``direction`` (derived for files that predate it)."""
    if not GRAVITY_FF_PATH.exists():
        return []
    saved = json.loads(GRAVITY_FF_PATH.read_text())
    samples = saved["samples"]
    if samples and "direction" not in samples[0]:
        if "start" not in saved:
            samples = samples[1:]
        prev = np.array(saved.get("start") or saved["samples"][0]["reached"])
        for s in samples:
            s["direction"] = np.sign(np.array(s["goal"]) - prev).tolist()
            prev = np.array(s["reached"])
    return samples


async def main(args):
    if args.refit:
        fit(saved_samples())
        return
    calib = CalibrationStore()
    hw = calib.hardware_config()
    bridge = LiveStationBridge(host=args.host, port=args.port, target_bus=hw["bus_serial"],
                               cam_top=hw["camera_top"], cam_wrist=hw["camera_wrist"], follow_motor_rx=True)
    await bridge.connect()
    await asyncio.sleep(2.0)
    motors = (await bridge.get_state()).motors
    if args.around_current:
        poses = plan_poses(calib, motors, [m.position for m in motors[:7]], WRIST_SPREAD_TICKS, seed=1)
    else:
        poses = plan_poses(calib, motors, NOMINAL_TICKS)
    for y, z, q in poses:
        print(f"pose y={y:+.0f} z={z:.0f} cm  gravity Nm {np.round(gravity_torque(q), 3).tolist()}")
    if not args.execute:
        return
    limits = MotionLimits()
    samples = []
    q_prev = np.array(calib.ticks_to_rad(motors))
    for y, z, q in poses:
        path = np.array([q_prev + s * (q - q_prev) for s in np.linspace(0, 1, 30)])
        if min(lowest_point_m(calib, p) for p in path) < 0.08:
            print(f"skip y={y} z={z}: joint-space path too low")
            continue
        duration = max(2.0, 1.875 * float(np.abs(q - q_prev).max()) / np.radians(15.0))
        run = await stream_joint_path(bridge, calib, path, duration, limits, gravity_ff=False)
        if not run["ok"] and run.get("settled") is not False:  # an unsettled endpoint is the sag being fitted
            print("aborted:", run)
            return
        await asyncio.sleep(HOLD_S)
        state = await bridge.get_state()
        reached = np.array(calib.ticks_to_rad(state.motors))
        shortfall = q - reached
        samples.append({"goal": q.tolist(), "reached": reached.tolist(), "torque": gravity_torque(reached).tolist(),
                        "direction": np.sign(q - q_prev).tolist()})
        print(f"y={y:+.0f} z={z:.0f}: shortfall deg {np.round(np.degrees(shortfall), 2).tolist()}")
        q_prev = reached

    fit((saved_samples() if args.append else []) + samples)


def fit(samples):
    """Least-squares ``shortfall = gain * torque + friction * direction`` per joint; writes the file."""
    goal = np.array([s["goal"] for s in samples])
    reached = np.array([s["reached"] for s in samples])
    torque = np.array([s["torque"] for s in samples])
    direction = np.array([s["direction"] for s in samples])
    gains, friction = [0.0] * 7, [0.0] * 7
    for j in FIT_JOINTS:
        A = np.c_[torque[:, j], direction[:, j]]
        (gains[j], friction[j]), *_ = np.linalg.lstsq(A, goal[:, j] - reached[:, j], rcond=None)
        resid = goal[:, j] - reached[:, j] - A @ [gains[j], friction[j]]
        print(f"joint {j + 1}: gain {gains[j]:.3f} rad/Nm, friction {np.degrees(friction[j]):.2f} deg, "
              f"residual rms {np.degrees(np.sqrt(np.mean(resid ** 2))):.2f} deg")
    for j in FRICTION_ONLY_JOINTS:
        moved = direction[:, j] != 0
        if moved.any():
            friction[j] = float(np.mean((goal[moved, j] - reached[moved, j]) * direction[moved, j]))
            print(f"joint {j + 1}: friction {np.degrees(friction[j]):.2f} deg")
    GRAVITY_FF_PATH.write_text(json.dumps({"gain_rad_per_Nm": [float(g) for g in gains],
                                           "friction_rad": [float(f) for f in friction],
                                           "samples": samples}, indent=2) + "\n")
    print(f"wrote {GRAVITY_FF_PATH} ({len(samples)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="move the arm and fit (default: plan only)")
    parser.add_argument("--refit", action="store_true", help="refit from the samples saved in the gains file")
    parser.add_argument("--around-current", action="store_true", help="sample poses around the live pose")
    parser.add_argument("--append", action="store_true", help="fit together with the samples already saved")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    asyncio.run(main(parser.parse_args()))
