#!/usr/bin/env python3
"""Capture one wrist view per run and compare the observed board pose with the stored T_base_board.

A consistent offset across poses means the board moved; scatter around zero is calibration noise.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desk_core.station_bridge import LiveStationBridge
from fit_joint_offsets import TICKS_PER_RAD
from recover_scene_to_base import BASE_LINK, GRIPPER_FLANGE, load_urdf
from solve_scene_to_base import board_pose, pose_difference


async def grab(args, folder):
    bridge = LiveStationBridge(host=args.host, port=args.port, target_bus=args.bus, cam_wrist=args.wrist_camera)
    await bridge.connect()
    await asyncio.sleep(1.5)
    before = [m.position for m in (await bridge.get_state()).motors if 1 <= m.id <= 7]
    jpeg = await bridge.get_frame("wrist")
    after = [m.position for m in (await bridge.get_state()).motors if 1 <= m.id <= 7]
    if max(abs(a - b) for a, b in zip(before, after)) > 3:
        raise SystemExit("Arm moved during capture; hold it still and retry.")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "wrist.jpg").write_bytes(jpeg)
    (folder / "capture.json").write_text(json.dumps({"wrist_image": "wrist.jpg", "joint_ticks": after}) + "\n")


def evaluate(folder, mount_fit, calibration, robot):
    capture = json.loads((folder / "capture.json").read_text())
    zeros = np.array([mount_fit["joint_zero_ticks"][str(j)] for j in range(1, 8)], dtype=float)
    signs = np.array([mount_fit["joint_signs"][str(j)] for j in range(1, 8)], dtype=float)
    q = signs * (np.asarray(capture["joint_ticks"], dtype=float) - zeros) / TICKS_PER_RAD
    base_flange = robot.fk(q, target_link=GRIPPER_FLANGE, base_link=BASE_LINK)
    cam_board, rmse, _ = board_pose(folder / capture["wrist_image"], calibration["wrist"])
    observed = base_flange @ np.asarray(mount_fit["T_ee_cam_wrist"]) @ cam_board
    return observed, rmse, capture["joint_ticks"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Session folder; each run adds pose-NN/")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--wrist-mount", type=Path, required=True)
    parser.add_argument("--no-capture", action="store_true", help="Only re-evaluate saved poses.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--bus", default="5B8E115307")
    parser.add_argument("--wrist-camera", default="11391:19040:1:40")
    args = parser.parse_args()

    if not args.no_capture:
        index = len(list(args.folder.glob("pose-*")))
        asyncio.run(grab(args, args.folder / f"pose-{index:02d}"))

    mount_fit = json.loads(args.wrist_mount.read_text())
    calibration = json.loads(args.calibration.read_text())
    robot = load_urdf()
    stored = np.asarray(mount_fit["T_base_board"])
    offsets, results = [], []
    for folder in sorted(args.folder.glob("pose-*")):
        try:
            observed, rmse, ticks = evaluate(folder, mount_fit, calibration, robot)
        except SystemExit as error:
            print(f"{folder.name}: {error}")
            continue
        mm, deg = pose_difference(observed, stored)
        offset = 1000 * (observed[:3, 3] - stored[:3, 3])
        offsets.append(offset)
        results.append({"pose": folder.name, "ticks": ticks, "offset_mm_base_xyz": offset.round(1).tolist(),
                        "distance_mm": round(mm, 1), "rotation_deg": round(deg, 2), "reprojection_px": round(rmse, 2)})
        print(f"{folder.name}: offset xyz {np.round(offset, 1)} mm | {mm:.1f} mm {deg:.2f} deg | ticks {ticks}")
    if len(offsets) >= 2:
        offsets = np.array(offsets)
        mean, spread = offsets.mean(axis=0), offsets.std(axis=0, ddof=1)
        print(f"mean offset {np.round(mean, 1)} mm (|mean| {np.linalg.norm(mean):.1f}), "
              f"spread {np.round(spread, 1)} mm (|spread| {np.linalg.norm(spread):.1f})")
        (args.folder / "summary.json").write_text(json.dumps(
            {"poses": results, "mean_offset_mm": mean.round(2).tolist(), "spread_mm": spread.round(2).tolist()},
            indent=2) + "\n")


if __name__ == "__main__":
    main()
