#!/usr/bin/env python3
"""Collect fixed-checkerboard hand-eye samples from a manually posed follower."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from recover_scene_to_base import DEFAULT_CALIBRATION, GRIPPER_FLANGE, load_urdf
from desk_core.calib_store import CalibrationStore
from desk_core.station_bridge import LiveStationBridge



async def capture(args):
    wrist = json.loads(args.calibration.read_text())["wrist"]
    calib = CalibrationStore(args.joint_calibration)
    if not args.joint_calibration:
        print("Recording raw ticks. Joint angles and FK use the current calibration defaults.")
    robot = load_urdf()
    bridge = LiveStationBridge(host=args.host, port=args.port, target_bus=args.bus,
                               cam_wrist=args.camera)
    await bridge.connect()
    for _ in range(50):
        keys = [k for k in await bridge.list_cameras() if args.camera in k]
        if keys:
            break
        await asyncio.sleep(0.1)
    if len(keys) != 1:
        raise ValueError(f"Select one wrist camera from {await bridge.list_cameras()}")
    camera = keys[0]
    args.directory.mkdir(parents=True, exist_ok=False)
    samples = []
    baseline = None
    last_frame = None
    previous_status = None
    window = "Follower wrist calibration"
    cv2.namedWindow(window)
    while True:
        jpeg = await bridge.get_frame("wrist")
        state = await bridge.get_state()
        now = time.monotonic()
        stamp = bridge.frame_received_at[camera]
        motors = [m for m in state.motors if 1 <= m.id <= 7]
        angles = calib.ticks_to_rad(motors)
        ticks = np.array([m.position for m in motors])
        times = [bridge.rx_received_at.get((args.bus, m.id), 0) for m in motors]
        joint_age = now - min(times) if times else float("inf")
        valid = state.bus_serial == args.bus and len(ticks) == 7 and not any(m.error_status for m in motors)
        if not valid or joint_age > 0.7:
            baseline = None
        elif baseline is None or np.max(abs(ticks - baseline)) > 3:
            baseline = ticks.copy()
            still_since = now
        stable = baseline is not None and now - still_since >= 0.7
        fresh = valid and joint_age < 0.35 and now - stamp < 0.35
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None or list(image.shape[1::-1]) != wrist["image_size"]:
            raise ValueError("Wrist image does not match the calibrated resolution")
        distinct = fresh and all(np.max(abs(ticks - s["joint_ticks"])) >= 25 for s in samples)
        ready = stable and distinct and stamp != last_frame and stamp >= still_since
        status = "Hold still; waiting for fresh joints and video"
        if not valid:
            status = "Check follower bus, seven encoders and motor status"
        elif not fresh:
            status = f"Waiting for fresh data: joints {joint_age:.1f}s / video {now-stamp:.1f}s"
        elif ready:
            status = "READY - press SPACE to save"
        elif stable and not distinct:
            status = "Pose already saved - change wrist tilt"
        if ready and previous_status != status:
            print(status, flush=True)
        previous_status = status
        cv2.putText(image, f"{len(samples)} saved | {status} | Q quit", (10, 30),
                    0, 0.7, (0, 255, 0) if ready else (0, 180, 255), 2)
        cv2.imshow(window, image)
        key = cv2.waitKey(20) & 255
        if key in (27, ord("q")):
            break
        if ready and key == ord(" "):
            name = f"wrist-{len(samples):03d}.jpg"
            pose = robot.fk(angles, target_link=GRIPPER_FLANGE)
            (args.directory / name).write_bytes(jpeg)
            samples.append(dict(image=name, joint_ticks=ticks.tolist(), joints_rad=angles,
                                T_base_ee=pose.tolist(), camera=camera, bus=args.bus,
                                frame_received_mono=stamp, joint_received_mono=times))
            temp = args.directory / "samples.tmp"
            temp.write_text(json.dumps(samples, indent=2))
            temp.replace(args.directory / "samples.json")
            last_frame = stamp
        await asyncio.sleep(0.02)


if __name__ == "__main__":
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New session directory")
    for flag, env in (("host", "STATION_HOST"), ("camera", "CAMERA_WRIST"), ("bus", "STATION_BUS_SERIAL")):
        parser.add_argument("--" + flag, default=os.getenv(env), required=not os.getenv(env))
    parser.add_argument("--port", type=int, default=int(os.getenv("STATION_PORT", "8888")))
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--joint-calibration", type=Path, help="Rig JSON with measured joint signs and zeros")
    args = parser.parse_args()
    try:
        asyncio.run(capture(args))
    finally:
        cv2.destroyAllWindows()
