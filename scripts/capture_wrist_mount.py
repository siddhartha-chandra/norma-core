#!/usr/bin/env python3
"""Collect fixed-checkerboard hand-eye samples from a manually posed follower."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from recover_scene_to_base import DEFAULT_CALIBRATION, GRIPPER_FLANGE, load_urdf
from desk_core.calib_store import CalibrationStore
from desk_core.station_bridge import LiveStationBridge


def board_hint(image, pattern):
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, pattern)
    if not found:
        return "Grid unclear: aim at the board and fit all four edges in view."
    points = corners.reshape(-1, 2)
    height, width = gray.shape
    low = points.min(axis=0)
    high = points.max(axis=0)
    if min(low[0]/width, low[1]/height, 1-high[0]/width, 1-high[1]/height) < 0.05:
        return "Grid near image edge: center the board or move the wrist back."
    return "Full grid detected. Keep the board fixed."


async def capture(args):
    wrist = json.loads(args.calibration.read_text())["wrist"]
    calib = CalibrationStore(args.joint_calibration)
    if not args.joint_calibration:
        print("Recording raw ticks. Joint angles and FK use the current calibration defaults.")
    robot = load_urdf()
    bridge = LiveStationBridge(host=args.host, port=args.port, target_bus=args.bus,
                               cam_wrist=args.camera, follow_motor_rx=True)
    await bridge.connect()
    for _ in range(50):
        keys = [k for k in await bridge.list_cameras() if args.camera in k]
        if keys:
            break
        await asyncio.sleep(0.1)
    if len(keys) != 1:
        raise ValueError(f"Select one wrist camera from {await bridge.list_cameras()}")
    camera = keys[0]
    manifest = args.directory / "samples.json"
    samples = json.loads(manifest.read_text()) if manifest.exists() else []
    for sample in samples:
        if sample["camera"] != camera or sample["bus"] != args.bus:
            raise ValueError("Existing session uses a different wrist camera or motor bus.")
        if not (args.directory / sample["image"]).is_file() or len(sample["joint_ticks"]) != 7:
            raise ValueError("Incomplete existing sample: " + sample["image"])
    next_number = 1 + max((int(p.stem[6:]) for p in args.directory.glob("wrist-*.jpg")
                           if p.stem[6:].isdigit()), default=-1)
    print(f"Opened {args.directory} with {len(samples)} saved poses", flush=True)
    board = wrist["board"]
    pattern = (board["inner_columns"], board["inner_rows"])
    detector = None
    hint = "Checking board framing..."
    hint_stamp = last_check = float("-inf")
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
        if detector is not None and detector.done():
            hint = detector.result()
            hint_stamp = checked_frame
            detector = None
        if detector is None and now - last_check >= 0.8:
            checked_frame = stamp
            detector = asyncio.create_task(asyncio.to_thread(board_hint, image.copy(), pattern))
            last_check = now
        distinct = fresh and all(np.max(abs(ticks - s["joint_ticks"])) >= 25 for s in samples)
        ready = stable and distinct and stamp != last_frame and stamp >= still_since
        status = "Hold this pose briefly while the joints settle."
        if not valid:
            status = "Check follower bus, seven encoders and motor status"
        elif not fresh:
            status = f"Waiting for fresh data: joints {joint_age:.1f}s / video {now-stamp:.1f}s"
        elif ready:
            status = "READY - press SPACE to save"
        elif stable and not distinct:
            status = "Similar pose saved: try a sideways or forward tilt, keeping the board in view."
        if ready and previous_status != status:
            print(status, flush=True)
        previous_status = status
        # Add the guidance below the image so it cannot obscure the board.
        panel = cv2.copyMakeBorder(image, 0, 88, 0, 0, cv2.BORDER_CONSTANT)
        guide = hint if now - hint_stamp < 2 else "Checking board framing..."
        for row, line in enumerate((status, guide, f"{len(samples)} saved  |  Aim for 15 varied poses  |  Q quit")):
            color = (90, 240, 130) if row == 0 and ready else (220, 220, 220)
            cv2.putText(panel, line, (12, image.shape[0] + 23 + 26*row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1, cv2.LINE_AA)
        cv2.imshow(window, panel)
        key = cv2.waitKey(20) & 255
        if key in (27, ord("q")):
            break
        if ready and key == ord(" "):
            name = f"wrist-{next_number:03d}.jpg"
            pose = robot.fk(angles, target_link=GRIPPER_FLANGE)
            with (args.directory / name).open("xb") as stream:
                stream.write(jpeg)
            next_number += 1
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
    parser.add_argument("directory", type=Path, help="Session directory (resumes when --resume is supplied)")
    parser.add_argument("--resume", action="store_true", help="Append poses from the same fixed-board setup")
    for flag, env in (("host", "STATION_HOST"), ("camera", "CAMERA_WRIST"), ("bus", "STATION_BUS_SERIAL")):
        parser.add_argument("--" + flag, default=os.getenv(env), required=not os.getenv(env))
    parser.add_argument("--port", type=int, default=int(os.getenv("STATION_PORT", "8888")))
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--joint-calibration", type=Path, help="Rig JSON with measured joint signs and zeros")
    args = parser.parse_args()
    try:
        args.directory.mkdir(parents=True, exist_ok=args.resume)
        with (args.directory / ".capture.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(1, "A capture window is already writing to this session.\n")
            asyncio.run(capture(args))
    finally:
        cv2.destroyAllWindows()
