#!/usr/bin/env python3
"""Save one scene frame, one wrist frame and follower joint ticks for the scene-to-base solve."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desk_core.station_bridge import LiveStationBridge


async def capture(args):
    bridge = LiveStationBridge(host=args.host, port=args.port, target_bus=args.bus,
                               cam_top=args.scene_camera, cam_wrist=args.wrist_camera)
    await bridge.connect()
    await asyncio.sleep(2.0)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scene.jpg").write_bytes(await bridge.get_frame("top"))
    (args.output / "wrist.jpg").write_bytes(await bridge.get_frame("wrist"))
    state = await bridge.get_state()
    ticks = [m.position for m in state.motors if 1 <= m.id <= 7]
    record = {"scene_image": "scene.jpg", "wrist_image": "wrist.jpg", "bus": state.bus_serial,
              "scene_camera": args.scene_camera, "wrist_camera": args.wrist_camera, "joint_ticks": ticks}
    (args.output / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--bus", default="5B8E115307")
    parser.add_argument("--scene-camera", default="11391:19040:1:38")
    parser.add_argument("--wrist-camera", default="11391:19040:1:40")
    asyncio.run(capture(parser.parse_args()))


if __name__ == "__main__":
    main()
