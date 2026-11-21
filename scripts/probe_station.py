#!/usr/bin/env python3
"""Quick hardware probe: prints connected buses, motors, calibration ranges, and saves cam snapshots."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from desk_core.station_bridge import LiveStationBridge  # noqa: E402


async def main() -> None:
    host = os.environ.get("STATION_HOST", "127.0.0.1")
    port = int(os.environ.get("STATION_PORT", "8888"))
    bus = os.environ.get("STATION_BUS_SERIAL", "") or None

    bridge = LiveStationBridge(host=host, port=port, target_bus=bus)
    await bridge.connect()

    try:
        st = await bridge.get_state()
        print(f"\nSelected Bus: {st.bus_serial} ({len(st.motors)} motors)")
        for m in st.motors:
            cal = "OK" if m.range_max > m.range_min else "UNCALIBRATED"
            print(f"  Motor {m.id}: pos={m.position} norm={m.to_norm():.3f} range=[{m.range_min}, {m.range_max}] ({cal}) cur={m.current_ma}mA err={m.error_status}")
    except Exception as exc:
        print(f"\n[Motors] Not ready yet: {exc}")

    cams = await bridge.list_cameras()
    print(f"\nDiscovered Cameras ({len(cams)}):")
    for i, cam_id in enumerate(cams):
        jpeg = bridge._frames.get(cam_id)
        if jpeg:
            out = ROOT / f"cam_{i}.jpg"
            out.write_bytes(jpeg)
            print(f"  [{i}] id={cam_id!r} -> saved {out.name} ({len(jpeg)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
