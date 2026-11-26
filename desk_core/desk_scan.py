"""Run the periodic overhead desk scan as a standalone process."""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable

from .mcp_server import localize_3d

DEFAULT_INTERVAL_S = 120.0
log = logging.getLogger(__name__)


async def run_desk_scan(
    interval_s: float = DEFAULT_INTERVAL_S,
    *,
    provider: str = "auto",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Scan immediately, then on a fixed cadence without overlapping scans."""
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("Scan interval must be finite and positive.")

    deadline = clock()
    after_frame_ref = None
    while True:
        try:
            scan = await localize_3d(camera="top", denoise=False, provider=provider, after_frame_ref=after_frame_ref)
            if scan["stale_frame"]:
                log.warning("No new overhead frame; skipped %s", scan["frame_ref"])
            else:
                after_frame_ref = scan["frame_ref"]
                detections = scan["detections"]
                written = sum("sighting" in detection for detection in detections)
                report = log.warning if written != len(detections) else log.info
                report("Recorded %d/%d sightings from %s", written, len(detections), after_frame_ref)
        except Exception:
            log.exception("Desk scan failed; retrying at the next interval")

        deadline += interval_s
        now = clock()
        if deadline < now:
            deadline += math.ceil((now - deadline) / interval_s) * interval_s
        await sleep(max(0.0, deadline - now))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", metavar="SECONDS", type=float, default=DEFAULT_INTERVAL_S,
                        help="time between scans (default: %(default)s seconds)")
    parser.add_argument("--provider", default="auto", help="vision provider, e.g. auto or opencv (offline)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(run_desk_scan(args.interval, provider=args.provider))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
