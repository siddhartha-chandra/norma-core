#!/usr/bin/env python3
"""End-to-end verification for Robo-Jev (#12) and Mission Control Dashboard (#14)."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ["DESK_MOCK"] = "1"
os.environ["DESK_NO_DOTENV"] = "1"
os.environ["DESK_TTS"] = "0"
os.environ["STATION_HOST"] = ""

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from desk_core.dashboard import app
from desk_core.interfaces import SceneObstacle
from desk_core.robo_jev import RoboJevGatekeeper, check_reflex_and_abort
from desk_core.station_bridge import MockStationBridge


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{' — ' + detail if detail else ''}")
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def _make_scene(cx: int, cy: int) -> bytes:
    img = np.full((480, 640, 3), 225, dtype=np.uint8)
    cv2.rectangle(img, (cx - 25, cy - 18), (cx + 25, cy + 18), (200, 70, 20), -1)
    cv2.circle(img, (cx, cy), 6, (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


async def test_robo_jev() -> None:
    print("Robo-Jev <150 ms Reflex Gatekeeper (#12)")
    bridge = MockStationBridge()
    await bridge.connect()
    gk = RoboJevGatekeeper()

    jpeg0 = _make_scene(240, 240)
    armed = gk.arm_watch(jpeg0, pixel=(240.0, 240.0), query="keys", depth_m=0.62)
    check("arm_watch captures reference crop", armed["armed"] is True, str(armed["crop_shape"]))

    # Unshifted check
    rep0 = gk.evaluate(curr_jpeg=jpeg0, obstacles=[SceneObstacle("mug", x=0.35, y=0.20, radius_m=0.045)], target_xy=(0.25, 0.0))
    check("unshifted target passes in <150 ms", rep0["ok"] is True and rep0["latency_ms"] < 150.0, f"{rep0['latency_ms']} ms")

    # Slide target 80 px sideways (simulated 8 cm perturbation) & override bridge frame
    jpeg_shifted = _make_scene(320, 240)

    async def _shifted_frame(camera: str = "top", denoise: bool = True) -> bytes:
        return jpeg_shifted

    bridge.get_frame = _shifted_frame  # type: ignore[method-assign]
    abort_rep = await check_reflex_and_abort(gk, bridge, target_xy=(0.25, 0.0))
    check(
        "shifted target triggers target_shifted + abort in <150 ms",
        abort_rep["target_shifted"] is True and abort_rep["aborted"] is True and abort_rep["latency_ms"] < 150.0 and abort_rep["shift_px"] >= 60.0,
        f"shift_px={abort_rep['shift_px']} latency={abort_rep['latency_ms']}ms",
    )

    # Obstacle path_clear check
    rep_block = gk.evaluate(obstacles=[SceneObstacle("mug", x=0.20, y=0.0, radius_m=0.05)], start_xy=(0.10, 0.0), target_xy=(0.30, 0.0))
    check("obstacle along path triggers path_blocked", rep_block["ok"] is False and rep_block["blocker"] == "mug", str(rep_block))


def test_dashboard() -> None:
    print("\nMission Control Dashboard + E-STOP + TTS (#14)")
    with TestClient(app) as client:
        r_idx = client.get("/")
        check("GET / serves Mission Control HTML", r_idx.status_code == 200 and "DESK-CORE MISSION CONTROL" in r_idx.text)

        r_tel = client.get("/api/telemetry")
        tel = r_tel.json()
        check("GET /api/telemetry returns 8 motors + robo_jev", r_tel.status_code == 200 and len(tel["motors"]) == 8 and "robo_jev" in tel)

        r_top = client.get("/api/frame/top.jpg")
        r_wrist = client.get("/api/frame/wrist.jpg")
        check("GET /api/frame/top.jpg & wrist.jpg return JPEGs", r_top.status_code == 200 and r_wrist.status_code == 200 and len(r_top.content) > 500)

        r_loc = client.post("/api/localize", json={"query": "keys", "camera": "top", "use_gemini": False})
        check("POST /api/localize returns 3D world pose", r_loc.status_code == 200 and "world" in r_loc.json())

        r_arm = client.post("/api/robo_jev/arm")
        r_chk = client.post("/api/robo_jev/check")
        check("POST /api/robo_jev/arm & check succeed", r_arm.status_code == 200 and r_chk.status_code == 200 and r_chk.json()["latency_ms"] < 150.0)

        r_fl = client.post("/api/first_light")
        check("POST /api/first_light completes routine", r_fl.status_code == 200 and r_fl.json()["ok"] is True)

        r_est = client.post("/api/estop")
        check("POST /api/estop engages hold", r_est.status_code == 200 and r_est.json()["ok"] is True)

        r_logs = client.get("/api/logs")
        check("GET /api/logs returns dashboard events", r_logs.status_code == 200 and len(r_logs.json()["dashboard_events"]) >= 3)


def main() -> int:
    asyncio.run(test_robo_jev())
    test_dashboard()
    print("\nROBO-JEV & MISSION CONTROL DASHBOARD PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
