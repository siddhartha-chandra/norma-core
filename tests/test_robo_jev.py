"""Mid-reach safety and recovery against the internal simulated desk and Station bridge."""
import asyncio
import time

import pytest

from desk_core import mcp_server as srv
from desk_core.calib_store import CalibrationStore
from desk_core.interfaces import SceneObstacle
from desk_core.robo_jev import RoboJevGatekeeper, guarded_recover


def test_recovery_rechecks_interrupted_path_without_caller_coordinates(calib, mock_bridge):
    gatekeeper = RoboJevGatekeeper()
    goal = (0.30, 0.0)
    obstacles = []

    async def simulate():
        state = await mock_bridge.get_state()
        start = calib.fk_ee_pose(calib.ticks_to_rad(state.motors))[:2, 3]
        x, y = (start + goal) / 2
        obstacles[:] = [SceneObstacle("mug", x=x, y=y, radius_m=0.04)]
        gatekeeper.interruption = {**gatekeeper.evaluate(obstacles=obstacles, target_xy=goal, start_xy=start),
                                   "target_xy": list(goal)}
        await mock_bridge.abort(source="reflex")

        async def recover():
            return await guarded_recover(gatekeeper, mock_bridge, calib, obstacles_fn=lambda: obstacles,
                                         obstacle_wait_timeout_s=0.03, poll_interval_s=0.01)

        blocked = await recover()
        assert not blocked["ok"] and blocked["reason"] == "obstacle_timeout", blocked
        assert mock_bridge.abort_source == "reflex"
        obstacles[:] = [SceneObstacle("mug", x=goal[0], y=goal[1] + 0.35, radius_m=0.04)]
        recovered = await recover()
        assert recovered["ok"] and recovered["target_xy"] == list(goal), recovered
        assert mock_bridge.abort_source is None

    asyncio.run(simulate())


@pytest.mark.parametrize("trigger_camera", ["top", "wrist"])
def test_eight_cm_slide_halts_stream_and_allows_a_fresh_reach(mock_bridge, monkeypatch, trigger_camera):
    monkeypatch.setenv("DESK_MOCK_SYNTHETIC", "1")
    calib = CalibrationStore()
    calib.data.pop("H_table", None)
    for camera, focal, height in [("top", 680.0, 0.65), ("wrist", 520.0, 0.25)]:
        calib.data[f"K_{camera}"] = [[focal, 0, 320], [0, focal, 240], [0, 0, 1]]
        calib.data[f"dist_{camera}"] = [0] * 5
        calib.data[f"T_world_cam_{camera}"] = [[0, -1, 0, .28], [-1, 0, 0, 0], [0, 0, -1, height], [0, 0, 0, 1]]
    calib.data["z_table"] = 0.0
    gatekeeper = RoboJevGatekeeper()
    monkeypatch.setattr(srv, "_bridge", mock_bridge)
    monkeypatch.setattr(srv, "_calib", calib)
    monkeypatch.setattr(srv, "_robo_jev", gatekeeper)
    monkeypatch.setattr(srv, "_obstacles", {})
    monkeypatch.setattr(srv, "_logger", None)

    x, y, z = 0.27, -0.02, 0.0
    mock_bridge.add_mock_object("keys", x, y, z, radius_m=0.015)
    mock_bridge.add_mock_object("mug", 0.37, 0.18, z, color="brown", shape="cylinder")
    first_command = asyncio.Event()
    sends = []
    hold_at = None
    send = mock_bridge.send_joint_targets
    abort = mock_bridge.abort

    async def record_send(*args, **kwargs):
        result = await send(*args, **kwargs)
        sends.append(time.perf_counter())
        first_command.set()
        return result

    async def record_abort(*args, **kwargs):
        nonlocal hold_at
        result = await abort(*args, **kwargs)
        hold_at = time.perf_counter()
        return result

    monkeypatch.setattr(mock_bridge, "send_joint_targets", record_send)
    monkeypatch.setattr(mock_bridge, "abort", record_abort)

    async def simulate():
        for camera in ("top", "wrist"):
            if camera != trigger_camera:
                mock_bridge.set_frame_override(camera, await mock_bridge.get_frame(camera))
            pixel = calib.project_world_to_pixel((x, y, z), camera=camera)
            assert (await srv.robo_jev_arm_watch(*pixel, query="blue keys", camera=camera))["armed"]

        before = (await mock_bridge.get_state()).positions()
        reach = asyncio.create_task(srv.move_to_pose(x, y, 0.15))
        await asyncio.wait_for(first_command.wait(), timeout=2)
        await asyncio.sleep(0.25)
        assert not reach.done()
        assert (await mock_bridge.get_state()).positions() != before

        moved_at = time.perf_counter()
        mock_bridge.add_mock_object("keys", x, y + 0.08, z, radius_m=0.015)
        result = await reach
        assert result["ok"] is False, result
        assert result["reflex"]["reason"] == "target_shifted", result
        assert result["reflex"]["camera"] == trigger_camera, result
        assert hold_at is not None
        latency_ms = (hold_at - moved_at) * 1000
        assert latency_ms < 150, f"{latency_ms:.1f} ms from 8 cm slide to Station hold"
        assert all(stamp < hold_at for stamp in sends)
        assert result["reflex"]["abort_result"]["positions"] == (await mock_bridge.get_state()).positions()
        await srv.observe(views=["top"], denoise=False)
        assert mock_bridge.abort_source == "reflex"

        mock_bridge.set_frame_override("top", None)
        mock_bridge.set_frame_override("wrist", None)
        recovery = await srv.robo_jev_guard()
        assert recovery["ok"] and recovery["action"] == "recalibrated", recovery
        assert recovery["world"]["y"] == pytest.approx(y + 0.08, abs=0.01)
        assert mock_bridge.abort_source is None
        replanned = await srv.move_to_pose(recovery["world"]["x"], recovery["world"]["y"], 0.15)
        assert replanned["ok"], replanned
        assert mock_bridge.get_mock_scene_state()["ee_pos"]["y"] == pytest.approx(y + 0.08, abs=0.01)
        await mock_bridge.abort()
        assert (await srv.robo_jev_guard())["reason"] == "manual_estop"
        assert mock_bridge.abort_source == "manual"
        print(f"{trigger_camera}: 8 cm slide to Station hold {latency_ms:.1f} ms; re-localized and completed a fresh reach")

    asyncio.run(simulate())
