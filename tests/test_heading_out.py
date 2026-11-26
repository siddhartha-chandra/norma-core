import asyncio
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest

from desk_core import mcp_server as srv
from desk_core.dashboard import api_frame
from desk_core.station_bridge import LiveStationBridge, MockStationBridge


@pytest.fixture
def desk(monkeypatch):
    monkeypatch.setenv("DESK_MOCK_HEADING_OUT", "1")
    monkeypatch.delenv("DESK_MOCK_HIVE_SEED", raising=False)
    bridge = MockStationBridge()
    monkeypatch.setattr(srv, "_bridge", bridge)
    monkeypatch.setattr(srv, "_obstacles", {})
    monkeypatch.setattr(srv, "_unlocalized_obstacles", set())
    monkeypatch.setattr(srv, "_overlay_path", ())
    return bridge


def test_heading_out_dispatches_and_detours_with_mock_keys(desk):
    async def run():
        initial = desk.get_mock_scene_state()
        ticks = (await desk.get_state()).positions()
        command = await srv.plan_command("I’m heading out!")
        assert command["template"] == "heading_out"
        assert command["steps"] == [{"tool": "heading_out", "args": {"execute": True}}]

        preview = await srv.heading_out()
        assert preview["ok"], preview
        assert preview["detour"] in ("around", "over")
        assert desk.get_mock_scene_state() == initial
        assert (await desk.get_state()).positions() == ticks
        assert srv._obstacles["mug"].shape == "cylinder"
        assert srv._obstacles["departure_tray"].shape == "box"

        result = await srv.execute_sequence(command["steps"])
        assert result["ok"], result
        transfer = result["trace"][0]["result"]
        assert transfer["executed"] and transfer["detour"] == preview["detour"]
        assert transfer["transfer_flange_m"] == preview["transfer_flange_m"]
        assert (await desk.get_state()).positions() != ticks
        final = desk.get_mock_scene_state()
        assert final["held_object"] is None
        assert final["objects"]["mug"] == initial["objects"]["mug"]
        tray = final["objects"]["departure_tray"]
        keys = final["objects"]["keys"]
        assert keys["x"] == pytest.approx(tray["x"], abs=0.002)
        assert keys["y"] == pytest.approx(tray["y"], abs=0.002)
        assert keys["z"] > tray["z"]

    asyncio.run(run())


def test_xray_overhead_jpeg_uses_scene_and_preflighted_path(desk):
    async def run():
        raw = await desk.get_frame("top")
        wrist = await desk.get_frame("wrist")
        assert (await srv.get_frame(show_overlay=True, save_npy=False))[1].data == raw
        assert (await srv.heading_out())["ok"]
        overlaid = (await srv.get_frame(show_overlay=True, save_npy=False))[1].data
        assert (await api_frame("top", show_overlay=True)).body == overlaid
        assert (await srv.get_frame(save_npy=False))[1].data == raw
        assert (await srv.get_frame("wrist", show_overlay=True, save_npy=False))[1].data == wrist
        image = cv2.imdecode(np.frombuffer(overlaid, dtype=np.uint8), cv2.IMREAD_COLOR).astype(int)
        original = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR).astype(int)
        changed = np.max(abs(image - original), axis=2) > 55
        blue, green, red = image.transpose(2, 0, 1)
        assert np.count_nonzero(changed & (red > green + 90) & (red > blue + 90)) > 100
        assert np.count_nonzero(changed & (green > red + 90) & (green > blue + 90)) > 100

        await srv.update_scene_obstacle("mug", 0, 0, remove=True)
        assert not srv._overlay_path

    asyncio.run(run())


def test_blocked_heading_out_does_not_move_or_leave_a_green_path(desk, monkeypatch):
    async def run():
        assert (await srv.heading_out())["ok"]
        keys = desk.get_mock_scene_state()["objects"]["keys"]
        await srv.update_scene_obstacle("blocked keys", keys["x"], keys["y"], 0.10,
                                        radius_m=0.05, height_m=0.20)
        send = AsyncMock()
        monkeypatch.setattr(desk, "send_joint_targets", send)
        result = await srv.heading_out(execute=True)
        assert not result["ok"] and not result["executed"]
        assert "Collision" in result["problems"][0]
        assert not srv._overlay_path
        send.assert_not_called()

    asyncio.run(run())


def test_occupied_tray_is_rejected_before_any_mock_motion(desk, monkeypatch):
    tray = desk.get_mock_scene_state()["objects"]["departure_tray"]
    desk.add_mock_object("wallet", tray["x"], tray["y"], z=0.018, shape="wallet")
    initial = desk.get_mock_scene_state()
    send = AsyncMock()
    monkeypatch.setattr(desk, "send_joint_targets", send)

    result = asyncio.run(srv.heading_out(execute=True))

    assert not result["ok"] and not result["executed"]
    assert result["problems"] == ["Departure tray is occupied by wallet"]
    assert desk.get_mock_scene_state() == initial
    send.assert_not_called()


def test_live_detections_register_primitives_without_moving(monkeypatch):
    class CameraOnly:
        get_frame = AsyncMock(return_value=b"test camera frame")

    def detector(jpeg_bytes, query, calib, **kwargs):
        assert jpeg_bytes == b"test camera frame" and kwargs["camera"] == "top"
        return {"found": True, "backend": "gemini", "size_m": [0.08, 0.10],
                "world": {"x": 0.25, "y": 0.10 if query == "coffee mug" else -0.10}}

    monkeypatch.setattr(srv, "_bridge", CameraOnly())
    monkeypatch.setattr(srv, "_obstacles", {})
    monkeypatch.setattr(srv, "_unlocalized_obstacles", set())
    monkeypatch.setattr(srv, "_overlay_path", ())
    detection = AsyncMock(side_effect=detector)
    monkeypatch.setattr(srv, "localize_object_3d", detection)
    result = asyncio.run(srv.heading_out(execute=True))
    assert not result["ok"] and not result["executed"]
    assert [call.args[1] for call in detection.call_args_list] == ["coffee mug", "Departure Tray"]
    mug = srv._obstacles["mug"]
    tray = srv._obstacles["departure_tray"]
    assert mug.shape == "cylinder" and mug.y == 0.10
    assert tray.shape == "box" and tray.y == -0.10
    assert mug.z - mug.height_m / 2 == pytest.approx(srv._calib.z_table())
    assert tray.z - tray.size_xyz[2] / 2 == pytest.approx(srv._calib.z_table())


def test_missed_live_mug_is_retired_and_blocks_mujo_co_preflight(monkeypatch):
    live = LiveStationBridge.__new__(LiveStationBridge)
    live.get_frame = AsyncMock(return_value=b"test camera frame")
    live.get_state = AsyncMock()
    monkeypatch.setattr(srv, "_bridge", live)
    monkeypatch.setattr(srv, "_obstacles", {})
    monkeypatch.setattr(srv, "_unlocalized_obstacles", set())
    monkeypatch.setattr(srv, "_overlay_path", ())
    vision = AsyncMock(side_effect=[
        {"found": False},
        {"found": True, "backend": "gemini", "size_m": [0.08, 0.10], "world": {"x": 0.25, "y": -0.10}},
    ])
    monkeypatch.setattr(srv, "localize_object_3d", vision)

    async def run():
        await srv.update_scene_obstacle("mug", 0.20, 0.10)
        report = await srv.heading_out(execute=True)
        assert not report["executed"] and report["unlocalized_obstacles"] == ["mug"]
        assert set(srv._obstacles) == {"departure_tray"}
        plan = await srv.plan_and_move(0.20, 0.10, 0.15, 0.60)
        assert not plan["ok"] and plan["problems"] == ["Obstacle positions unknown: mug"]
        live.get_state.assert_not_awaited()
        await srv.update_scene_obstacle("mug", 0.21, 0.11)
        assert not srv._unlocalized_obstacles

    asyncio.run(run())
