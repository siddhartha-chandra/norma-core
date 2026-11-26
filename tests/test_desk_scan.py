import asyncio
from unittest.mock import AsyncMock

import pytest

from desk_core import desk_scan, mcp_server, vision_3d
from desk_core.sightings import SightingsStore
from desk_core.station_bridge import LiveStationBridge


def test_scene_scan_never_uses_wrist_as_overhead(monkeypatch):
    bridge = LiveStationBridge("localhost", 8888, cam_top="overhead-id", cam_wrist="wrist-id")
    bridge._frames["wrist-id"] = b"wrist-frame"
    bridge._frame_order.append("wrist-id")
    monkeypatch.setattr(mcp_server, "_bridge", bridge)
    monkeypatch.setattr(mcp_server, "_logger", None)

    with pytest.raises(RuntimeError, match="No uniquely configured top camera"):
        asyncio.run(mcp_server.localize_3d(provider="opencv"))


def test_default_ten_minute_scan_in_mock_simulation(tmp_path, monkeypatch, mock_bridge):
    monkeypatch.setenv("DESK_MOCK_SYNTHETIC", "1")
    monkeypatch.setattr(mcp_server, "_bridge", mock_bridge)
    monkeypatch.setattr(mcp_server, "_logger", None)
    store = SightingsStore(db_path=tmp_path / "sightings.db", normfs_enabled=False)
    monkeypatch.setattr(mcp_server, "_sightings", store)
    motion = {}
    for name in ("send_joint_targets", "set_torque", "grasp_with_verify", "abort"):
        motion[name] = AsyncMock(side_effect=AssertionError(f"Unexpected motion: {name}"))
        monkeypatch.setattr(mock_bridge, name, motion[name])
    initial_positions = [m.position for m in asyncio.run(mock_bridge.get_state()).motors]
    initial_scene = mock_bridge.get_mock_scene_state()
    now = 0.0
    frames = []

    async def sleep(delay):
        nonlocal now
        rows = store.read_sqlite()
        newest = rows[0].frame_ref
        sightings = [row for row in rows if row.frame_ref == newest]
        frames.append(newest)
        assert now == 120 * (len(frames) - 1)
        assert len(sightings) == len(initial_scene["objects"])
        assert len(rows) == len(frames) * len(sightings)
        assert len({(row.x, row.y) for row in sightings}) == len(sightings)
        assert len({row.time for row in sightings}) == 1
        assert {row.camera for row in sightings} == {"top"}
        if now == 600:
            raise asyncio.CancelledError
        assert delay == 120
        now += delay

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(desk_scan.run_desk_scan(provider="opencv", clock=lambda: now, sleep=sleep))

    assert frames == [f"mock://top#{n}" for n in range(1, 7)]
    for command in motion.values():
        command.assert_not_called()
    assert [m.position for m in asyncio.run(mock_bridge.get_state()).motors] == initial_positions
    assert mock_bridge.get_mock_scene_state() == initial_scene


def test_scene_scan_records_named_detections_and_skips_repeated_frame(tmp_path, monkeypatch, mock_bridge):
    monkeypatch.setenv("DESK_MOCK_SYNTHETIC", "1")
    monkeypatch.setattr(mcp_server, "_bridge", mock_bridge)
    monkeypatch.setattr(mcp_server, "_logger", None)
    monkeypatch.setattr(mock_bridge, "last_frame_ref", lambda camera: "mock://top#cached")
    store = SightingsStore(db_path=tmp_path / "sightings.db", normfs_enabled=False)
    monkeypatch.setattr(mcp_server, "_sightings", store)
    calls = []

    def call_vision(jpeg, query, provider, model):
        calls.append((query, provider))
        return vision_3d._parse_vlm_json(
            """{"detections": [
                {"found": true, "label": "blue keys", "point": [400, 200], "box_2d": [360, 180, 440, 220]},
                {"found": true, "label": "brown mug", "point": [600, 500], "box_2d": [550, 460, 650, 540]},
                {"found": false, "label": "wallet"},
                {"found": true, "label": "notebook"}
            ]}""",
            query,
        )

    monkeypatch.setattr(vision_3d, "_call_vlm_sync", call_vision)

    async def scan_twice():
        first = await mcp_server.localize_3d(provider="gemini", denoise=False)
        second = await mcp_server.localize_3d(provider="gemini", denoise=False, after_frame_ref=first["frame_ref"])
        return first, second

    first, second = asyncio.run(scan_twice())
    assert calls == [(None, "gemini")]
    assert {det["label"] for det in first["detections"]} == {"blue keys", "brown mug"}
    assert first["stale_frame"] is False
    assert second == {"detections": [], "frame_ref": first["frame_ref"], "stale_frame": True}
    rows = store.read_sqlite()
    assert sorted(row.label for row in rows) == ["blue keys", "brown mug"]
    assert {row.frame_ref for row in rows} == {first["frame_ref"]}


def test_scan_retries_on_the_next_tick_after_a_failure(monkeypatch):
    scan = AsyncMock(side_effect=[
        RuntimeError("camera unavailable"),
        {"detections": [], "frame_ref": "mock://top#1", "stale_frame": False},
    ])
    monkeypatch.setattr(desk_scan, "localize_3d", scan)
    now = 0.0

    async def sleep(delay):
        nonlocal now
        assert delay == 2.0
        now += delay
        if now == 4.0:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(desk_scan.run_desk_scan(2.0, clock=lambda: now, sleep=sleep))
    assert scan.await_count == 2
