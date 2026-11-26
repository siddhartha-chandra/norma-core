"""Tests for the desk object sightings history store (issue #26)."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass

import pytest

from desk_core import mcp_server as srv
from desk_core.sightings import DEFAULT_NORMFS_QUEUE, Sighting, SightingsStore
from desk_core.station_bridge import LiveStationBridge, MockStationBridge


@dataclass
class _FakeEntryId:
    ID: bytes


@dataclass
class _FakeEntry:
    ID: _FakeEntryId
    Data: memoryview


class _FakeQueueRead:
    def __init__(self, entries: list[_FakeEntry]):
        self.data: asyncio.Queue = asyncio.Queue()
        self.err = None
        for entry in entries:
            self.data.put_nowait(entry)
        self.data.put_nowait(None)


class _FakeNormFSClient:
    """In-memory stand-in for `station_py.Client`."""

    def __init__(self, *, connected: bool = True, setup_done: bool = True, fail_write: Exception | None = None):
        self.connected = connected
        self.setup_done = setup_done
        self.fail_write = fail_write
        self.queues: dict[str, list[tuple[bytes, bytes]]] = {}

    async def enqueue(self, queue_id: str, data: bytes) -> bytes:
        if self.fail_write is not None:
            raise self.fail_write
        bucket = self.queues.setdefault(queue_id, [])
        entry_id = len(bucket).to_bytes(8, "little")
        bucket.append((entry_id, bytes(data)))
        return entry_id

    def read_from_tail(self, queue_id: str, offset: bytes, limit: int, step: int, buf_size: int) -> _FakeQueueRead:
        bucket = self.queues.get(queue_id, [])
        if not bucket:
            return _FakeQueueRead([])
        shift = int.from_bytes(offset, "little") if offset else 0
        start = max(0, len(bucket) - 1 - shift)
        return self.read_from_offset(queue_id, start.to_bytes(8, "little"), limit, step, buf_size)

    def read_from_offset(self, queue_id: str, offset: bytes, limit: int, step: int, buf_size: int) -> _FakeQueueRead:
        bucket = self.queues.get(queue_id, [])
        start = int.from_bytes(offset, "little")
        selected = bucket[start : start + limit : max(1, step)]
        entries = [_FakeEntry(_FakeEntryId(eid), memoryview(payload)) for eid, payload in selected]
        return _FakeQueueRead(entries)


def test_normfs_primary_write_and_read(tmp_path):
    client = _FakeNormFSClient()
    db_path = tmp_path / "sightings.db"
    store = SightingsStore(normfs_client=client, db_path=db_path)

    res1 = asyncio.run(
        store.record("keys", 0.27, -0.06, time_s=1700000001.5, frame_ref="normfs://usbvideo/abc#01", z=0.0, camera="top", confidence=0.95)
    )
    res2 = asyncio.run(
        store.record("mug", 0.30, 0.02, time_s=1700000005.0, frame_ref="normfs://usbvideo/abc#02", z=0.0, camera="top", confidence=0.88)
    )

    assert res1.backend == "normfs"
    assert res1.fallback_reason is None
    assert res2.backend == "normfs"
    assert not db_path.exists()

    raw_entries = client.queues[DEFAULT_NORMFS_QUEUE]
    assert len(raw_entries) == 2
    first_payload = json.loads(raw_entries[0][1].decode("utf-8"))
    assert first_payload["label"] == "keys"
    assert first_payload["x"] == pytest.approx(0.27)
    assert first_payload["y"] == pytest.approx(-0.06)
    assert first_payload["time"] == pytest.approx(1700000001.5)
    assert first_payload["frame_ref"] == "normfs://usbvideo/abc#01"

    all_sightings = asyncio.run(store.read_normfs())
    assert [s.label for s in all_sightings] == ["mug", "keys"]

    only_keys = asyncio.run(store.read_normfs(label="KEYS"))
    assert len(only_keys) == 1
    assert only_keys[0].frame_ref == "normfs://usbvideo/abc#01"


def test_sqlite_fallback_when_normfs_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("DESK_SIGHTINGS_NORMFS", "0")
    client = _FakeNormFSClient()
    db_path = tmp_path / "sightings.db"
    store = SightingsStore(normfs_client=client, db_path=db_path)

    res = asyncio.run(
        store.record("keys", 0.27, -0.06, time_s=1700000100.25, frame_ref="mock://top#1", z=0.0, camera="top", confidence=0.9)
    )

    assert res.backend == "sqlite"
    assert res.fallback_reason == "normfs_disabled"
    assert client.queues == {}
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT label, x, y, time, frame_ref FROM sightings").fetchall()
    conn.close()
    assert rows == [("keys", 0.27, -0.06, 1700000100.25, "mock://top#1")]

    loaded = store.read_sqlite()
    assert len(loaded) == 1
    assert loaded[0] == Sighting(
        label="keys",
        x=0.27,
        y=-0.06,
        time=1700000100.25,
        frame_ref="mock://top#1",
        z=0.0,
        camera="top",
        confidence=0.9,
    )


@pytest.mark.parametrize(
    "client,expected_reason",
    [
        (None, "normfs_unavailable"),
        (_FakeNormFSClient(connected=False), "normfs_unavailable"),
        (_FakeNormFSClient(setup_done=False), "normfs_unavailable"),
        (_FakeNormFSClient(fail_write=RuntimeError("broken pipe")), "RuntimeError: broken pipe"),
    ],
)
def test_sqlite_fallback_when_normfs_unavailable_or_errors(tmp_path, client, expected_reason):
    db_path = tmp_path / "sightings.db"
    store = SightingsStore(normfs_client=client, db_path=db_path, normfs_enabled=True)

    res = asyncio.run(store.record("pen", 0.15, 0.08, time_s=1700000200.0, frame_ref="mock://top#3"))

    assert res.backend == "sqlite"
    assert res.fallback_reason == expected_reason
    rows = store.read_sqlite()
    assert len(rows) == 1
    assert rows[0].label == "pen"
    assert rows[0].x == pytest.approx(0.15)
    assert rows[0].y == pytest.approx(0.08)
    assert rows[0].frame_ref == "mock://top#3"


def test_list_sightings_merges_normfs_and_sqlite_with_filters(tmp_path):
    client = _FakeNormFSClient()
    db_path = tmp_path / "sightings.db"
    store = SightingsStore(normfs_client=client, db_path=db_path, normfs_enabled=True)

    # First sighting lands in NormFS
    r1 = asyncio.run(store.record("keys", 0.20, 0.01, time_s=100.0, frame_ref="normfs://cam#1"))
    assert r1.backend == "normfs"

    # Simulate temporary NormFS outage -> second sighting lands in SQLite
    client.connected = False
    r2 = asyncio.run(store.record("keys", 0.22, 0.03, time_s=200.0, frame_ref="normfs://cam#2"))
    assert r2.backend == "sqlite"

    # NormFS recovers -> third sighting lands in NormFS
    client.connected = True
    r3 = asyncio.run(store.record("mug", 0.30, 0.05, time_s=300.0, frame_ref="normfs://cam#3"))
    assert r3.backend == "normfs"

    merged = asyncio.run(store.list_sightings())
    assert [(s.label, s.time) for s in merged] == [("mug", 300.0), ("keys", 200.0), ("keys", 100.0)]

    before = asyncio.run(store.list_sightings(label="keys", before_time=150.0))
    assert len(before) == 1
    assert before[0].time == 100.0


@pytest.mark.parametrize("station_available", [True, False])
def test_mcp_history_reads_normfs_and_sqlite_on_cold_start(tmp_path, monkeypatch, mock_bridge, station_available):
    client = _FakeNormFSClient()
    db_path = tmp_path / "sightings.db"
    writer = SightingsStore(normfs_client=client, db_path=db_path, normfs_enabled=True)
    mock_bridge.client = client

    def create_bridge(*, follow_motor_rx):
        if not station_available:
            async def connect():
                raise OSError("Station offline")
            monkeypatch.setattr(mock_bridge, "connect", connect)
        return mock_bridge

    monkeypatch.setenv("DESK_SIGHTINGS_DB", str(db_path))
    monkeypatch.setenv("DESK_SIGHTINGS_NORMFS", "1")
    monkeypatch.setattr(srv, "_bridge", None)
    monkeypatch.setattr(srv, "_connect_cooldown_until", 0.0)
    monkeypatch.setattr(srv, "_sightings", None)
    monkeypatch.setattr(srv, "create_bridge", create_bridge)

    async def run():
        client.connected = False
        await writer.record("keys", 0.20, 0.01, time_s=100.0, frame_ref="mock://top#1")
        client.connected = True
        await writer.record("keys", 0.23, 0.04, time_s=200.0, frame_ref="normfs://cam#2")

        latest = await srv.last_seen(" KEYS ")
        assert latest["found"] is True
        assert (latest["time"], latest["frame_ref"]) == (
            (200.0, "normfs://cam#2") if station_available else (100.0, "mock://top#1")
        )
        assert await srv.where_was("keys", 150.0) == {
            "found": True, "label": "keys", "x": 0.20, "y": 0.01, "time": 100.0, "frame_ref": "mock://top#1",
        }

    asyncio.run(run())


def test_mcp_history_finds_matches_beyond_first_normfs_page(tmp_path, monkeypatch, mock_bridge):
    client = _FakeNormFSClient()
    store = SightingsStore(normfs_client=client, db_path=tmp_path / "sightings.db", normfs_enabled=True)
    monkeypatch.setattr(srv, "_sightings", store)
    monkeypatch.setattr(srv, "_bridge", mock_bridge)

    async def run():
        await store.record("keys", 0.2, 0.1, time_s=20.0, frame_ref="old-keys")
        await store.record("pen", 0.3, 0.1, time_s=30.0, frame_ref="old-pen")
        for _ in range(1000):
            await store.record("mug", 0.2, 0.1, time_s=40.0, frame_ref="mug")
        await store.record("pen", 0.1, 0.2, time_s=50.0, frame_ref="new-pen")

        client.connected = False
        await store.record("keys", 0.4, 0.1, time_s=10.0, frame_ref="older-sqlite-keys")
        client.connected = True

        assert (await srv.last_seen("keys"))["frame_ref"] == "old-keys"
        assert (await srv.where_was("pen", 30.0))["frame_ref"] == "old-pen"
        assert (await srv.last_seen("pen"))["frame_ref"] == "new-pen"

    asyncio.run(run())


def test_sighting_validation(tmp_path):
    store = SightingsStore(db_path=tmp_path / "sightings.db", normfs_enabled=False)
    with pytest.raises(ValueError):
        asyncio.run(store.record("   ", 0.1, 0.2, time_s=1.0, frame_ref="f1"))
    with pytest.raises(ValueError):
        asyncio.run(store.record("keys", float("nan"), 0.2, time_s=1.0, frame_ref="f1"))


def test_bridge_last_frame_ref():
    mock = MockStationBridge(num_joints=8)
    assert mock.last_frame_ref("top") is None
    asyncio.run(mock.get_frame("top"))
    assert mock.last_frame_ref("top") == "mock://top#1"
    asyncio.run(mock.get_frame("top"))
    assert mock.last_frame_ref("top") == "mock://top#2"
    asyncio.run(mock.get_frame("wrist"))
    assert mock.last_frame_ref("wrist") == "mock://wrist#1"

    live = LiveStationBridge(follow_motor_rx=False)
    live._frames["cam_top_uid"] = b"\xff\xd8\xff\xd9"
    live._frame_order.append("cam_top_uid")
    live._key_to_qid["cam_top_uid"] = "usbvideo/0123456789abcdef0123456789abcdef"
    live._record_frame_receipt("cam_top_uid", b"\xff\xd8\xff\xd9", 123456789, b"\x00\x2a")
    assert live.last_frame_ref("top") == "normfs://usbvideo/0123456789abcdef0123456789abcdef#002a"


def test_mcp_localize_3d_logs_to_sqlite_in_mock_simulation(tmp_path, monkeypatch, mock_bridge):
    monkeypatch.setenv("DESK_MOCK_SYNTHETIC", "1")
    monkeypatch.setenv("DESK_SIGHTINGS_DB", str(tmp_path / "sightings.db"))
    monkeypatch.setattr(srv, "_bridge", mock_bridge)
    monkeypatch.setattr(srv, "_logger", None)
    store = SightingsStore(client_provider=lambda: getattr(srv._bridge, "client", None), db_path=tmp_path / "sightings.db")
    monkeypatch.setattr(srv, "_sightings", store)

    res = asyncio.run(srv.localize_3d(query="blue keys", camera="top", use_gemini=False, denoise=False))

    assert res["found"] is True
    assert res["sighting"]["backend"] == "sqlite"
    assert res["sighting"]["frame_ref"] == "mock://top#1"

    rows = store.read_sqlite()
    assert len(rows) == 1
    assert rows[0].label == res["label"]
    assert rows[0].x == pytest.approx(res["world"]["x"])
    assert rows[0].y == pytest.approx(res["world"]["y"])
    assert rows[0].frame_ref == "mock://top#1"
    assert rows[0].time == pytest.approx(res["sighting"]["time"])


def test_mcp_localize_3d_logs_to_normfs_when_available_and_skips_when_not_found(tmp_path, monkeypatch, mock_bridge):
    monkeypatch.setenv("DESK_MOCK_SYNTHETIC", "1")
    fake_client = _FakeNormFSClient()
    mock_bridge.client = fake_client
    db_path = tmp_path / "sightings.db"
    store = SightingsStore(client_provider=lambda: getattr(srv._bridge, "client", None), db_path=db_path)
    monkeypatch.setattr(srv, "_bridge", mock_bridge)
    monkeypatch.setattr(srv, "_logger", None)
    monkeypatch.setattr(srv, "_sightings", store)

    res = asyncio.run(srv.localize_3d(query="blue keys", camera="top", use_gemini=False, denoise=False))
    assert res["found"] is True
    assert res["sighting"]["backend"] == "normfs"
    assert not db_path.exists()

    normfs_rows = asyncio.run(store.read_normfs())
    assert len(normfs_rows) == 1
    assert normfs_rows[0].label == res["label"]
    assert normfs_rows[0].x == pytest.approx(res["world"]["x"])
    assert normfs_rows[0].y == pytest.approx(res["world"]["y"])
    assert normfs_rows[0].frame_ref == "mock://top#1"

    # With NormFS disabled, the next scan goes to SQLite instead.
    monkeypatch.setenv("DESK_SIGHTINGS_NORMFS", "0")
    res2 = asyncio.run(srv.localize_3d(query="red cube", camera="top", use_gemini=False, denoise=False))
    assert res2["found"] is True
    assert res2["sighting"]["backend"] == "sqlite"
    assert res2["sighting"]["fallback_reason"] == "normfs_disabled"
    assert len(store.read_sqlite()) == 1
    assert len(fake_client.queues[DEFAULT_NORMFS_QUEUE]) == 1

    # A blank frame (not found) must not log any sighting.
    import cv2
    import numpy as np

    blank = np.full((480, 640, 3), 240, dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", blank)
    assert ok
    mock_bridge.set_frame_override("top", enc.tobytes())
    res3 = asyncio.run(srv.localize_3d(query="blue keys", camera="top", use_gemini=False, denoise=False))
    assert res3["found"] is False
    assert "sighting" not in res3
    assert len(store.read_sqlite()) == 1
