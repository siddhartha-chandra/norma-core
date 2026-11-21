#!/usr/bin/env python3
"""Offline test for LiveStationBridge auto-reconnect (Bear-Bonnes issue #3).

No Station or hardware needed: a fake station_py.Client reproduces exactly
what a Station restart does — the socket comes back (station_py handles that)
but every follow stream dies with an error on its error queue and never pushes
to the data queue again. Verifies the supervisor rebuilds subscriptions and
that motion commands are blocked while telemetry is stale.

    uv run --offline python scripts/test_reconnect.py
"""
from __future__ import annotations

import asyncio
import os
import struct
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from desk_core.station_bridge import (  # noqa: E402
    CMD_STALE_LIMIT_S,
    FULL_STATE_LEN,
    RAM_PRESENT_CURRENT,
    RAM_PRESENT_POSITION,
    RAM_STATUS,
    LiveStationBridge,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{' — ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def rx_entry(position: int, motor_id: int = 1) -> types.SimpleNamespace:
    buf = bytearray(FULL_STATE_LEN)
    struct.pack_into("<H", buf, RAM_PRESENT_POSITION, position)
    struct.pack_into("<H", buf, RAM_PRESENT_CURRENT, 30)
    buf[RAM_STATUS] = 0
    return types.SimpleNamespace(Data=bytes(buf))


class FakeRxReader:
    def __init__(self, data):
        self._d = bytes(data)

    def get_data(self) -> bytes:
        return self._d

    def get_bus(self):
        return types.SimpleNamespace(get_serial_number=lambda: "TESTBUS")

    def get_motor_id(self) -> int:
        return 1


class FakeInferenceReader:
    def __init__(self, data):
        pass

    def get_buses(self):
        return []


class FakeClient:
    """Mimics station_py.Client across a Station restart."""

    def __init__(self):
        self.connected = True
        self.setup_done = True
        self.follow_log: list[tuple[str, asyncio.Queue, asyncio.Queue]] = []

    def follow(self, qid: str, q: asyncio.Queue) -> asyncio.Queue:
        eq: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.follow_log.append((qid, q, eq))
        return eq

    def latest_queue(self, qid: str) -> asyncio.Queue:
        return next(q for (c, q, _e) in reversed(self.follow_log) if c == qid)

    async def wait_ready(self, timeout=None):
        return

    def restart_station(self):
        """Socket dies then returns; every live subscription errors out."""
        self.connected = False
        self.setup_done = False
        for _qid, _data_q, err_q in self.follow_log:
            err_q.put_nowait(RuntimeError("read stream closed"))
        self.connected = True
        self.setup_done = True


async def main() -> int:
    print("LiveStationBridge auto-reconnect (fake Station)")
    bridge = LiveStationBridge(host="127.0.0.1", port=1)
    client = FakeClient()
    bridge.client = client
    # Stand in for the protobuf modules connect() would inject.
    bridge._st3215 = types.SimpleNamespace(RxEnvelopeReader=FakeRxReader, InferenceStateReader=FakeInferenceReader)
    bridge._inference_pb = types.SimpleNamespace(InferenceRxReader=lambda *a, **k: types.SimpleNamespace(get_entries=lambda: []))
    bridge._drivers = types.SimpleNamespace(QueueDataType=types.SimpleNamespace(QDT_USB_VIDEO_FRAMES=21))
    bridge._discover_local_video_queues = lambda max_age_s=0.0: []  # type: ignore[assignment]

    async def _noop_warmup(max_s: float = 0.0) -> None:
        return None

    bridge._warmup = _noop_warmup  # type: ignore[method-assign]

    # -- first link up ----------------------------------------------------------
    bridge._make_followers()
    bridge._start_workers()
    await client.latest_queue("st3215/rx").put(rx_entry(2000))
    await asyncio.sleep(0.2)
    st = await bridge.get_state()
    check("telemetry parsed into motor state", st.bus_serial == "TESTBUS" and st.motors[0].position == 2000)
    check("state carries link age", st.link_age_ms is not None and st.link_age_ms < 3000, f"age={st.link_age_ms}ms")
    check("link healthy → no resync needed", bridge._dead_reason() is None)
    check("socket up → link_status live", bridge.link_status()["socket_up"] is True)

    # -- Station restarts ---------------------------------------------------------
    client.restart_station()
    reason = bridge._dead_reason()
    check("dead subscription detected", reason is not None and "died" in str(reason), str(reason))

    bridge._last_rx_mono = time.monotonic() - (CMD_STALE_LIMIT_S + 1)
    try:
        await bridge.send_joint_targets({1: 2100})
        check("motion blocked on stale telemetry", False, "send_joint_targets did not raise")
    except RuntimeError as exc:
        check("motion blocked on stale telemetry", "stale" in str(exc), str(exc)[:60])

    # Stale E-STOP must fall back to cutting torque rather than failing or writing stale GoalPosition
    sent_packs = []

    async def _fake_send(_client, pack):
        sent_packs.append(pack)

    bridge._commands = types.SimpleNamespace(DriverCommand=lambda **kw: kw)
    bridge._drivers.StationCommandType = types.SimpleNamespace(STC_ST3215_COMMAND=1)
    bridge._st3215.ST3215SyncWriteCommand = lambda **kw: kw
    bridge._st3215.ST3215SyncWriteCommand_MotorWrite = lambda **kw: kw
    bridge._st3215.Command = lambda **kw: types.SimpleNamespace(encode=lambda: b"cmd")
    bridge._send_commands = _fake_send
    estop_res = await bridge.abort()
    check("stale abort falls back to torque_off", estop_res.get("mode") == "torque_off_stale" and len(sent_packs) == 1)

    n_rx_before = sum(1 for c, _q, _e in client.follow_log if c == "st3215/rx")
    await bridge._resync(reason)
    n_rx_after = sum(1 for c, _q, _e in client.follow_log if c == "st3215/rx")
    check("subscriptions re-issued after resync", n_rx_after == n_rx_before + 1, f"{n_rx_before} -> {n_rx_after}")
    check("resync clears flag", bridge._resyncing is False)

    new_q = client.latest_queue("st3215/rx")
    old_q = client.follow_log[0][1]
    check("consumers bound to fresh queues", new_q is not old_q)

    await new_q.put(rx_entry(1800))
    await asyncio.sleep(0.2)
    st2 = await bridge.get_state()
    check("post-restart telemetry flows", st2.motors[0].position == 1800, f"pos={st2.motors[0].position}")
    check("link healthy again", bridge._dead_reason() is None)

    for t in bridge._tasks:
        t.cancel()
    await asyncio.gather(*bridge._tasks, return_exceptions=True)

    if FAILURES:
        print(f"\nRECONNECT TEST FAIL: {len(FAILURES)} checks failed: {FAILURES}", file=sys.stderr)
        return 1
    print("\nRECONNECT TEST PASS — supervisor rebuilds subscriptions and refuses stale motion")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
