"""Races between live motor startup and the reflex stop, with fake Station telemetry and writes."""
import asyncio
import logging
import time
from unittest.mock import Mock

import pytest

from desk_core.station_bridge import EStopLatched, LiveStationBridge, MotorState, RAM_GOAL_POSITION
from station_py import Client, new_station_client


@pytest.mark.parametrize("cancelled", [False, True])
def test_initial_station_setup_failure_stops_connection_task(monkeypatch, cancelled):
    writer = Mock()
    wait_ready = Client.wait_ready

    async def run():
        setup_started = asyncio.Event()
        setup_stopped = asyncio.Event()

        async def open_connection(*args):
            return asyncio.StreamReader(), writer

        async def stall_setup(self):
            setup_started.set()
            try:
                await asyncio.Future()
            finally:
                setup_stopped.set()

        async def wait_briefly(self, timeout):
            await setup_started.wait()
            await wait_ready(self, timeout=0.01)

        monkeypatch.setattr(asyncio, "open_connection", open_connection)
        monkeypatch.setattr(Client, "perform_initial_setup", stall_setup)
        monkeypatch.setattr(Client, "wait_ready", wait_briefly)

        connecting = asyncio.create_task(new_station_client("localhost", logging.getLogger(__name__)))
        await setup_started.wait()
        if cancelled:
            connecting.cancel()
        with pytest.raises(asyncio.CancelledError if cancelled else TimeoutError):
            await connecting

        assert setup_stopped.is_set()
        writer.close.assert_called_once()

    asyncio.run(run())


@pytest.mark.parametrize("stop_at_read", [1, 2])
def test_reflex_stop_prevents_in_flight_servo_startup(monkeypatch, stop_at_read):
    bridge = LiveStationBridge("localhost", 1, follow_motor_rx=True)
    bridge.target_bus = "test-bus"
    bridge._motor_state[bridge.target_bus] = {1: MotorState(1, 2000, 0, 1000, 3000)}
    bridge._last_rx_mono = time.monotonic()
    bridge.rx_received_at[(bridge.target_bus, 1)] = bridge._last_rx_mono
    writes = []
    state = bridge.get_state
    reads = 0

    async def collect(_client, commands):
        writes.extend(commands)

    async def simulate_read():
        nonlocal reads
        reads += 1
        if reads == stop_at_read:
            await bridge.abort(source="reflex")
        return await state()

    monkeypatch.setattr(bridge, "_sync_write", lambda bus, reg, values: reg)
    monkeypatch.setattr(bridge, "_send_commands", collect)
    monkeypatch.setattr(bridge, "get_state", simulate_read)

    async def simulate():
        with pytest.raises(EStopLatched):
            await bridge.send_joint_targets({1: 2020})
        assert bridge.abort_source == "reflex"
        assert writes == [RAM_GOAL_POSITION]
        assert not bridge._torque_started

    asyncio.run(simulate())
