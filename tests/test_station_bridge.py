"""Races between live motor startup and the reflex stop, with fake Station telemetry and writes."""
import asyncio
import time

import pytest

from desk_core.station_bridge import EStopLatched, LiveStationBridge, MotorState, RAM_GOAL_POSITION


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
