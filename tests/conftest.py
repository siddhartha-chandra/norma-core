"""Shared fixtures for the desk_core motion and IK tests (offline: mock bridge, no Station)."""
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from desk_core.calib_store import CalibrationStore  # noqa: E402
from desk_core.station_bridge import ELROBOT_MOTOR_RANGES, MockStationBridge, MotorState  # noqa: E402

# A reachable pose in front of the base with every joint well inside its range (hardware-tested).
FRONT_POSE_TICKS = [2049, 2233, 1063, 1098, 1857, 1300, 1399, 2048]


@pytest.fixture(scope="session")
def calib() -> CalibrationStore:
    return CalibrationStore()


def motors_at(ticks) -> list[MotorState]:
    return [MotorState(mid, int(t), 0, *ELROBOT_MOTOR_RANGES[mid]) for mid, t in enumerate(ticks, start=1)]


@pytest.fixture
def front_motors() -> list[MotorState]:
    return motors_at(FRONT_POSE_TICKS)


@pytest.fixture
def mock_bridge() -> MockStationBridge:
    bridge = MockStationBridge(num_joints=8)
    for m, t in zip(bridge._motors.values(), FRONT_POSE_TICKS):
        m.position = t
    return bridge


class StuckBridge(MockStationBridge):
    """Accepts targets but never moves, like a jammed or unpowered arm."""

    def __init__(self):
        super().__init__(num_joints=8)
        for m, t in zip(self._motors.values(), FRONT_POSE_TICKS):
            m.position = t
        self.aborts = 0

    async def send_joint_targets(self, targets, max_step=150):
        return {mid: self._motors[mid].position for mid in targets}

    async def abort(self):
        self.aborts += 1
        return await super().abort()


class SaggingBridge(MockStationBridge):
    """Settles a fixed number of ticks short of each target on the gravity-loaded joints, like the ST3215s."""

    def __init__(self, sag=None):
        super().__init__(num_joints=8)
        for m, t in zip(self._motors.values(), FRONT_POSE_TICKS):
            m.position = t
        self.sag = sag or {2: -30, 3: 25, 4: 20}

    async def send_joint_targets(self, targets, max_step=150):
        shifted = {mid: int(t) + self.sag.get(int(mid), 0) for mid, t in targets.items()}
        return await super().send_joint_targets(shifted, max_step=max_step)


@pytest.fixture
def no_sleep(monkeypatch):
    """Skip the real-time waits in desk_core.motion (settling pauses between correction iterations)."""
    import asyncio

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setattr("desk_core.motion.asyncio.sleep", fast_sleep)


def rad(calib, ticks):
    return np.array(calib.ticks_to_rad(motors_at(ticks)))
