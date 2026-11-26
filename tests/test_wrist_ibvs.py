import asyncio
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from desk_core import mcp_server
from desk_core.pose_estimation import pixels_on_plane
from desk_core.station_bridge import LiveStationBridge, MockStationBridge, RobotState
from desk_core.stereo_sweep import _blob_mask
from desk_core.wrist_ibvs import alignment_step


K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
FLANGE = np.eye(4)
FLANGE[:3, :3] = Rotation.from_euler("x", -90, degrees=True).as_matrix()
FLANGE[:3, 3] = [0, 0, 0.15]
MOUNT = np.eye(4)
MOUNT[:3, :3] = Rotation.from_euler("x", -90, degrees=True).as_matrix()
MOUNT[:3, 3] = [0.004, 0, 0.01]


def frame_of_marker(camera, center, yaw_deg, plane_z=0.0):
    yaw = math.radians(yaw_deg)
    major = np.array([math.cos(yaw), math.sin(yaw)]) * 0.018
    minor = np.array([-math.sin(yaw), math.cos(yaw)]) * 0.005
    corners = np.array([center + a * major + b * minor for a, b in ((-1, -1), (-1, 1), (1, 1), (1, -1))])
    points = (np.c_[corners, np.full(4, plane_z)] - camera[:3, 3]) @ camera[:3, :3]
    image_points = points @ K.T
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.fillConvexPoly(frame, np.rint(image_points[:, :2] / image_points[:, 2:]).astype(np.int32), (0, 30, 220))
    return frame


def test_center_and_perpendicular_jaw_yaw_converge_with_camera_offset():
    flange = FLANGE.copy()
    target = np.array([0.013, -0.007])
    first = None
    for _ in range(10):
        camera = flange @ MOUNT
        frame = frame_of_marker(camera, target, 53)
        step = alignment_step(frame, K, None, camera, flange, 0.0)
        if first is None:
            first = step
        flange[:3, 3] = step.position
        flange[:3, :3] = step.rotation.as_matrix()
        if step.error_mm < 0.5 and abs(math.degrees(step.yaw_error_rad)) < 2:
            break
    assert first.error_mm == pytest.approx(math.hypot(9, 17), abs=0.3)
    assert math.degrees(first.yaw_error_rad) == pytest.approx(-37, abs=1)
    assert step.error_mm < 0.5
    assert abs(math.degrees(step.yaw_error_rad)) < 2
    np.testing.assert_allclose(flange[:3, 1], FLANGE[:3, 1], atol=1e-8)
    assert flange[2, 3] == FLANGE[2, 3]


def test_plane_area_centroid_is_unbiased_with_a_tilted_camera():
    camera = FLANGE @ MOUNT
    camera[:3, :3] = Rotation.from_euler("x", 45, degrees=True).as_matrix() @ camera[:3, :3]
    camera[:3, 3] = [0, -0.09, 0.09]
    frame = frame_of_marker(camera, np.zeros(2), 90)
    pixels = cv2.findNonZero(_blob_mask(frame, "color")).reshape(-1, 2)
    image_weighted_centroid = pixels_on_plane(pixels, K, None, camera, 0.0).mean(axis=0)

    assert np.linalg.norm(image_weighted_centroid) > 0.001
    step = alignment_step(frame, K, None, camera, FLANGE, 0.0)
    assert step.error_mm < 0.5
    assert abs(math.degrees(step.yaw_error_rad)) < 0.5


@pytest.mark.parametrize("color, message", [
    ((200, 200, 200), "No complete colored marker"),
    ((0, 30, 220), "Marker yaw is ambiguous"),
])
def test_empty_and_ambiguous_frames_are_rejected(color, message):
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 25, color, cv2.FILLED)
    with pytest.raises(ValueError, match=message):
        alignment_step(frame, K, None, FLANGE @ MOUNT, FLANGE, 0.0)


def test_invalid_tolerance_is_reported_before_connecting(monkeypatch):
    async def unexpected_bridge():
        pytest.fail("Invalid tolerance must not connect to the Station.")

    monkeypatch.setattr(mcp_server, "_get_bridge", unexpected_bridge)
    assert not asyncio.run(mcp_server.servo_align_wrist(tol_mm=float("nan")))["ok"]


def test_unavailable_camera_times_out_without_joint_commands(monkeypatch, calib):
    class SlowCamera(MockStationBridge):
        async def get_frame(self, camera="top", denoise=False):
            await asyncio.Future()

        async def send_joint_targets(self, targets, max_step):
            pytest.fail("Alignment must not move without a frame.")

    monkeypatch.setattr(mcp_server, "_bridge", SlowCamera(num_joints=8))
    monkeypatch.setattr(mcp_server, "_calib", calib)
    monkeypatch.setattr(mcp_server, "_mujoco_planner", lambda *args: None)
    monkeypatch.setattr(mcp_server, "TIMEOUT_S", 0.01)
    result = asyncio.run(mcp_server.servo_align_wrist())
    assert not result["ok"]
    assert result["error"] == "Wrist alignment timed out."
    assert result["iterations"] == 0


@pytest.mark.parametrize("settled_ns, captures", [(200, [100, 199, 201]), (0, [0, 0, 0])])
def test_live_feedback_skips_frames_from_before_settle(monkeypatch, calib, settled_ns, captures):
    monkeypatch.setitem(calib.data, "shoulder_torque_limit_nm", 0.6)
    monkeypatch.setitem(calib.data, "image_size_wrist", [640, 480])
    monkeypatch.setattr(mcp_server, "_calib", calib)
    monkeypatch.setattr(mcp_server, "_logger", None)
    wrist_id = calib.hardware_config()["camera_wrist"]
    bridge = LiveStationBridge(cam_wrist=wrist_id, follow_motor_rx=True)
    monkeypatch.setattr(mcp_server, "_bridge", bridge)
    monkeypatch.setattr(bridge, "_resolve_cam_key", lambda _: wrist_id)
    mock = MockStationBridge(num_joints=8)
    monkeypatch.setattr(bridge, "get_state", mock.get_state)
    monkeypatch.setattr(mcp_server, "_mujoco_planner", lambda *args: None)
    monkeypatch.setattr(mcp_server, "_live_preflight_state", lambda _, state: state)
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(mcp_server, "time", SimpleNamespace(monotonic=lambda: clock.now))
    period = 0.01
    monkeypatch.setattr(mcp_server, "PERIOD_S", period)
    frames = iter(zip([0, period / 2, period], captures, [0, 100, 200]))

    async def get_frame(_):
        elapsed, captured, pixel = next(frames)
        clock.now += elapsed
        bridge.frame_received_at[wrist_id] = clock.now
        bridge.frame_station_ns[wrist_id] = captured
        return cv2.imencode(".jpg", np.full((480, 640, 3), pixel, np.uint8))[1].tobytes()

    async def move(*args, **kwargs):
        return {"ok": True, "executed": True, "settled_station_ns": settled_ns}

    accepted = []

    def align(frame, *args):
        accepted.append(frame[0, 0, 0])
        return SimpleNamespace(error_mm=2 if len(accepted) == 1 else 0, yaw_error_rad=0,
                               position=np.zeros(3), rotation=Rotation.identity())

    monkeypatch.setattr(bridge, "get_frame", get_frame)
    monkeypatch.setattr(mcp_server, "_plan_and_move", move)
    monkeypatch.setattr(mcp_server, "alignment_step", align)

    result = asyncio.run(mcp_server.servo_align_wrist())
    assert result["ok"]
    assert result["iterations"] == 2
    assert accepted == [0, 200]


@pytest.mark.parametrize("torque_limit", [0.6, 0.001])
def test_live_large_yaw_uses_coarse_preflighted_roll(monkeypatch, calib, torque_limit):
    bus = calib.hardware_config()["bus_serial"]
    wrist_id = calib.hardware_config()["camera_wrist"]
    bridge = LiveStationBridge(target_bus=bus, cam_wrist=wrist_id, follow_motor_rx=True)
    mock = MockStationBridge(num_joints=8)
    motors = list(mock._motors.values())
    for motor, tick in zip(motors, [2108, 1933, 2046, 2155, 1868, 2962, 2139, 2864]):
        motor.position = tick
    bridge._torque_started = set(range(1, 8))
    monkeypatch.setitem(calib.data, "shoulder_torque_limit_nm", torque_limit)
    monkeypatch.setitem(calib.data, "K_wrist", K.tolist())
    monkeypatch.setitem(calib.data, "image_size_wrist", [640, 480])
    monkeypatch.setitem(calib.data, "dist_wrist", [0.] * 5)
    monkeypatch.setattr(mcp_server, "_calib", calib)
    monkeypatch.setattr(mcp_server, "_bridge", bridge)
    monkeypatch.setattr(mcp_server, "_logger", None)

    q = np.array(calib.ticks_to_rad(motors))
    flange = calib.fk_ee_pose(q)
    yaw = math.degrees(math.atan2(flange[1, 0], flange[0, 0])) - 45
    rolled = q.copy()
    rolled[-1] += math.pi / 4
    z = calib.z_table()
    eye = calib.T_world_cam("wrist", rolled)
    target = pixels_on_plane([K[:2, 2]], K, None, eye, z)[0, :2] + [0.0016, 0]
    writes = []

    async def get_state():
        now = time.monotonic()
        bridge._last_rx_mono = now
        bridge.rx_received_at.update({(bus, m.id): now for m in motors})
        bridge.rx_station_ns.update({(bus, m.id): time.monotonic_ns() for m in motors})
        return RobotState(bus, motors)

    async def get_frame(_camera):
        eye = calib.T_world_cam("wrist", calib.ticks_to_rad(motors))
        frame = frame_of_marker(eye, target, yaw, z)
        bridge.frame_station_ns[wrist_id] = time.monotonic_ns()
        bridge.frame_received_at[wrist_id] = time.monotonic()
        return cv2.imencode(".jpg", frame)[1].tobytes()

    def sync_write(_bus, _address, values):
        return {mid: int.from_bytes(raw, "little") for mid, raw in values.items()}

    async def send_commands(_client, commands):
        for targets in commands:
            changed = {m.id: targets[m.id] - m.position for m in motors
                       if m.id in targets and targets[m.id] != m.position}
            writes.append(changed)
            for mid, tick in targets.items():
                motors[mid - 1].position = tick

    monkeypatch.setattr(bridge, "get_state", get_state)
    monkeypatch.setattr(bridge, "get_frame", get_frame)
    monkeypatch.setattr(bridge, "_resolve_cam_key", lambda _: wrist_id)
    monkeypatch.setattr(bridge, "_sync_write", sync_write)
    monkeypatch.setattr(bridge, "_send_commands", send_commands)

    result = asyncio.run(mcp_server.servo_align_wrist())
    if torque_limit < 0.6:
        assert not result["ok"] and "Shoulder torque" in result["error"]
        assert not writes
        return
    assert result["ok"], result
    assert result["error_mm"] < 1 and abs(result["yaw_error_deg"]) < 2
    assert 0.16 <= result["prerotation_s"] and result["ibvs_s"] < 1.5
    assert all(set(step) == {7} and 200 < abs(step[7]) < 400 for step in writes[:2])
    assert len(writes) > 2
