"""desk_core.motion: timing, feed-forward, planning rules, streaming, lag abort and end-pose correction."""
import asyncio
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import FRONT_POSE_TICKS, SaggingBridge, StuckBridge, motors_at, rad
from desk_core import motion
from desk_core.motion import (GRAVITY_FF_MAX_RAD, MotionLimits, allowed_bounds_rad, correct_final_pose, flange_error,
                              gravity_offset_rad, gravity_torque, lowest_point_m, min_jerk, path_directions,
                              plan_flange_move, stream_joint_path)

FAST = MotionLimits(rate_hz=50.0, min_duration_s=0.2)


@pytest.fixture
def no_ff(monkeypatch):
    monkeypatch.setattr(motion, "gravity_gains", lambda: (np.zeros(7), np.zeros(7)))


def line_path(q0, delta, n=20):
    return np.array([q0 + s * np.asarray(delta) for s in np.linspace(0, 1, n)])


# --- timing ---------------------------------------------------------------------------------------------------

def test_min_jerk_profile():
    assert min_jerk(0.0) == 0.0 and min_jerk(1.0) == 1.0
    assert min_jerk(0.5) == pytest.approx(0.5)
    assert min_jerk(-1.0) == 0.0 and min_jerk(2.0) == 1.0
    tau = np.linspace(0, 1, 1001)
    s = np.array([min_jerk(x) for x in tau])
    assert np.all(np.diff(s) >= 0)
    assert np.max(np.diff(s) / np.diff(tau)) == pytest.approx(1.875, rel=1e-3)


def test_agent_bounds_accept_defaults_and_tested_fast_pace():
    assert MotionLimits().out_of_bounds() == []
    fast = MotionLimits(speed_cm_s=6.0, joint_speed_deg_s=90.0, min_duration_s=1.0, lag_limit_ticks=120)
    assert fast.out_of_bounds() == []


@pytest.mark.parametrize("field, value", [("speed_cm_s", 20.0), ("speed_cm_s", 0.0), ("joint_speed_deg_s", 500.0),
                                          ("min_duration_s", 0.1), ("lag_limit_ticks", 4000), ("floor_cm", 0.0),
                                          ("rot_speed_deg_s", math.nan)])
def test_agent_bounds_reject_out_of_range(field, value):
    problems = replace(MotionLimits(), **{field: value}).out_of_bounds()
    assert len(problems) == 1 and problems[0].startswith(f"{field}=")


def test_sample_interpolates_between_waypoints():
    path = np.array([[0.0] * 7, [1.0] * 7, [3.0] * 7])
    np.testing.assert_allclose(motion._sample(path, 0.0), 0.0)
    np.testing.assert_allclose(motion._sample(path, 0.25), 0.5)
    np.testing.assert_allclose(motion._sample(path, 0.75), 2.0)
    np.testing.assert_allclose(motion._sample(path, 1.0), 3.0)


# --- friction directions and feed-forward ---------------------------------------------------------------------

def test_path_directions_straight_move():
    path = line_path(np.zeros(7), [0.2, -0.2, 0.0, 0.0005, 0, 0, 0], n=50)
    d = path_directions(path)
    assert d.shape == path.shape
    assert np.all(d[:, 0] > 0.99) and np.all(d[:, 1] < -0.99)
    assert np.all(d[:, 2] == 0.0)
    assert np.all(np.abs(d[:, 3]) < 0.2)


def test_path_directions_follow_reversals_smoothly():
    t = np.linspace(0, 2 * np.pi, 400)
    path = np.zeros((400, 7))
    path[:, 0] = 0.3 * np.sin(t)
    d = path_directions(path)[:, 0]
    assert d[50] > 0.9 and d[200] < -0.9 and d[350] > 0.9
    assert np.all(np.abs(d) <= 1.0)
    assert np.abs(np.diff(d)).max() < 0.5


def test_path_directions_single_waypoint():
    np.testing.assert_array_equal(path_directions(np.zeros((1, 7))), np.zeros((1, 7)))


def test_gravity_torque_shape_and_pose_dependence(calib):
    q_front = rad(calib, FRONT_POSE_TICKS)
    g = gravity_torque(q_front)
    assert g.shape == (7,)
    assert abs(g[0]) < 1e-3
    assert g[1] > 0.05
    stretched = q_front.copy()
    stretched[1] -= 0.3
    assert not np.allclose(gravity_torque(stretched), g)


def test_gravity_offset_zero_without_gains(calib, no_ff):
    np.testing.assert_array_equal(gravity_offset_rad(rad(calib, FRONT_POSE_TICKS)), np.zeros(7))


def test_gravity_offset_combines_gravity_and_friction(calib, monkeypatch):
    gains, friction = np.full(7, 0.05), np.full(7, 0.01)
    monkeypatch.setattr(motion, "gravity_gains", lambda: (gains, friction))
    q = rad(calib, FRONT_POSE_TICKS)
    np.testing.assert_allclose(gravity_offset_rad(q), gains * gravity_torque(q))
    direction = np.array([1, -1, 0, 1, -1, 0, 1], dtype=float)
    np.testing.assert_allclose(gravity_offset_rad(q, direction=direction),
                               gains * gravity_torque(q) + friction * direction)


def test_gravity_offset_is_bounded(calib, monkeypatch):
    monkeypatch.setattr(motion, "gravity_gains", lambda: (np.full(7, 100.0), np.zeros(7)))
    offset = gravity_offset_rad(rad(calib, FRONT_POSE_TICKS))
    assert np.abs(offset).max() == pytest.approx(GRAVITY_FF_MAX_RAD)


def test_gravity_gains_file_is_loaded():
    motion.gravity_gains.cache_clear()
    gains, friction = motion.gravity_gains()
    assert gains.shape == friction.shape == (7,)
    assert gains[1:4].min() > 0


def test_gravity_gains_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(motion, "GRAVITY_FF_PATH", tmp_path / "missing.json")
    motion.gravity_gains.cache_clear()
    try:
        gains, friction = motion.gravity_gains()
        assert not gains.any() and not friction.any()
    finally:
        motion.gravity_gains.cache_clear()


# --- limits and geometry --------------------------------------------------------------------------------------

def test_allowed_bounds_shrink_by_margin_but_include_current_pose(calib):
    ticks = list(FRONT_POSE_TICKS)
    motors = motors_at(ticks)
    ticks[2] = motors[2].range_min + 10
    motors = motors_at(ticks)
    lower, upper = allowed_bounds_rad(calib, motors)
    q_now = np.array(calib.ticks_to_rad(motors))
    assert np.all(q_now >= lower - 1e-9) and np.all(q_now <= upper + 1e-9)
    full = np.array(calib.bounds_rad(motors[:7]))
    inner = [j for j in range(7) if j != 2]
    assert np.all(lower[inner] > full[inner, 0]) and np.all(upper[inner] < full[inner, 1])
    assert min(abs(lower[2] - q_now[2]), abs(upper[2] - q_now[2])) < 1e-9


def test_allowed_bounds_ignore_uncalibrated_full_span_ranges(calib):
    calibrated = motors_at(FRONT_POSE_TICKS)
    full_span = [replace(m, range_min=0, range_max=4095) for m in calibrated]
    for a, b in zip(allowed_bounds_rad(calib, calibrated), allowed_bounds_rad(calib, full_span)):
        assert np.allclose(a, b)


def test_lowest_point_at_front_pose(calib):
    assert 0.10 < lowest_point_m(calib, rad(calib, FRONT_POSE_TICKS)) < 0.20


def test_flange_error_zero_at_goal(calib):
    path = line_path(rad(calib, FRONT_POSE_TICKS), [0, 0.05, 0, 0, 0, 0, 0])
    err = flange_error(calib, path, path[-1])
    assert err["off_target_mm"] == 0.0 and err["rotation_off_target_deg"] == 0.0
    assert np.linalg.norm(err["flange_moved_cm"]) > 0.5


# --- planning -------------------------------------------------------------------------------------------------

def test_plan_flange_move_small_lift(calib, front_motors):
    plan = plan_flange_move(calib, front_motors, [0.0, 0.0, 0.01], Rotation.identity())
    assert plan["ok"], plan["problems"]
    s = plan["summary"]
    assert s["flange_target_cm"][2] - s["flange_start_cm"][2] == pytest.approx(1.0, abs=0.01)
    assert s["start_ticks"] == FRONT_POSE_TICKS[:7]
    assert len(s["goal_ticks"]) == len(s["delta_ticks"]) == 7
    assert plan["duration_s"] >= MotionLimits().min_duration_s
    assert plan["path"].shape[1] == 7


def test_plan_duration_follows_cartesian_speed(calib, front_motors):
    limits = MotionLimits(speed_cm_s=1.0, joint_speed_deg_s=500.0, min_duration_s=0.1)
    plan = plan_flange_move(calib, front_motors, [0.0, -0.03, 0.0], Rotation.identity(), limits=limits)
    assert plan["ok"], plan["problems"]
    assert plan["duration_s"] == pytest.approx(3.0, rel=1e-6)


def test_plan_duration_follows_rotation_speed(calib, front_motors):
    limits = MotionLimits(rot_speed_deg_s=10.0, joint_speed_deg_s=500.0, min_duration_s=0.1)
    plan = plan_flange_move(calib, front_motors, np.zeros(3), Rotation.from_euler("y", 20, degrees=True),
                            limits=limits)
    assert plan["ok"], plan["problems"]
    assert plan["duration_s"] == pytest.approx(2.0, rel=1e-3)


def test_plan_refuses_unreachable_move(calib, front_motors):
    plan = plan_flange_move(calib, front_motors, [0.0, -0.5, 0.0], Rotation.identity())
    assert not plan["ok"]
    assert any(p.startswith("waypoint") for p in plan["problems"])


def test_floor_rule_allows_moving_up_from_below_the_floor(calib, front_motors, monkeypatch):
    solver = motion.get_solver(str(calib.urdf_path()))
    monkeypatch.setattr(motion, "lowest_point_m", lambda c, q: float(solver.fk(q)[2, 3]))
    limits = MotionLimits(floor_cm=100.0)
    up = plan_flange_move(calib, front_motors, [0.0, 0.0, 0.01], Rotation.identity(), limits=limits)
    assert up["ok"], up["problems"]
    down = plan_flange_move(calib, front_motors, [0.0, 0.0, -0.01], Rotation.identity(), limits=limits)
    assert not down["ok"]
    assert any("table" in p for p in down["problems"])


# --- streaming ------------------------------------------------------------------------------------------------

def test_stream_reaches_goal_and_records_trace(calib, mock_bridge, no_ff):
    q0 = rad(calib, FRONT_POSE_TICKS)
    path = line_path(q0, [0.05, 0.05, -0.05, 0.05, 0, 0, 0.05])
    trace = []
    run = asyncio.run(stream_joint_path(mock_bridge, calib, path, 0.3, FAST, trace=trace))
    assert run["ok"]
    goal = calib.rad_to_ticks(list(path[-1]), motors_at(FRONT_POSE_TICKS))
    assert run["final_ticks"] == [goal[mid] for mid in range(1, 8)]
    assert len(trace) == run["cycles"] > 0
    t, measured, expected = trace[-1]
    assert len(measured) == len(expected) == 7
    assert all(a <= b for a, b in zip([r[0] for r in trace], [r[0] for r in trace][1:]))


def test_stream_adds_feed_forward_to_commands(calib, mock_bridge, monkeypatch):
    monkeypatch.setattr(motion, "gravity_gains", lambda: (np.full(7, 0.05), np.zeros(7)))
    q0 = rad(calib, FRONT_POSE_TICKS)
    path = line_path(q0, np.zeros(7), n=5)
    run = asyncio.run(stream_joint_path(mock_bridge, calib, path, 0.2, FAST))
    ff_goal = calib.rad_to_ticks(list(path[-1] + gravity_offset_rad(path[-1])), motors_at(FRONT_POSE_TICKS))
    assert run["final_ticks"][1] == ff_goal[2] != FRONT_POSE_TICKS[1]


def test_stream_aborts_on_tracking_lag(calib, no_ff):
    bridge = StuckBridge()
    q0 = rad(calib, FRONT_POSE_TICKS)
    path = line_path(q0, [0.5, 0, 0, 0, 0, 0, 0])
    run = asyncio.run(stream_joint_path(bridge, calib, path, 0.3, FAST))
    assert not run["ok"]
    assert "lag" in run["aborted"]
    assert bridge.aborts == 1


def test_stream_aborts_and_reraises_on_error(calib, mock_bridge, no_ff):
    aborts = []

    async def boom(*args, **kwargs):
        raise RuntimeError("link lost")

    async def abort():
        aborts.append(1)
        return {"aborted": True}

    mock_bridge.send_joint_targets = boom
    mock_bridge.abort = abort
    path = line_path(rad(calib, FRONT_POSE_TICKS), np.zeros(7), n=3)
    with pytest.raises(RuntimeError, match="link lost"):
        asyncio.run(stream_joint_path(mock_bridge, calib, path, 0.2, FAST))
    assert aborts == [1]


# --- end-pose correction --------------------------------------------------------------------------------------

def test_correction_cancels_servo_sag(calib, no_ff, no_sleep):
    bridge = SaggingBridge()
    q0 = rad(calib, FRONT_POSE_TICKS)
    path = line_path(q0, [0, 0.05, -0.05, 0.05, 0, 0, 0])
    assert asyncio.run(stream_joint_path(bridge, calib, path, 0.3, FAST))["ok"]
    before = flange_error(calib, path, calib.ticks_to_rad(asyncio.run(bridge.get_state()).motors))
    assert before["off_target_mm"] > 5
    result = asyncio.run(correct_final_pose(bridge, calib, path, FAST))
    assert result["off_target_mm"][0] == pytest.approx(before["off_target_mm"], abs=0.1)
    assert result["off_target_mm"][-1] <= FAST.correct_tol_mm
    assert result["offset_ticks"][1:4] == [30, -25, -20]


def test_correction_releases_offset_when_stalled(calib, no_ff, no_sleep):
    bridge = SaggingBridge(sag={2: -400})
    path = line_path(rad(calib, FRONT_POSE_TICKS), np.zeros(7), n=3)
    goal = calib.rad_to_ticks(list(path[-1]), motors_at(FRONT_POSE_TICKS))
    for _ in range(3):
        asyncio.run(bridge.send_joint_targets(goal))
    result = asyncio.run(correct_final_pose(bridge, calib, path, FAST))
    assert result["released"]
    assert result["iterations"] < FAST.correct_iterations
    assert result["off_target_mm"][1] < result["off_target_mm"][0]
    assert result["off_target_mm"][-1] > FAST.correct_tol_mm
    assert result["offset_ticks"] == [0] * 7
    assert bridge._motors[2].position == goal[2] - 400


def test_correction_offset_is_bounded(calib, no_ff, no_sleep):
    bridge = SaggingBridge(sag={2: -400})
    path = line_path(rad(calib, FRONT_POSE_TICKS), np.zeros(7), n=3)
    asyncio.run(bridge.send_joint_targets(calib.rad_to_ticks(list(path[-1]), motors_at(FRONT_POSE_TICKS))))
    limits = replace(FAST, correct_iterations=1)
    result = asyncio.run(correct_final_pose(bridge, calib, path, limits))
    assert not result["released"]
    assert abs(result["offset_ticks"][1]) == limits.correct_max_offset_ticks


def test_correction_skips_when_already_on_target(calib, mock_bridge, no_ff, no_sleep):
    path = line_path(rad(calib, FRONT_POSE_TICKS), np.zeros(7), n=3)
    result = asyncio.run(correct_final_pose(mock_bridge, calib, path, FAST))
    assert result["iterations"] == 0
    assert result["offset_ticks"] == [0] * 7
    assert math.isclose(result["off_target_mm"][0], 0.0, abs_tol=0.5)
