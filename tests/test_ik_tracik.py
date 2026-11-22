"""TRAC-IK solver: FK/IK round trips, joint-limit handling, diagnostics and Cartesian planning."""
import math
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import FRONT_POSE_TICKS, rad
from desk_core import ik_tracik
from desk_core.ik_tracik import MAX_POSITION_ERROR_M, MAX_ROTATION_ERROR_RAD, TracIKSolver, get_solver


@pytest.fixture(scope="module")
def solver() -> TracIKSolver:
    return get_solver()


@pytest.fixture
def q_front(calib):
    return rad(calib, FRONT_POSE_TICKS)


def test_constructor_keeps_stdout_clean(capfd):
    """KDL's URDF warning must not reach fd 1, which carries the MCP stdio protocol."""
    TracIKSolver()
    out, err = capfd.readouterr()
    assert out == ""


def test_chain_has_seven_joints_with_urdf_limits(solver):
    assert solver.dof == 7
    lower, upper = solver.urdf_limits
    assert lower.shape == upper.shape == (7,)
    assert np.all(lower < upper)


def test_get_solver_is_cached():
    assert get_solver() is get_solver()


@pytest.mark.parametrize("seed", range(5))
def test_fk_ik_round_trip(solver, q_front, seed):
    rng = np.random.default_rng(seed)
    q_target = q_front + rng.uniform(-0.15, 0.15, 7)
    T = solver.fk(q_target)
    sol = solver.solve(T[:3, 3], T[:3, :3], q_front)
    assert sol.ok, sol.reason
    assert sol.position_error_m <= MAX_POSITION_ERROR_M
    assert sol.rotation_error_rad <= MAX_ROTATION_ERROR_RAD
    np.testing.assert_allclose(solver.fk(sol.q), T, atol=1e-3)


def test_solution_respects_custom_limits(solver, q_front):
    T = solver.fk(q_front + 0.05)
    lower, upper = q_front - 0.3, q_front + 0.3
    sol = solver.solve(T[:3, 3], T[:3, :3], q_front, lower, upper)
    assert sol.ok, sol.reason
    assert np.all(sol.q >= lower - 1e-9) and np.all(sol.q <= upper + 1e-9)


def test_blocked_by_limits_names_the_joint(solver, q_front):
    q_target = q_front.copy()
    q_target[2] += 0.4
    T = solver.fk(q_target)
    # Pin every joint near the start so the 7-DoF redundancy cannot route around the joint 3 limit.
    lower, upper = q_front - 0.05, q_front + 0.05
    sol = solver.solve(T[:3, 3], T[:3, :3], q_front, lower, upper)
    assert not sol.ok
    assert sol.reason.startswith("blocked by limits")
    assert "joint 3" in sol.reason


def test_unreachable_pose_is_reported(solver, q_front):
    sol = solver.solve([2.0, 0.0, 0.0], np.eye(3), q_front)
    assert not sol.ok
    assert "unreachable" in sol.reason


def test_seed_outside_limits_is_clipped(solver, q_front):
    T = solver.fk(q_front)
    sol = solver.solve(T[:3, 3], T[:3, :3], q_front + 10.0)
    assert sol.ok, sol.reason


def test_plan_cartesian_straight_line(solver, q_front):
    shift = np.array([0.0, -0.02, 0.01])
    plan = solver.plan_cartesian(q_front, shift)
    assert plan.ok, plan.problems
    assert plan.summary["waypoints"] == math.ceil(np.linalg.norm(shift) / 0.002)
    assert len(plan.path) == plan.summary["waypoints"] + 1
    T0, T1 = solver.fk(plan.path[0]), solver.fk(plan.path[-1])
    np.testing.assert_allclose(T1[:3, 3], T0[:3, 3] + shift, atol=MAX_POSITION_ERROR_M)
    np.testing.assert_allclose(T1[:3, :3], T0[:3, :3], atol=1e-2)
    flange_track = np.array([solver.fk(q)[:3, 3] for q in plan.path])
    deviation = np.linalg.norm(np.cross(flange_track - T0[:3, 3], shift / np.linalg.norm(shift)), axis=1)
    assert deviation.max() < 1e-3
    assert np.abs(np.diff(plan.path, axis=0)).max() < 0.15


@pytest.mark.parametrize("rot_frame", ["flange", "base"])
def test_plan_cartesian_rotation_frames(solver, q_front, rot_frame):
    rotation = Rotation.from_euler("z", 10, degrees=True)
    plan = solver.plan_cartesian(q_front, np.zeros(3), rotation, rot_frame)
    assert plan.ok, plan.problems
    R0 = Rotation.from_matrix(solver.fk(plan.path[0])[:3, :3])
    R1 = Rotation.from_matrix(solver.fk(plan.path[-1])[:3, :3])
    expected = R0 * rotation if rot_frame == "flange" else rotation * R0
    assert (expected.inv() * R1).magnitude() < math.radians(0.5)
    assert plan.summary["rotation_deg"] == pytest.approx(10.0, abs=0.01)


def test_plan_cartesian_stops_at_first_problem(solver, q_front):
    plan = solver.plan_cartesian(q_front, np.array([0.0, -1.0, 0.0]))
    assert not plan.ok
    assert plan.problems[0].startswith("waypoint ")
    assert 1 <= len(plan.path) < plan.summary["waypoints"] + 1


def test_solve_ik_plugin_returns_seven_floats(solver, q_front):
    T = solver.fk(q_front)
    qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
    q = ik_tracik.solve_ik(*T[:3, 3], qx, qy, qz, qw, seed_joints_rad=list(q_front))
    assert len(q) == 7 and all(isinstance(v, float) for v in q)
    np.testing.assert_allclose(q, q_front, atol=1e-2)


def test_solve_ik_plugin_honours_bounds_and_raises(solver, q_front):
    T = solver.fk(q_front)
    qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
    tight = [(v + 0.5, v + 0.6) for v in q_front]
    with pytest.raises(ValueError, match="TRAC-IK"):
        ik_tracik.solve_ik(*T[:3, 3], qx, qy, qz, qw, seed_joints_rad=list(q_front), joint_bounds_rad=tight)


def test_default_urdf_exists():
    assert os.path.exists(ik_tracik.DEFAULT_URDF)
