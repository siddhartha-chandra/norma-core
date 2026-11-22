"""Offline checks for the motion scripts: gravity fit, infinity tracing/plotting and the move_flange MCP client."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import FRONT_POSE_TICKS, REPO, motors_at

SCRIPTS = REPO / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fit_mod():
    return load_script("fit_gravity_ff")


@pytest.fixture(scope="module")
def trace_mod():
    return load_script("trace_infinity")


@pytest.fixture(scope="module")
def plot_mod():
    return load_script("plot_trace")


# --- fit_gravity_ff -------------------------------------------------------------------------------------------

def synthetic_samples(gains, friction, n=40, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        goal = rng.uniform(-1, 1, 7)
        torque = rng.uniform(0.05, 0.6, 7)
        direction = rng.choice([-1.0, 1.0], 7)
        reached = goal - (gains * torque + friction * direction)
        samples.append({"goal": goal.tolist(), "reached": reached.tolist(), "torque": torque.tolist(),
                        "direction": direction.tolist()})
    return samples


def test_fit_recovers_gains_and_friction(fit_mod, monkeypatch, tmp_path):
    out = tmp_path / "gravity_ff.json"
    monkeypatch.setattr(fit_mod, "GRAVITY_FF_PATH", out)
    gains = np.array([0.0, 0.04, 0.07, 0.08, 0.0, 0.07, 0.0])
    friction = np.radians([0.4, 0.8, 1.5, 1.6, 0.5, 0.5, 0.5])
    fit_mod.fit(synthetic_samples(gains, friction))
    saved = json.loads(out.read_text())
    np.testing.assert_allclose(np.array(saved["gain_rad_per_Nm"])[[1, 2, 3, 5]], gains[[1, 2, 3, 5]], atol=1e-9)
    np.testing.assert_allclose(saved["friction_rad"], friction, atol=1e-9)
    assert len(saved["samples"]) == 40


def test_saved_samples_migrates_legacy_file(fit_mod, monkeypatch, tmp_path):
    legacy = tmp_path / "gravity_ff.json"
    goals = [[0.1] * 7, [0.3] * 7, [0.2] * 7]
    samples = [{"goal": g, "reached": g, "torque": [0.1] * 7} for g in goals]
    legacy.write_text(json.dumps({"gain_rad_per_Nm": [0] * 7, "samples": samples}))
    monkeypatch.setattr(fit_mod, "GRAVITY_FF_PATH", legacy)
    migrated = fit_mod.saved_samples()
    assert len(migrated) == 2
    assert migrated[0]["direction"] == [1.0] * 7
    assert migrated[1]["direction"] == [-1.0] * 7


def test_saved_samples_without_file(fit_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(fit_mod, "GRAVITY_FF_PATH", tmp_path / "none.json")
    assert fit_mod.saved_samples() == []


def test_fit_poses_are_safe_and_spread(fit_mod, calib):
    from desk_core.motion import allowed_bounds_rad, lowest_point_m

    motors = motors_at(FRONT_POSE_TICKS)
    poses = fit_mod.plan_poses(calib, motors, FRONT_POSE_TICKS[:7])
    lower, upper = allowed_bounds_rad(calib, motors)
    assert len(poses) == fit_mod.N_POSES
    for y_cm, z_cm, q in poses:
        assert np.all(q >= lower) and np.all(q <= upper)
        assert lowest_point_m(calib, q) >= 0.10
        assert y_cm < -5
    torques = [fit_mod.gravity_torque(q)[1] for _, _, q in poses]
    assert max(torques) - min(torques) > 0.1


def test_committed_gains_are_plausible():
    saved = json.loads((REPO / "calib" / "gravity_ff.json").read_text())
    gains, friction = np.array(saved["gain_rad_per_Nm"]), np.array(saved["friction_rad"])
    assert np.all((gains[[1, 2, 3, 5]] > 0.01) & (gains[[1, 2, 3, 5]] < 0.2))
    assert np.all((friction >= 0) & (friction < np.radians(3)))


# --- trace_infinity / plot_trace ------------------------------------------------------------------------------

def test_infinity_offsets_shape(trace_mod):
    offsets, length = trace_mod.infinity_offsets(0.10, 0.05, loops=2)
    np.testing.assert_allclose(offsets[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(offsets[-1], 0.0, atol=1e-6)
    assert offsets[:, 0].max() == pytest.approx(0.05, abs=1e-3)
    assert offsets[:, 2].max() == pytest.approx(0.025, abs=1e-3)
    assert np.all(offsets[:, 1] == 0.0)
    steps = np.linalg.norm(np.diff(offsets, axis=0), axis=1)
    assert steps.max() <= trace_mod.STEP_M * 1.01
    assert length == pytest.approx(steps.sum(), rel=1e-3)


def test_plan_tip_path_follows_curve(trace_mod, calib):
    from desk_core.ik_tracik import get_solver

    offsets, _ = trace_mod.infinity_offsets(0.04, 0.02, loops=1)
    path, problem = trace_mod.plan_tip_path(calib, motors_at(FRONT_POSE_TICKS), offsets)
    assert path is not None, problem
    solver = get_solver(str(calib.urdf_path()))
    tips = np.array([trace_mod.tip_position(solver, q) for q in path])
    np.testing.assert_allclose(tips - tips[0], offsets, atol=1e-3)


def test_plan_tip_path_refuses_unreachable_curve(trace_mod, calib):
    offsets, _ = trace_mod.infinity_offsets(1.0, 0.5, loops=1)
    path, problem = trace_mod.plan_tip_path(calib, motors_at(FRONT_POSE_TICKS), offsets)
    assert path is None and problem.startswith("waypoint")


def test_trace_error_metrics(plot_mod):
    planned = np.c_[np.linspace(0, 0.1, 11), np.zeros(11), np.zeros(11)]
    measured = planned + [0, 0, 0.002]
    np.testing.assert_allclose(plot_mod.shape_error_mm(planned, measured), 2.0, atol=1e-9)
    np.testing.assert_allclose(plot_mod.lag_error_mm(measured, planned), 2.0, atol=1e-9)


def test_plot_writes_image(plot_mod, tmp_path):
    trace_file = tmp_path / "trace.npz"
    t = np.linspace(0, 5, 50)
    planned = np.c_[0.05 * np.sin(t), np.zeros(50), 0.025 * np.sin(2 * t)]
    np.savez(trace_file, planned=planned, t=t, measured=planned + 0.001, expected=planned,
             width_cm=10.0, height_cm=5.0, loops=1)
    trace = plot_mod.load(trace_file)
    out = tmp_path / "trace.png"
    plot_mod.plot(trace, out, before=trace)
    assert out.stat().st_size > 10_000


def test_saved_trace_meets_accuracy():
    """The committed hardware trace (docs/motion) stays a valid record of the achieved accuracy."""
    plot_mod = load_script("plot_trace")
    trace = plot_mod.load(REPO / "docs" / "motion" / "infinity_trace.npz")
    assert np.median(trace["shape_mm"]) < 6.0
    assert trace["shape_mm"].max() < 15.0


# --- move_flange MCP client (mock bridge, real stdio MCP server) ----------------------------------------------

def run_move_flange(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(SCRIPTS / "move_flange.py"), "--mock", *args],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_move_flange_dry_run_over_mcp():
    proc = run_move_flange("--dy", "1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dry run OK" in proc.stdout
    assert "execution:" not in proc.stdout


def test_move_flange_execute_over_mcp():
    proc = run_move_flange("--dz", "1", "--execute", "--yes")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result_line = next(line for line in proc.stdout.splitlines() if line.strip().startswith("result:"))
    result = json.loads(result_line.split("result:", 1)[1])
    assert result["off_target_mm"] < 2.0


def test_move_flange_refusal_exit_code():
    proc = run_move_flange("--dy", "-60")
    assert proc.returncode == 1
    assert "REFUSED" in proc.stdout


def test_move_flange_rejects_unbounded_pace_over_mcp():
    proc = run_move_flange("--dz", "1", "--speed-cm-s", "30", "--lag-limit-ticks", "4000")
    assert proc.returncode == 1
    assert "speed_cm_s=30.0 outside" in proc.stdout and "lag_limit_ticks=4000 outside" in proc.stdout


def test_move_flange_refuses_while_another_runs():
    from desk_core import mcp_server

    async def overlapping():
        async with mcp_server._flange_lock:
            return await mcp_server.move_flange(dz_cm=1.0)

    result = asyncio.run(overlapping())
    assert not result["ok"] and result["problems"] == ["another move_flange is still running"]
