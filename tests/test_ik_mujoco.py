import asyncio
from dataclasses import replace
import time

import mujoco
import numpy as np
import pytest

from desk_core import mcp_server
from desk_core.ik_mujoco import MAX_WAYPOINT_STEP_RAD, MuJoCoPlanner, PlanningError
from desk_core.ik_tracik import TOOL_FRAME
from desk_core.interfaces import SceneObstacle
from desk_core.station_bridge import LiveStationBridge, MockStationBridge, RobotState


@pytest.fixture
def scene(calib):
    bridge = MockStationBridge(num_joints=8)
    state = asyncio.run(bridge.get_state())

    def planner(obstacles=(), limit=0.6, table_z=None):
        return MuJoCoPlanner(calib.urdf_path(), calib.z_table() if table_z is None else table_z,
                             list(obstacles), limit, calib.bounds_rad(state.motors))

    return planner, bridge, np.asarray(calib.ticks_to_rad(state.motors))


def test_fk_and_analytical_torque_match_reference_models(calib, scene):
    make_planner, _, start = scene
    planner = make_planner()
    for q in (start, start + [0.2, -0.2, 0.12, -0.12, 0.15, -0.2, 0.3]):
        position, rotation = planner.flange_pose(q)
        expected = calib.fk_ee_pose(q, TOOL_FRAME)
        np.testing.assert_allclose(position, expected[:3, 3], atol=1e-9)
        np.testing.assert_allclose(rotation.as_matrix(), expected[:3, :3], atol=1e-9)
        holding_torque = planner.shoulder_torque(q)
        mujoco.mj_forward(planner.model, planner.data)
        shoulder_dof = planner.model.jnt_dofadr[planner.shoulder]
        assert holding_torque == pytest.approx(planner.data.qfrc_bias[shoulder_dof], abs=1e-9)


@pytest.mark.parametrize("shape", ["cylinder", "box"])
def test_rejects_swept_collision_with_clear_endpoints(scene, shape):
    make_planner, _, q = scene
    midpoint, _ = make_planner().flange_pose(q)
    obstacle = SceneObstacle("sweep", shape, *midpoint, radius_m=0.03, height_m=0.04,
                             size_xyz=(0.06, 0.06, 0.06))
    planner = make_planner([obstacle])
    start, goal = q.copy(), q.copy()
    start[0], goal[0] = -0.4, 0.4
    assert planner.plan_trajectory(start, start).peak_shoulder_torque_nm < 0.6
    assert planner.plan_trajectory(goal, goal).peak_shoulder_torque_nm < 0.6
    with pytest.raises(PlanningError, match=r"Waypoint \d+: Collision clearance violated: .*sweep"):
        planner.plan_trajectory(start, goal)


def test_table_collision_and_torque_limit_are_hard_constraints(scene):
    make_planner, _, q = scene
    with pytest.raises(PlanningError, match=r"Collision clearance violated: .*table"):
        make_planner(table_z=0.22).plan_trajectory(q, q)
    with pytest.raises(PlanningError, match="Shoulder torque"):
        make_planner(limit=0.5).plan_trajectory(q, q)


def test_ik_lift_tucks_elbow_and_wrist_before_shoulder(scene):
    make_planner, _, start = scene
    planner = make_planner()
    target = start.copy()
    target[1] -= 0.2
    position, rotation = planner.flange_pose(target)
    goal = planner.solve_ik(position, rotation.as_quat(), start)
    actual_position, actual_rotation = planner.flange_pose(goal)
    np.testing.assert_allclose(actual_position, position, atol=0.0005)
    assert (actual_rotation * rotation.inv()).magnitude() < np.deg2rad(0.3)
    plan = planner.plan_trajectory(start, goal)
    assert 0 < plan.tuck_end < len(plan.waypoints) - 1
    tuck = plan.waypoints[plan.tuck_end]
    np.testing.assert_array_equal(plan.waypoints[:plan.tuck_end + 1, 1], start[1])
    assert np.all(np.abs(tuck[2:4] - start[2:4]) > 0)
    assert abs(planner.shoulder_torque(tuck)) < abs(planner.shoulder_torque(start))
    assert np.max(np.abs(np.diff(plan.waypoints, axis=0))) <= MAX_WAYPOINT_STEP_RAD
    assert plan.peak_shoulder_torque_nm <= 0.6


def test_plan_and_move_preflights_before_sending_any_mock_targets(scene, monkeypatch):
    make_planner, bridge, start = scene
    monkeypatch.setattr(mcp_server, "_bridge", bridge)
    monkeypatch.setattr(mcp_server, "_obstacles", {})
    send = bridge.send_joint_targets
    sent = []

    async def track_targets(targets, max_step):
        sent.append(targets)
        return await send(targets, max_step)

    monkeypatch.setattr(bridge, "send_joint_targets", track_targets)
    target = start.copy()
    target[1] -= 0.2
    xyz, rotation = make_planner().flange_pose(target)
    args = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), [*xyz, *rotation.as_quat()]))

    async def run():
        initial_ticks = (await bridge.get_state()).positions()
        preview = await mcp_server.plan_and_move(**args, shoulder_torque_limit_nm=0.6)
        assert preview["ok"] and not preview["executed"]
        assert not sent
        solution = await mcp_server.solve_ik(**args, shoulder_torque_limit_nm=0.6)
        assert len(solution) == 7
        await mcp_server.update_scene_obstacle("blocker", *xyz, radius_m=0.03, height_m=0.04)
        blocked = await mcp_server.plan_and_move(**args, shoulder_torque_limit_nm=0.6, execute=True)
        assert not blocked["ok"] and not blocked["executed"]
        assert "Collision" in blocked["problems"][0]
        await mcp_server.update_scene_obstacle("blocker", *xyz, remove=True)
        overloaded = await mcp_server.plan_and_move(**args, shoulder_torque_limit_nm=0.5, execute=True)
        assert not overloaded["ok"] and not overloaded["executed"]
        assert "Shoulder torque" in overloaded["problems"][0]
        assert not sent
        assert (await bridge.get_state()).positions() == initial_ticks
        moved = await mcp_server.plan_and_move(**args, shoulder_torque_limit_nm=0.6, execute=True)
        assert moved["ok"] and moved["executed"]
        assert len(sent) == moved["waypoints"] - 1
        tuck = moved["tuck_end_waypoint"]
        assert tuck and all(command[2] == initial_ticks[2] for command in sent[:tuck])
        assert sent[tuck - 1][3] != initial_ticks[3] and sent[tuck - 1][4] != initial_ticks[4]
        assert sent[tuck][2] != initial_ticks[2]
        assert (await bridge.get_state()).positions() == {**initial_ticks, **sent[-1]}

    asyncio.run(run())


@pytest.fixture
def live_station(scene, calib, monkeypatch):
    _, mock, _ = scene
    motors = [replace(m) for m in asyncio.run(mock.get_state()).motors]
    motors[-1].position = motors[-1].range_max
    bus = calib.data["hardware"]["bus_serial"]
    state = RobotState(bus, motors)
    bridge = LiveStationBridge(target_bus=bus, follow_motor_rx=True)
    bridge._torque_started = {m.id for m in motors[:-1]}
    packets = []

    async def get_state():
        now = time.monotonic()
        bridge._last_rx_mono = now
        bridge.rx_received_at.update({(bus, m.id): now for m in motors})
        return state

    def sync_write(bus_serial, address, values):
        assert bus_serial == bus
        return {mid: int.from_bytes(value, "little") for mid, value in values.items()}

    async def send_commands(client, commands):
        packets.extend(commands)
        for targets in commands:
            for motor in motors:
                if motor.id in targets:
                    motor.position = targets[motor.id]

    monkeypatch.setattr(bridge, "get_state", get_state)
    monkeypatch.setattr(bridge, "_sync_write", sync_write)
    monkeypatch.setattr(bridge, "_send_commands", send_commands)
    monkeypatch.setattr(mcp_server, "_bridge", bridge)
    monkeypatch.setattr(mcp_server, "_calib", calib)
    monkeypatch.setattr(mcp_server, "_obstacles", {})
    return bridge, state, packets


def test_live_plan_uses_rig_alignment_and_sends_nothing_for_rejected_paths(live_station, calib, monkeypatch):
    bridge, state, packets = live_station
    monkeypatch.setitem(calib.data["joint_signs"], "2", -1)
    monkeypatch.setitem(calib.data["joint_zero_ticks"], "2", 2057)
    planner = mcp_server._mujoco_planner(state, [], 0.6)
    start = np.asarray(calib.ticks_to_rad(state.motors))
    target = start.copy()
    target[1] -= 0.2
    xyz, rotation = planner.flange_pose(target)
    args = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), [*xyz, *rotation.as_quat()]))
    send = bridge._send_commands

    async def send_with_shoulder_sag(client, commands):
        previous = state.positions()[2]
        await send(client, commands)
        if state.motors[1].position > previous:
            state.motors[1].position -= 1

    monkeypatch.setattr(bridge, "_send_commands", send_with_shoulder_sag)

    async def run():
        start_ticks = state.positions()
        await mcp_server.update_scene_obstacle("blocker", *xyz, radius_m=0.03, height_m=0.04)
        blocked = await mcp_server.plan_and_move(**args, shoulder_torque_limit_nm=0.6, execute=True)
        assert not blocked["ok"] and "Collision" in blocked["problems"][0]
        await mcp_server.update_scene_obstacle("blocker", *xyz, remove=True)
        overloaded = await mcp_server.plan_and_move(**args, shoulder_torque_limit_nm=0.5, execute=True)
        assert not overloaded["ok"] and "Shoulder torque" in overloaded["problems"][0]
        assert not packets and state.positions() == start_ticks
        moved = await mcp_server.plan_and_move(**args, shoulder_torque_limit_nm=0.6, execute=True)
        assert moved["ok"] and moved["executed"]
        assert len(packets) == moved["waypoints"] - 1
        assert all(8 not in target for target in packets)
        assert state.positions()[2] > start_ticks[2]
        assert state.positions()[2] == packets[-1][2] - 1
        assert calib.ticks_to_rad(state.motors) == moved["measured_joints_rad"]

    asyncio.run(run())


def test_live_bridge_refuses_pose_drift_and_clamping_before_writing(live_station):
    bridge, state, packets = live_station
    snapshot = replace(state, motors=[replace(m) for m in state.motors])

    async def run():
        await bridge.get_state()
        with pytest.raises(RuntimeError, match="exact preflighted waypoint"):
            await bridge.send_joint_targets({1: 2500}, max_step=5, preflight_start=snapshot)
        state.motors[-1].position -= 1
        with pytest.raises(RuntimeError, match="no longer matches"):
            await bridge.send_joint_targets({1: 2050}, preflight_start=snapshot)
        assert not packets

    asyncio.run(run())


def test_live_stall_replaces_active_target_with_present_position(live_station, calib, monkeypatch):
    bridge, state, packets = live_station
    planner = mcp_server._mujoco_planner(state, [], 0.6)
    goal = np.asarray(calib.ticks_to_rad(state.motors))
    goal[1] -= 0.2
    xyz, rotation = planner.flange_pose(goal)
    args = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), [*xyz, *rotation.as_quat()]))

    async def stall(targets, tol_ticks):
        state.motors[2].position -= 1
        return None

    monkeypatch.setattr(bridge, "wait_preflight_settle", stall)
    result = asyncio.run(mcp_server.plan_and_move(**args, shoulder_torque_limit_nm=0.6, execute=True))
    assert not result["ok"] and result["executed"]
    assert result["hold"]["mode"] == "hold"
    assert len(packets) == 2
    assert packets[1][3] == packets[0][3] - 1
    assert packets[1] == state.positions()


def test_live_gripper_models_open_tolerance(live_station, calib):
    bridge, state, _ = live_station
    state.motors[-1].position -= 25

    async def run():
        assert (await mcp_server.release())["ok"]
        snapshot = mcp_server._live_preflight_state(bridge, await bridge.get_state())
        position = np.asarray(calib.ticks_to_rad(snapshot.motors))
        open_model = mcp_server._mujoco_planner(snapshot, [], 0.6)
        measured_model = mcp_server._mujoco_planner(snapshot, [], 0.6, snapshot.motors[-1].to_norm())
        for model in (open_model, measured_model):
            model.flange_pose(position)
        jaw = "Gripper_Jaw_01_v1_1"
        assert not np.allclose(open_model.data.body(jaw).xpos, measured_model.data.body(jaw).xpos)
        state.motors[-1].position -= 1
        with pytest.raises(PlanningError, match="open calibrated gripper"):
            mcp_server._live_preflight_state(bridge, await bridge.get_state())

    asyncio.run(run())
