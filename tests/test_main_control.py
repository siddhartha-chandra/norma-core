"""Offline checks for the numbered pick-and-place handoffs and stop conditions."""
import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from desk_core.main_control import MainControl


TOOL = {
    "tool_quaternion_xyzw": [-2**-0.5, 0, 0, 2**-0.5],
    "tcp_offset_m": [0, 0.103, 0],
    "reference_jaw_yaw_deg": 0,
    "jaw_opening_mm": 54,
}


@pytest.fixture
def workflow(calib, mock_bridge, monkeypatch):
    async def run_mock_path(fn, *args):
        return fn(*args)

    monkeypatch.setattr("desk_core.main_control.asyncio.to_thread", run_mock_path)
    tools = {name: AsyncMock(return_value={"ok": True}) for name in ("estop", "update_scene_obstacle", "release")}
    tools["grasp"] = AsyncMock(return_value={"ok": True, "holding": True})
    tools["solve_ik"] = AsyncMock(side_effect=[[i * 0.01] * 7 for i in range(4)])
    control = MainControl(mock_bridge, calib, tools, {})
    monkeypatch.setattr(control, "observe", AsyncMock())
    monkeypatch.setattr(control, "detect_requested_object", AsyncMock(return_value={
        "ok": True, "found": True, "target_base_m": [0.2, 0.0, calib.z_table()],
    }))
    monkeypatch.setattr(control, "aim_wrist", AsyncMock(return_value={"ok": True}))
    monkeypatch.setattr(control, "capture_wrist_stereo", AsyncMock(return_value={"ok": True, "capture_id": "pair"}))
    monkeypatch.setattr(control, "_confirm_wrist_target", AsyncMock(return_value=(10, 20, 30, 40)))
    monkeypatch.setattr(control, "estimate_wrist_depth", AsyncMock(return_value={"ok": True}))
    monkeypatch.setattr(control, "measure_graspability", Mock(return_value={
        "fits": {"across": True, "along": False}, "corners_base_m": [[0.2, 0, calib.z_table()]] * 8,
    }))
    monkeypatch.setattr(control, "estimate_grasp_pose", Mock(return_value={
        "ik_target": {"x": 0.2, "y": 0, "z": 0.14, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        "grasp_point_base_m": [0.2, 0, 0.02],
    }))
    monkeypatch.setattr(control, "_validate_goal", Mock())
    monkeypatch.setattr(control, "_checked_path", Mock())
    monkeypatch.setattr(control, "_move_checked", AsyncMock())
    monkeypatch.setattr(control, "_require_held", AsyncMock())
    monkeypatch.setattr(control, "_home_gripper", AsyncMock())
    return control


def test_missing_object_stops_before_motion_or_tool_calibration(workflow):
    workflow.detect_requested_object.return_value = {"ok": False, "found": False}
    result = asyncio.run(workflow.master_pick_place("keys", execute=True))
    assert not result["found"] and not result["executed"]
    workflow.aim_wrist.assert_not_awaited()
    workflow.tools["estop"].assert_not_awaited()


def test_dry_run_passes_scene_target_to_aim_only(workflow):
    result = asyncio.run(workflow.master_pick_place("keys"))
    assert result["ok"] and not result["executed"]
    workflow.aim_wrist.assert_awaited_once_with([0.2, 0.0, workflow.calib.z_table()], False)
    workflow.capture_wrist_stereo.assert_not_awaited()


def test_full_workflow_passes_capture_and_ik_results_to_motion(workflow):
    result = asyncio.run(workflow.master_pick_place("keys", execute=True, tool=TOOL))
    assert result["ok"], result
    assert result["capture_id"] == "pair" and result["home"]
    workflow.estimate_wrist_depth.assert_awaited_once_with("pair", roi=(10, 20, 30, 40))
    assert workflow.tools["solve_ik"].await_count == 4
    assert workflow.tools["solve_ik"].await_args_list[1].kwargs["seed_joints_rad"] == [0.0] * 7
    assert workflow._move_checked.await_args_list[1].args[0] == [0.01] * 7
    assert workflow.tools["release"].await_count == 2
    workflow._home_gripper.assert_awaited_once_with(1256)


def test_grasp_without_contact_stops_before_place(workflow):
    workflow.tools["grasp"].return_value = {"ok": True, "holding": False}
    result = asyncio.run(workflow.master_pick_place("keys", execute=True, tool=TOOL))
    assert not result["ok"] and result["failed_step"] == "pick", result
    workflow.tools["estop"].assert_awaited_once()
    workflow.tools["release"].assert_awaited_once()
    workflow._home_gripper.assert_not_awaited()


def test_lift_allows_initial_object_support_and_requires_final_clearance(monkeypatch):
    class Rig:
        def urdf_path(self):
            return None

        def z_table(self):
            return 0.0

        def fk_ee_pose(self, q):
            transform = np.eye(4)
            transform[2, 3] = q[0]
            return transform

    control = MainControl(None, Rig(), {}, {})
    monkeypatch.setattr("desk_core.main_control._jaw_bounds", lambda _: np.array([[0, 0, 0.02]]))
    monkeypatch.setattr("desk_core.main_control.lowest_point_m", lambda *_: 0.02)
    q0 = np.zeros(7)
    carried = np.zeros((4, 3))
    assert len(control._checked_path(q0, q0 + 0.01, carried, lift_from_table=True)) == 2
    with pytest.raises(ValueError, match="carried object"):
        control._checked_path(q0, q0 + 0.01, carried)
    with pytest.raises(ValueError, match="did not clear the table"):
        control._checked_path(q0, q0 + 0.001, carried, lift_from_table=True)
