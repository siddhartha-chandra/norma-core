import asyncio
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from fastmcp import Client
from PIL import Image

from desk_core import mcp_server
from desk_core.station_bridge import MockStationBridge


class MarkerTwin(MockStationBridge):
    def __init__(self, calib):
        super().__init__(num_joints=8)
        self._motors[8].position = self._motors[8].range_max
        self.calib = calib
        self.writes = 0
        self.mat_contacts = 0
        self.minimum_jaw_z = math.inf
        self.pen_hits = set()

    def build(self, state, markers, zone):
        planner = mcp_server._mujoco_planner(state, [], 0.6)
        spec = planner.spec.copy()
        spec.option.timestep = 0.002
        for geom in spec.geoms:
            geom.margin = 0
            geom.conaffinity = 0
            geom.contype = 2 if planner.link_groups[planner.model.body(geom.parent.name).id] else 0
        table = spec.geom("table")
        table.contype, table.conaffinity = 4, 3
        table.group = 1
        table.rgba = [0.21, 0.24, 0.28, 1]
        xmin, xmax, ymin, ymax = zone
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, name="pen zone",
                                size=[(xmax - xmin) / 2, (ymax - ymin) / 2, 0.0005],
                                pos=[(xmin + xmax) / 2, (ymin + ymax) / 2, self.calib.z_table()],
                                rgba=[0.12, 0.40, 0.26, 1], contype=16, conaffinity=0, group=1)
        for name, x, y, azimuth, color in markers:
            body = spec.worldbody.add_body(name=name, pos=[x, y, self.calib.z_table() + 0.007],
                                           euler=[90, 0, azimuth])
            body.add_freejoint()
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.007, 0.026, 0],
                          mass=0.016, friction=[0.6, 0.004, 0.001], rgba=[*color, 1],
                          contype=1, conaffinity=7, condim=6, group=1)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = planner.qadr
        self.dadr = self.model.jnt_dofadr[:len(self.qadr)]
        self.data.qpos[self.qadr] = self.calib.ticks_to_rad(state.motors)
        self.table_id = self.model.geom("table").id
        self.arm_geoms = set(np.flatnonzero(self.model.geom_contype == 2))
        self.marker_ids = {self.model.geom(name).id: name for name, *_ in markers}
        self.jaws = {}
        for geom_id in self.arm_geoms:
            if self.model.geom(geom_id).name.startswith("Gripper_Jaw_"):
                mesh_id = self.model.geom_dataid[geom_id]
                first, count = self.model.mesh_vertadr[mesh_id], self.model.mesh_vertnum[mesh_id]
                self.jaws[geom_id] = self.model.mesh_vert[first:first + count]
        mujoco.mj_forward(self.model, self.data)

    def step(self, q, qdot):
        self.data.qpos[self.qadr], self.data.qvel[self.dadr] = q, qdot
        mujoco.mj_step(self.model, self.data)
        for contact in self.data.contact:
            pair = set(contact.geom)
            if pair & self.arm_geoms:
                self.mat_contacts += self.table_id in pair
                self.pen_hits.update(self.marker_ids[g] for g in pair if g in self.marker_ids)
        for geom_id, vertices in self.jaws.items():
            z = vertices @ self.data.geom_xmat[geom_id].reshape(3, 3)[2] + self.data.geom_xpos[geom_id, 2]
            self.minimum_jaw_z = min(self.minimum_jaw_z, z.min())

    async def send_joint_targets(self, targets, max_step):
        before = np.array(self.calib.ticks_to_rad(list(self._motors.values())))
        outcome = await super().send_joint_targets(targets, max_step)
        goal = np.array(self.calib.ticks_to_rad(list(self._motors.values())))
        count = max(1, math.ceil(np.max(np.abs(goal - before)) / 0.0009))
        for q in np.linspace(before, goal, count + 1)[1:]:
            self.step(q, (goal - before) / (count * self.model.opt.timestep))
        self.writes += 1
        return outcome

    def settle(self):
        q = np.array(self.calib.ticks_to_rad(list(self._motors.values())))
        for _ in range(300):
            self.step(q, np.zeros_like(q))
        mujoco.mj_forward(self.model, self.data)

    def position(self, name):
        return self.data.body(name).xpos.copy()

    def footprint(self, name):
        geom = self.model.geom(name)
        axis = self.data.geom_xmat[geom.id].reshape(3, 3)[:2, 2]
        radius, half_length = geom.size[:2]
        extent = half_length * np.abs(axis) + radius * np.sqrt(1 - axis ** 2)
        center = self.position(name)[:2]
        return center - extent, center + extent

    def screenshot(self, name):
        directory = os.environ.get("DESK_SWEEP_ARTIFACT_DIR")
        if directory:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.04, 0.245, 0]
            camera.distance, camera.azimuth, camera.elevation = 0.37, 90, -90
            scene_option = mujoco.MjvOption()
            scene_option.geomgroup[0] = 0
            with mujoco.Renderer(self.model, height=480, width=640) as renderer:
                renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
                Image.fromarray(renderer.render()).save(Path(directory) / f"{name}.png")


@pytest.mark.parametrize(("start", "end", "z_mm"), [((0, 0), (0, 0), 6), ((0, 0), (0, 1), 0),
                                                    ((0, 0), (0, 1), math.nan), ((0, math.inf), (0, 1), 6)])
def test_sweep_aside_rejects_invalid_geometry_without_connecting(start, end, z_mm, monkeypatch):
    async def no_bridge():
        pytest.fail("Invalid geometry must be rejected before connecting")

    monkeypatch.setattr(mcp_server, "_get_bridge", no_bridge)
    result = asyncio.run(mcp_server.sweep_aside(start, end, z_mm, execute=True))
    assert not result["ok"] and not result["executed"]


def test_sweep_aside_previews_and_rejects_a_blocked_route_before_writes(monkeypatch):
    bridge = MockStationBridge(num_joints=8)
    monkeypatch.setattr(mcp_server, "_bridge", bridge)
    monkeypatch.setattr(mcp_server, "_obstacles", {})

    async def no_writes(*args, **kwargs):
        pytest.fail("Preview or blocked sweep must not send a motor target")

    monkeypatch.setattr(bridge, "send_joint_targets", no_writes)

    async def run():
        async with Client(mcp_server.mcp) as client:
            args = {"start_xy": [-0.064, 0.205], "end_xy": [0.063, 0.205], "z_mm": 6}
            preview = await client.call_tool("sweep_aside", args)
            assert preview.data["ok"] and not preview.data["executed"], preview.data
            await mcp_server.update_scene_obstacle("obstacle", 0.018, 0.205, 0.014, shape="box",
                                                   size_xyz=[0.016, 0.12, 0.022])
            blocked = await client.call_tool("sweep_aside", {**args, "execute": True})
            assert not blocked.data["ok"] and not blocked.data["executed"], blocked.data
            assert "Collision" in blocked.data["problems"][0], blocked.data

    asyncio.run(run())


def test_sweep_aside_moves_two_rolling_markers_into_side_zone(calib, monkeypatch):
    markers = [
        ("blue", -0.008, 0.205, 13, (0.1, 0.35, 0.95)),
        ("orange", 0.013, 0.273, -11, (0.95, 0.32, 0.08)),
    ]
    zone = (0.043, 0.155, 0.162, 0.345)
    twin = MarkerTwin(calib)
    monkeypatch.setattr(mcp_server, "_bridge", twin)
    monkeypatch.setattr(mcp_server, "_obstacles", {})

    async def run():
        initial = await twin.get_state()
        twin.build(initial, markers, zone)
        twin.settle()
        before = {name: twin.position(name) for name, *_ in markers}
        twin.screenshot("sweep-before")
        expected_writes = 0
        async with Client(mcp_server.mcp) as client:
            for start, end, y in ((-0.064, 0.063, 0.205), (-0.045, 0.073, 0.273)):
                moved = await client.call_tool("sweep_aside", {"start_xy": [start, y], "end_xy": [end, y],
                                                               "z_mm": 6, "execute": True})
                assert moved.data["ok"] and moved.data["executed"], moved.data
                assert moved.data["peak_shoulder_torque_nm"] <= 0.6
                expected_writes += moved.data["waypoints"] - 1
                print(f"Sweep: {moved.data['waypoints']} waypoints, peak {moved.data['peak_shoulder_torque_nm']:.3f} N·m")
                twin.settle()
        twin.screenshot("sweep-after")
        xmin, xmax, ymin, ymax = zone
        for name, previous in before.items():
            x, y, z = twin.position(name)
            print(f"{name}: ({previous[0]:.3f}, {previous[1]:.3f}) -> ({x:.3f}, {y:.3f}) m")
            low, high = twin.footprint(name)
            assert np.all(low >= [xmin, ymin]) and np.all(high <= [xmax, ymax]), (name, low, high)
            assert x - previous[0] > 0.04 and z >= calib.z_table() + 0.006
        clearance = twin.minimum_jaw_z - calib.z_table()
        print(f"Minimum jaw/mat clearance: {clearance * 1000:.2f} mm; arm/mat contacts: {twin.mat_contacts}")
        assert twin.pen_hits == set(before)
        assert twin.writes == expected_writes
        assert twin.mat_contacts == 0 and clearance > 0.002
        assert (await twin.get_state()).positions()[8] == initial.positions()[8]

    asyncio.run(run())
