"""Run the wrist MCP tool on a MuJoCo-rendered marker; use MUJOCO_GL=egl when headless."""
import asyncio
import math
from pathlib import Path
import sys
import time

import cv2
import mujoco
import numpy as np
from fastmcp import Client
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk_core import mcp_server
from desk_core.pose_estimation import pixels_on_plane
from desk_core.station_bridge import MockStationBridge
from desk_core.urdf_fk import load_urdf
from desk_core.wrist_ibvs import PERIOD_S


WIDTH, HEIGHT = 640, 480
FOVY = 50
FOCAL = HEIGHT / (2 * math.tan(math.radians(FOVY / 2)))
K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]])
GL_FROM_OPTICAL = np.diag([1, -1, -1])
START_TICKS = [2108, 1933, 2046, 2155, 1868, 2962, 2139, 2864]


class RenderedWrist(MockStationBridge):
    def __init__(self, calib):
        super().__init__(num_joints=8)
        for motor, tick in zip(self._motors.values(), START_TICKS):
            motor.position = tick
        self.calib = calib
        self.renderer = None
        self.sample_times = []

    async def get_frame(self, camera="top", denoise=False):
        assert camera == "wrist"
        self.sample_times.append(time.monotonic())
        state = await self.get_state()
        q = self.calib.ticks_to_rad(state.motors)
        xyz, rotation = self.arm.flange_pose(q)
        flange = np.eye(4)
        flange[:3, :3], flange[:3, 3] = rotation.as_matrix(), xyz
        eye = flange @ self.calib.data["T_ee_cam_wrist"]
        self.scene.cam_pos[0] = eye[:3, 3]
        self.scene.cam_quat[0] = Rotation.from_matrix(eye[:3, :3] @ GL_FROM_OPTICAL).as_quat(scalar_first=True)
        mujoco.mj_forward(self.scene, self.data)
        self.renderer.update_scene(self.data, camera="wrist")
        rgb = self.renderer.render()
        return cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


async def main():
    calib = mcp_server.CalibrationStore()
    calib.data.update(K_wrist=K.tolist(), image_size_wrist=[WIDTH, HEIGHT], dist_wrist=[0.] * 5)
    bridge = RenderedWrist(calib)
    mcp_server._calib, mcp_server._bridge = calib, bridge
    initial = await bridge.get_state()
    bridge.arm = mcp_server._mujoco_planner(initial, [], 0.6)
    q = calib.ticks_to_rad(initial.motors)
    initial_eye = calib.T_world_cam("wrist", q)
    z = calib.z_table()
    target = pixels_on_plane([K[:2, 2]], K, None, initial_eye, z)[0] + [0.014, -0.008, 0]
    yaw = 52
    bridge.scene = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <worldbody>
        <light directional="true" dir="0 0 -1"/>
        <geom type="plane" size="1 1 .01" pos="0 0 {z}" rgba=".75 .75 .75 1"/>
        <geom type="box" size=".018 .006 .0002" pos="{target[0]} {target[1]} {z}"
              euler="0 0 {yaw}" rgba=".9 .06 .09 1"/>
        <camera name="wrist" fovy="{FOVY}"/>
      </worldbody>
    </mujoco>
    """)
    bridge.data = mujoco.MjData(bridge.scene)
    with mujoco.Renderer(bridge.scene, HEIGHT, WIDTH) as bridge.renderer:
        await bridge.get_frame("wrist")
        bridge.sample_times.clear()
        async with Client(mcp_server.mcp) as client:
            call_start = time.monotonic()
            result = (await client.call_tool("servo_align_wrist", {})).data
            wall_elapsed = time.monotonic() - call_start
    q = calib.ticks_to_rad((await bridge.get_state()).motors)
    final_eye = calib.T_world_cam("wrist", q)
    optical = pixels_on_plane([K[:2, 2]], K, None, final_eye, z)[0]
    _, final_flange = bridge.arm.flange_pose(q)
    jaw = load_urdf(calib.urdf_path()).joints_by_child["Gripper_Jaw_01_v1_1"]
    jaw_x, jaw_y = final_flange.apply(jaw.origin_T[:3, :3] @ jaw.axis)[:2]
    yaw_error = (yaw + 90 - math.degrees(math.atan2(jaw_y, jaw_x)) + 90) % 180 - 90
    error_mm = 1000 * np.linalg.norm(optical - target)
    periods = np.diff(bridge.sample_times)
    print(f"MCP result: {result}")
    print(f"Final MuJoCo ground truth: position_error_mm={error_mm:.3f} jaw_yaw_error_deg={yaw_error:.3f}")
    print(f"Nominal sim frame time: {len(periods) * PERIOD_S:.3f}s; MCP wall time: {wall_elapsed:.3f}s")
    print(f"Frame intervals (s): {np.round(periods, 3).tolist()}")
    assert result["ok"] and error_mm < 1 and abs(yaw_error) < 2 and wall_elapsed < 1.5


if __name__ == "__main__":
    asyncio.run(main())
