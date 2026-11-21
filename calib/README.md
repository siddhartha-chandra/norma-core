# Rig Calibration (`calib/rig_calib.json`)

Shared calibration contract for **Abhimanyu & Scot** (`#19`, `#4`, `#6`), **Suhail** (`#9` / Pink IK), and **Sid** (MCP & Gemini 4 Hydrogen).

## Conventions
- **Units:** meters, radians, ST3215 ticks (`0..4095`).
- **World frame:** ElRobot `base_link` (tabletop plane height `z_table` in `base_link` meters).
- **Camera frames:** OpenCV optical frame (`+X` right, `+Y` down, `+Z` forward out of lens).
- **Transforms (`T_A_B`):** `4x4` row-major homogeneous matrix mapping `p_A = T_A_B @ p_B`:
  - `T_world_cam_top`: fixed overhead camera -> `base_link`.
  - `T_ee_cam_wrist`: follower wrist camera -> `wrist_mount_link` (`Gripper_Base_v1_1`). At runtime, `T_world_cam_wrist(q) = T_world_ee(q) @ T_ee_cam_wrist`.
  - `H_table`: optional `3x3` 4-corner planar homography mapping `top` pixel `[u, v, 1]` to tabletop `[X, Y, 1]` at `z = z_table`.
- **Joint mapping:** `angle_rad = joint_signs[i] * (tick - joint_zero_ticks[i]) / (4096 / 2pi)`, aligned with Suhail's URDF zero pose (`hardware/elrobot/simulation/elrobot_follower.urdf`).

## Hot Reload
After updating `calib/rig_calib.json`, call MCP tool `reload_config()` (or `get_calib_status()` to inspect loaded state) without restarting the MCP server.
