# Rig Calibration (`calib/rig_calib.json`)

Shared calibration contract linking Station hardware, camera extrinsics/intrinsics, Pink IK, and MCP tools.

- Edit **`calib/rig_calib.json`** with real measurements (never overwrite `calib/rig_calib.example.json`, which stays as the clean reference schema).

## Conventions
- **Units:** meters, radians, ST3215 ticks (`0..4095`).
- **World frame:** ElRobot `base_link` (tabletop plane height `z_table` in `base_link` meters).
- **Camera frames:** OpenCV optical frame (`+X` right, `+Y` down, `+Z` forward out of lens).
- **Transforms (`T_A_B`):** `4x4` row-major homogeneous matrix mapping `p_A = T_A_B @ p_B`:
  - `T_world_cam_top`: fixed overhead camera → `base_link`.
  - `T_ee_cam_wrist`: follower wrist camera → `wrist_mount_link` (`Gripper_Base_v1_1`). At runtime, `T_world_cam_wrist(q) = T_world_ee(q) @ T_ee_cam_wrist` via URDF FK (`elrobot_follower.urdf`).
  - `dist_top` / `dist_wrist`: OpenCV 5-element `[k1, k2, p1, p2, k3]` distortion coefficients (automatically applied via `cv2.undistortPoints`).
  - `H_table`: optional `3x3` 4-corner planar homography mapping `top` pixel `[u, v, 1]` to tabletop `[X, Y, 1]` at `z = z_table`.
- **Joint mapping:** `angle_rad = joint_signs[i] * (tick - joint_zero_ticks[i]) / (4096 / 2pi)`, aligned with the URDF zero pose (`hardware/elrobot/simulation/elrobot_follower.urdf`).
- **Hardware IDs (`hardware.*`):** `bus_serial`, `camera_top`, `camera_wrist` auto-populate if omitted from `.env`.

## Verify & Hot Reload
```bash
# Sanity-check schema, SO(3) extrinsics, H_table agreement, joint_zero_ticks & wrist FK (offline):
uv run --offline --managed-python python scripts/verify_calib.py

# Sanity-check against the live Station + project base/gripper onto overhead image:
uv run --offline --managed-python python scripts/verify_calib.py --live --save-overlay
```
After updating `calib/rig_calib.json`, call MCP tool `reload_config()` (or `get_calib_status()`) without restarting the MCP server.
