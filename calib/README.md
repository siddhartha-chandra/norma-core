# Rig Calibration (`calib/rig_calib.json`)

Shared calibration contract linking Station hardware, camera extrinsics/intrinsics, Pink IK, and MCP tools.

- Edit **`calib/rig_calib.json`** with real measurements (never overwrite `calib/rig_calib.example.json`, which stays as the clean reference schema).

## Conventions
- **Units:** meters, radians, ST3215 ticks (`0..4095`).
- **World frame:** ElRobot `base_link` (tabletop plane height `z_table` in `base_link` meters).
- **Camera frames:** OpenCV optical frame (`+X` right, `+Y` down, `+Z` forward out of lens).
- **Transforms (`T_A_B`):** `4x4` row-major homogeneous matrix mapping `p_A = T_A_B @ p_B`:
  - `T_world_cam_top`: fixed overhead camera → `base_link`.
  - `T_ee_cam_wrist`: the gripper-mounted eye-in-hand camera (the `wrist` view id; there is no separate wrist camera) → `wrist_mount_link` (`Gripper_Base_v1_1`). At runtime, `T_world_cam_wrist(q) = T_world_ee(q) @ T_ee_cam_wrist` via URDF FK (`elrobot_follower.urdf`).
  - `dist_top` / `dist_wrist`: OpenCV 5-element `[k1, k2, p1, p2, k3]` distortion coefficients (automatically applied via `cv2.undistortPoints`).
  - `H_table`: optional `3x3` 4-corner planar homography mapping `top` pixel `[u, v, 1]` to tabletop `[X, Y, 1]` at `z = z_table`.
- **Joint mapping:** `angle_rad = joint_signs[i] * (tick - joint_zero_ticks[i]) / (4096 / 2pi)`, aligned with the URDF zero pose (`hardware/elrobot/simulation/elrobot_follower.urdf`).
- **Hardware IDs (`hardware.*`):** `bus_serial`, `camera_top`, `camera_wrist` auto-populate if omitted from `.env`.

## Regenerate (do not hand-edit calibrated values)
`rig_calib.json` is generated from result files under `calib_frames/`. Run from the repo root:
```bash
# 1. Wrist mount + board pose (fixed-board session; ranges from the station motor state).
#    hand_eye_combined/samples.json lists the hand_eye_1 views plus the board_check poses.
python3 scripts/fit_joint_offsets.py calib_frames/hand_eye_combined/samples.json \
  --calibration calib_frames/session1/camera_calibration.json \
  --ranges calib_frames/hand_eye_1/follower_ranges.json --rig-zeros \
  --output calib_frames/hand_eye_combined/fit_report.json \
  --mount-output calib_frames/hand_eye_combined/wrist_mount_final.json
# 2. Scene camera -> base_link (board must not have moved since step 1)
python3 scripts/capture_scene_board.py calib_frames/scene_1   # needs a running station
python3 scripts/solve_scene_to_base.py calib_frames/scene_1/capture.json \
  --calibration calib_frames/session1/camera_calibration.json \
  --wrist-mount calib_frames/hand_eye_combined/wrist_mount_final.json \
  --output calib_frames/scene_1/scene_to_base.json
# Optional: one run per arm pose; a consistent mean offset means the board moved
python3 scripts/check_board_moved.py calib_frames/board_check \
  --calibration calib_frames/session1/camera_calibration.json \
  --wrist-mount calib_frames/hand_eye_combined/wrist_mount_final.json
# 3. Write rig_calib.json, then verify
python3 scripts/export_rig_calib.py
python3 scripts/verify_calib.py
```
Redo the calibration when any of these change: station motor ranges or servo offsets (re-running auto-calibration or "reset calibration"), the wrist camera mount, the scene camera position, or the stream resolution (intrinsics are for 1280x720).

## Verify & Hot Reload
```bash
# Sanity-check schema, SO(3) extrinsics, H_table agreement, joint_zero_ticks & wrist FK (offline):
uv run --offline --managed-python python scripts/verify_calib.py

# Sanity-check against the live Station + project base/gripper onto overhead image:
uv run --offline --managed-python python scripts/verify_calib.py --live --save-overlay
```
After updating `calib/rig_calib.json`, call MCP tool `reload_config()` (or `get_calib_status()`) without restarting the MCP server.

For the current capture procedure, dataset inventory, offline hand-eye solve and validation status, see the [DESK-CORE hackathon handoff](../docs/desk-core-handoff.md).
