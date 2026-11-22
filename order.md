# Pick-and-place function order

`master_pick_place(query=...)` calls `MainControl.master_pick_place(...)` through `_get_main_control()`. The table follows the controller's call order and matches the step comments in the source. [MCP entry point](desk_core/mcp_server.py#L1115) · [Controller](desk_core/main_control.py#L470)

| Step | Trace stage | Functions and handoffs |
| --- | --- | --- |
| 1 | `observe` | `MainControl.observe()` → MCP `observe(views=["top", "wrist"])`. Check both cameras and follower telemetry. |
| 2 | `detect_requested_object` | `detect_requested_object(query, provider)` → `localize_3d(camera="top")`. Pass `target_base_m` to wrist aiming and capture. Return early if the object is absent. |
| 2 | `validate_setup` | `_tool_parameters()` and `_motor_state()` validate the measured tool and follower for live execution. `update_scene_obstacle()` registers any supplied tabletop obstacles before motion. |
| 3 | `aim_wrist` | `aim_wrist(target_base_m)` uses the current wrist view or plans a move that brings the target into view. The default dry run ends here. |
| 4 | `capture_wrist_stereo` | `capture_wrist_stereo(target_base_m)` captures a settled frame, makes one lateral move, and captures a second settled frame. Pass the returned `capture_id` downstream. |
| 4 | `estimate_wrist_depth` | `_confirm_wrist_target(capture_id, query, provider)` finds the object in frame A. Pass its ROI and the `capture_id` to `estimate_wrist_depth()`, which triangulates matches, segments the target, and estimates the top surface. |
| 5 | `measure_graspability` | `measure_graspability(capture_id, jaw_opening_mm)` measures the footprint and checks the jaw gap across and along the object. The master requires the across-object fit. |
| 6 | `estimate_grasp_pose` | `estimate_grasp_pose(capture_id, tool)` returns the object yaw and `ik_target` in `base_link` as XYZ and an `xyzw` quaternion. The controller derives pickup hover, pickup contact, placement hover, and release poses using the saved X. |
| 6 | `solve_ik` | `solve_ik()` solves pickup hover → pickup contact → placement hover → release. The controller passes the resulting joint angles to the route checks. |
| 6 | `check_motion_paths` | `_home_goal()` reads saved home targets when enabled. `_validate_goal()` checks joint limits and `_checked_path()` checks the routes against the table and registered obstacles before pickup. |
| 7 | `pick` | `_pick()` calls `release()` to open the jaws, then `_move_checked()` to reach pickup hover and contact. It calls `grasp()`, verifies contact, lifts to hover, and calls `_require_held()`. |
| 8 | `place` | `_place()` calls `_move_checked()` to reach placement hover and release height, checking the grip at both poses. It calls `release()` and retracts to placement hover. |
| 9 | `home` | With `return_home=True`, `_move_checked()` returns the arm to its saved joint ticks and `_home_gripper()` sets the saved gripper ticks. |
| Optional | `cloud` | With `build_cloud=True`, `_cloud(capture_id)` writes a dense PLY and attempts an Open3D preview after the motion stages. |

`_move_checked()` rechecks each leg from fresh motor telemetry and calls `stream_joint_path()`. If a stage fails after live motion is enabled, the controller attempts `estop()` followed by `observe()` and returns `failed_step` and the timing trace. [Motion streaming](desk_core/motion.py) · [Saved X](calib/drop_location.json) · [Saved home](calib/fixed_positions.json)

Run the observation and aim dry run from the repo root.

```bash
uv run --offline --managed-python python -m desk_core.main_control "red block"
```

Live execution takes `execute=True` in MCP or `--execute` in the CLI. It requires measured grasp tool calibration and registration of any tabletop obstacles. [Station setup and commands](pose_estimation.md#run-the-station-workflow)
