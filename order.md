# Pick-and-place function order

`master_pick_place(query=...)` calls `MainControl.master_pick_place(...)` through `_get_main_control()`. The table follows the controller's call order and matches the step comments in the source. [MCP entry point](desk_core/mcp_server.py#L1115) · [Controller](desk_core/main_control.py#L470)

| Step | Trace stage | Functions and handoffs |
| --- | --- | --- |
| 1 | `observe` | `MainControl.observe()` → MCP `observe(views=["top", "wrist"])`. Check both cameras and follower telemetry. |
| 2 | `detect_requested_object` | `detect_requested_object(query, provider)` → `localize_3d(camera="top")`. Pass `target_base_m` to wrist aiming and capture. Return early if the object is absent. |
| 2 | `validate_setup` | `_tool_parameters()` and `_motor_state()` validate the measured tool and follower for live execution. `update_scene_obstacle()` registers any supplied tabletop obstacles before motion. |
| 3 | `aim_wrist` | `aim_wrist(target_base_m)` uses the current wrist view or plans a move that brings the target into view. The default dry run ends here. **Before metric depth estimation, align the object to the middle of the wrist camera frame.** Merely "visible" is not enough: from an awkward wrist pose the object sits at the image edge or behind the jaws, and the stereo step finds no matches or fails the wrist/scene consistency check. |
| 4 | `capture_wrist_stereo` | `capture_wrist_stereo(target_base_m)` captures a settled frame, makes one lateral move, and captures a second settled frame. Pass the returned `capture_id` downstream. |
| 4 | `estimate_wrist_depth` | `_confirm_wrist_target(capture_id, query, provider)` finds the object in frame A. Pass its ROI and the `capture_id` to `estimate_wrist_depth()`, which triangulates matches, segments the target, and estimates the top surface. |
| 5 | `measure_graspability` | `measure_graspability(capture_id, jaw_opening_mm)` measures the footprint and checks the jaw gap across and along the object. The master requires the across-object fit. |
| 6 | `plan_tolerant_grasp` | `plan_tolerant_grasp(capture_id, tool)` ([`desk_core/tolerant_grasp.py`](desk_core/tolerant_grasp.py)) takes the measured 3D centre, planar yaw and box from `measure_graspability()`. Because the wide jaws self-centre the object, bounded least-squares IK lets the jaw tips sit off-centre along the closing axis (slack `(opening - width)/2 - 2 mm`) and the jaw heading deviate (largest angle keeping the projected span inside the opening). The jaws never touch the table: every corner of both jaw collision boxes (URDF meshes, open and closed, carried through flange pose · jaw-in-flange transform) must stay 8 mm above the table at the grasp (5 mm clearance plus 3 mm for gravity sag). The jaw-tip midpoint alone is not enough, because the jaws reach about 27 mm sideways and drop below it when the approach is tilted. It returns pickup hover and contact joint angles (`q_hover`, `q_contact`, vertical hover with a clear path) and the contact flange `ik_target`. Replaces `estimate_grasp_pose()` in the master workflow; that tool remains available. |
| 6 | `check_motion_paths` | `_drop_goal()` reads `joint_ticks` from the saved X and `_home_goal()` reads saved home targets when enabled. `_validate_goal()` checks joint limits and `_checked_path()` checks every route (jaw geometry, and the carried object once lifted) against the table and registered obstacles before pickup. No IK runs for placement. |
| 7 | `pick` | `_pick()` calls `release()` to open the jaws, then `_move_checked()` to reach pickup hover and contact. It calls `grasp()`, which closes across the object's short side and verifies contact, then `retract_to_hover()`. |
| 7 | `retract_to_hover` | `retract_to_hover()` (also an MCP tool) moves the closed gripper straight back to the planned `q_hover` through a checked path and calls `_require_held()`. Always retract before carrying. |
| 8 | `place` | `_place()` calls `move_to_drop()` (also an MCP tool), which replays `joint_ticks` from [`calib/drop_location.json`](calib/drop_location.json) through a checked joint path with the carried object (no IK). It checks the grip, then calls `release()`. |
| 9 | `home` | With `return_home=True`, `_move_checked()` returns the arm to its saved joint ticks and `_home_gripper()` sets the saved gripper ticks. |
| Optional | `cloud` | With `build_cloud=True`, `_cloud(capture_id)` writes a dense PLY and attempts an Open3D preview after the motion stages. |

For step 4, the proven alternative to the single slide is `scripts/wrist_stereo_sweep.py` (fallback slide directions, dense cloud) followed by `scripts/measure_object.py`. Centre the object in the wrist image first (`--object X Y Z` re-centres on the given point, so pass the object's actual position or use `--target U V` on the case pixel).

`_move_checked()` rechecks each leg from fresh motor telemetry and calls `stream_joint_path()`. If a stage fails after live motion is enabled, the controller attempts `estop()` followed by `observe()` and returns `failed_step` and the timing trace. [Motion streaming](desk_core/motion.py) · [Saved X](calib/drop_location.json) · [Saved home](calib/fixed_positions.json)

Run the observation and aim dry run from the repo root.

```bash
uv run --offline --managed-python python -m desk_core.main_control "red block"
```

Live execution takes `execute=True` in MCP or `--execute` in the CLI. It requires measured grasp tool calibration and registration of any tabletop obstacles. [Station setup and commands](pose_estimation.md#run-the-station-workflow)
