# Robot setup: run log and feedback

## 2026-11-22: earbuds case, scene-camera localization

Command: `python -m desk_core.main_control "earbuds case"` (dry run, no `--execute`, arm not moved).

| Stage | Result | Time |
|---|---|---|
| observe | ok (top + wrist cameras, Station RX fresh) | 5 ms |
| detect_requested_object | found (Claude Sonnet 5.5, confidence 0.97) | 2.8 s |
| validate_setup | skipped checks (dry run) | 0 ms |
| aim_wrist | FAILED: "No clear wrist viewing pose with room for a slide was found." | 0.26 s |

Localization (`localize_3d`, camera `top`):

- Pixel (1280x720): (750, 507), box_2d `[629, 531, 779, 641]` (normalised 0..1000, y/x order).
- Base frame: x = 0.0718 m, y = -0.1722 m, z = 0.00237 m (table). About 18.6 cm from the base.
- Rough footprint from the top camera: 0.122 x 0.094 m (box, over-estimates the case).
- Known top-camera error: about 2 cm.

Feedback / blockers:

- `aim_wrist` cannot find a wrist viewing pose: the case sits close to the base (inner workspace edge), the same region where earlier grasps needed custom view sampling.
- `calib/rig_calib.json` has no `grasp_tool` entry, so `--execute` will stop at `validate_setup` even if aiming succeeds. Needs `tool_quaternion_xyzw`, `tcp_offset_m`, `reference_jaw_yaw_deg`, `jaw_opening_mm`.
- On failure, `master_pick_place` drops the detection from its result, so coordinates had to be re-read with `localize_3d`.

## 2026-11-22: go home

Used `MainControl._home_goal` + `_move_checked(home, MotionLimits(lag_limit_ticks=15))` + `_home_gripper`
(same calls as stage 9 of `master_pick_place`), home from `calib/fixed_positions.json`.

- Start ticks: 2050, 2742, 930, 943, 1847, 1166, 241, gripper 2835.
- Final ticks: 2137, 2891, 930, 996, 1866, 1042, 139, gripper 1256 (exact match to saved home).
- Result: ok, about 76 s total (slow default limits; arm path plus gripper close).

## 2026-11-22: go home (2nd): did NOT move the arm

Same calls, but the real arm did not move. Cause: `mcp_server._get_bridge()` silently uses the MOCK bridge
when `STATION_HOST` is unset (it is not in `.env`). Both "go home" runs above ran on the mock, so their
"final ticks" were the mock's, not the robot's. Always run with `STATION_HOST=127.0.0.1 STATION_PORT=8888`
and check the bridge type is `LiveStationBridge` before moving.

## 2026-11-22: go home (3rd, real arm)

`STATION_HOST=127.0.0.1 STATION_PORT=8888`, live bridge confirmed, then `mcp_server.move_to_home(max_step=80)`
(direct joint ticks through Station, no IK), then gripper to 1256 via `send_joint_targets`.

- Start ticks: 2689, 2552, 930, 943, 1847, 1166, 241, gripper 2261.
- Final ticks (independent live read): 2139, 2890, 930, 968, 1868, 1053, 148, gripper 1259.
- All within 40 ticks of home (j4 28 off, j6 11 off, j7 9 off). About 4 s for the arm.

## 2026-11-22: pick and place earbuds case to X (main_control --execute)

Tool passed with `--tool /tmp/grasp_tool.json` (rig has no `grasp_tool`): quaternion [-0.7071, 0, 0, 0.7071]
(same as tests/test_main_control.py), tcp_offset_m [0.0018, 0.1030, -0.0007], reference_jaw_yaw_deg 0, jaw 54 mm.

Live detection: case at base (-0.107, -0.225, 0.002) m, top pixel (531, 530), confidence 0.90.

| Run | Start pose | Failed step | Error |
|---|---|---|---|
| 1 | home | capture_wrist_stereo | "No safe sideways stereo slide preserves the target view." (no motion) |
| 2 | FRONT_POSE ticks (2049, 2233, 1063, 1098, 1857, 1300, 1399) | capture_wrist_stereo | "The wrist failed to reach its planned pose." |
| 3 | where run 2 stopped | capture_wrist_stereo | same as run 2 |

Feedback:

- From home, joints 3, 6, 7 sit inside the 50-tick planner margin, so no stereo slide is feasible: start from a more open pose.
- `_execute_wrist_path` (stereo_sweep.py) streams with `gravity_ff=False` and then requires every joint within 25 ticks
  of the goal. On this arm the loaded joints sag past 25 ticks, so the slide always fails after moving.
- On failure `estop` holds position (mode "hold"), it does not cut torque: safe.

Change (approved): `desk_core/stereo_sweep.py` `_execute_wrist_path` final-pose tolerance 25 -> 40 ticks
(stereo uses the captured encoder pose, so a few ticks of sag do not affect depth).

| Run | Change | Failed step | Error |
|---|---|---|---|
| 4 | tolerance 40 | estimate_wrist_depth | "Need at least 4 distinct target matches, got 0." Gripper was closed: jaws hide the lower half of the case in the wrist view, the rest is glossy black with no texture (`calib_frames/mc_run/wrist.jpg`). |
| 5 | gripper opened first (`release`) | estimate_wrist_depth | "Wrist detection disagrees with the scene target projection." Wrist detection centre is outside the 160 px ROI projected from the top-camera target (top camera about 2 cm off). |

Feedback:

- Open the gripper before the stereo capture, otherwise the jaws occlude the target in the wrist camera.
- The ROI check in `_confirm_wrist_target` is too strict for the top camera's ~2 cm error at this distance.

Change (approved): `desk_core/main_control.py` `_confirm_wrist_target` accepts the wrist box centre up to 80 px
outside the projected ROI.

| Run | Failed step | Error |
|---|---|---|
| 6 | detect_requested_object | "Scene detection fell back to OpenCV." Transient vision API failure, no motion; next call worked. |
| 7 | estimate_wrist_depth | "Couldn't confirm earbuds case in the reference wrist frame." Case at the bottom of the wrist image, half hidden by the open jaw (`calib_frames/mc_run/wrist_run7.jpg`). |

Feedback:

- `aim_wrist` returns "already_visible" and does not re-centre, so the wrist keeps a poor view with the target
  right behind the jaws. Needs a view that puts the target in the upper/middle of the wrist image.
- Scene detection jitters: the same case read at x -0.107 and later -0.053 (y about -0.22).

## 2026-11-22: order.md flow, steps 1-2 replaced by scene-camera localization

Temporary driver `/tmp/mc_from3.py` (not in repo): `localize_3d(camera="top")` for the 2D/base target, then
MainControl steps 3-9 in order.md order. Live bridge only, estop on failure.

- Scene localization: pixel (617, 508), base (-0.041, -0.194, 0.002) m, confidence 0.86.
- From home: step 4 `capture_wrist_stereo` "No safe sideways stereo slide preserves the target view." (no motion).
- From FRONT_POSE with gripper open: steps 3 and 4 pass (wrist ROI [233, 179, 160, 160]); step 4
  `estimate_wrist_depth` "Need at least 4 distinct target matches, got 0." The case is fully visible
  (`calib_frames/mc_run/wrist_from3.jpg`) but glossy black and textureless: feature stereo finds no matches.
  The projected ROI sits up-left of the case (top-camera error).

### Step 4 replaced by the proper sweep (scripts/wrist_stereo_sweep.py + scripts/measure_object.py)

Case had moved again (checkerboard removed). From the held pose:

- `wrist_stereo_sweep.py --target 537 275 --roi 470 170 140 210 --execute --out calib_frames/grasp_earbud2/sweep`:
  dry-run picked 3.2 cm slide at -30 deg (joint swing 14 deg; +180/+-150 refused by joint 3 limit).
  OK on attempt 1: baseline 32.5 mm, 12 matches, reprojection 0.32 px, parallax 8.9 deg, height max 33 mm.
- `measure_object.py ... --dark-v-max 45`: 41 x 34 x 37 mm, yaw 23.4, centre (13.1, -219.0) mm. Mask covers only
  the upper half (glossy half dropped).
- `measure_object.py ...` (GrabCut): 49.6 x 47.6 x 36.4 mm, yaw 130.1, centre (6.2, -219.7) mm. Likely includes
  the loop/strap; near-square footprint makes yaw unreliable.
- Feedback: the sweep is much more robust than main_control's single slide (fallback directions, dense cloud).

### Grasp with the tolerant planner (/tmp/ge_grasp2.py)

`--centre 10 -219 --yaw-deg 23 --width-mm 34 --length-mm 50 --height-mm 37`

- Plan: 28/40 IK seeds valid in 1.9 s, jaw slack +-8 mm, heading slack +-21 deg; tip 0.1 mm from centre, jaw yaw
  134 deg, vertical approach, path lowest 8.4 cm, case clearance 4.5 cm.
- Execute: hover ok (lag 11 ticks), descend ok (lag 21, final 2.1 mm off), close stopped at 2593 (open 2839,
  close 1232) = about 45 mm gap, current ~0 mA. Lift ok.
- RESULT: NOT held. Case still on the table beyond the jaws (`calib_frames/grasp_earbud2/after_wrist.jpg`).
- The script's "holding" check (position - close > 120 ticks) reported True: it cannot tell a stalled jaw
  from a grip. Needs a current or gap-change check.
- Likely causes: footprint from the sweep was partial (41 x 34 mm, true case looks wider), so the jaw closed
  on an edge or the strap; yaw 23 deg may be off.
- A hand was on the arm in the top frame after the lift.

### Retry: full footprint, grasp held

- Object-centred sweep (`--object 0.010 -0.219 0.02`) centred on the rough estimate (just above the case),
  all 3 attempts failed; re-sweep aimed at the case pixel (`--target 615 465 --roi 530 370 180 200`) OK on
  attempt 3 (14 matches). Measurement (both segmentations agree): 64 x 40 x 40 mm, yaw 165.5, centre (-5.3, -229.4) mm.
- Tolerant planner: jaw slack +-5 mm, heading +-9 deg, tip 2 mm from centre, but the reachable plan needed
  joint 5 +123 / joint 6 +166 deg (joint 6 at its lower limit; straight lift also blocked by joint 6).
- Moved to FRONT_POSE ticks: joint 2 stalled at 1832 (target 2233) for 60 s while the others moved, which left
  the jaw tips about 3 cm above the table with the case between the open jaws (FK check had assumed all joints move).
- ROOT CAUSE of the misses: each gripper command moves at most 250 ticks (`ROBOT_MAX_STEP_TICKS`), so one
  close command only moved the jaws 250 ticks. Closing in repeated 250-tick steps until the jaws stop: jaw gap 19 mm
  (pinching the rounded top). 3 cm test lift: case HELD and lifted (`calib_frames/grasp_earbud3/test_lift_*.jpg`).
  The MCP `grasp` tool (`grasp_with_verify`) already closes in repeated steps, so main_control is not affected.

### Integration

- `desk_core/tolerant_grasp.py`: `plan_tolerant_grasp()` (port of the tested planner, no motion).
- `desk_core/main_control.py`: `MainControl.plan_tolerant_grasp()` after depth estimation and measurement;
  `master_pick_place` uses its `q_hover` / `q_contact` for the pick (replaces `estimate_grasp_pose` there),
  `solve_ik` only for placement.
- `desk_core/mcp_server.py`: new MCP tool `plan_tolerant_grasp(capture_id, tool)`; `order.md` step 6 updated.
- Tests: 85 passed. Offline check on the real measurement: 35 IK solutions in 1.3 s, tip 0.2 mm from centre.
- Open issue: from FRONT_POSE the best plan still needs a large wrist rotation (157 deg max joint move).

### Drop at X with saved joint ticks (no IK at run time)

- `calib/drop_location.json` had only `position_base_m`; solved once for a release pose above it (jaw tips
  72 mm above the table, 20 deg tilt) and saved as `joint_ticks`: 2591, 2048, 1093, 949, 1683, 1076, 994.
- Replayed the ticks through Station (60-tick steps, stall detection): reached 2588, 2010, 1088, 948, 1687,
  1084, 997 in about 15 s. Wrist view shows the X centred under the jaws (`calib_frames/grasp_earbud3/drop_wrist.jpg`).
- The case was no longer in the gripper at the drop (it is in a hand in the top frame; gripper had closed
  further to 1572 = pinch lost or the case was taken). `release` reported "did not settle".

### Pipeline changes after the drop test

- `move_to_drop` (MainControl + MCP): replays `joint_ticks` from `calib/drop_location.json` through
  `_move_checked` (checked joint path, carried object included), no IK. `master_pick_place` places with it;
  placement IK removed.
- `retract_to_hover` (MainControl + MCP): after `grasp()` closes across the short side, return to the planned
  `q_hover` through a checked path and verify the grip, before `move_to_drop`.
- Table collision fix: the tolerant planner only bounded the jaw-tip midpoint (12-22 mm). With a 25 deg tilt
  the jaw boxes reached 7.9 mm BELOW the table. Now every jaw collision-box corner (URDF meshes, open and
  closed, flange pose x jaw-in-flange transform; same frame as `calib.fk_ee_pose`, verified) must stay
  8 mm above the table. Same measurement now plans 16 deg tilt, jaw lowest +7.7 mm.
- `order.md`: centre the object in the wrist frame before depth estimation; new steps documented.
- Tests: 85 passed.

Previous stop: Arm sent home with `move_to_home(max_step=60)` on the live bridge, gripper to 1256.
Final (independent live read): 2134, 2892, 934, 981, 1863, 1053, 147, gripper 1259. The first gripper
close stalled at 1515 and needed a second command.
