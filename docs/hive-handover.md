# Hive tile sorting: hand-over

Written 2026-11-22 for an agent that picks this work up with no earlier context. Read the whole document before you change anything. The rules in section 3 override everything else here.

Three kinds of statement appear below, and they are labelled:

- **Verified**: checked today by running code or reading the file named.
- **From the log**: taken from `robot_setup.md`, a teammate's record of runs on the real arm. Not re-checked.
- **Proposed**: a design that nobody has run. Treat it as a starting point and change it when the arm disagrees.

## 1. What the user wants

One command makes the arm tidy a scattered set of tiles from the board game Hive:

1. Use the tiles that show their image face, up to about 12 of them. Flipping is out of scope.
2. Split the tiles into an off-white group (slightly yellow, not white) and a black group.
3. Within each colour, stack tiles that show the same insect into one pile.
4. Put the off-white piles in one straight row and the black piles in a second straight row, with the two rows facing each other. The insect in one row does not have to match the insect across from it.

The user also says the arm moves too slowly and jerkily, and wants the workflow to look smooth.

Settled by the user (section 6): no flipping, no insect matching across the rows, and the server runs on the Station machine. Still open: whether tiles with no image face are in the set (section 6).

## 2. Where things stand

All **verified** today unless marked.

| Item | State |
| --- | --- |
| Repository | repository root (`norma-core`) |
| Branch | `fix/live-ik-and-agent-tui` |
| Last commit | `965bde1`, the merge of `origin/feat/final`. Local only, not pushed. |
| Uncommitted | Two small fixes (below) and this document. |
| Real arm | Nothing was run on the real arm today. |

### The merge

`origin/feat/final` held a teammate's work: wrist stereo depth, a grasp planner, a numbered pick-and-place workflow (`desk_core/main_control.py`), saved home and drop poses, and stricter bridge checks. It contains everything that was on `feat/epipolar`. It is merged. Two files conflicted and were resolved like this:

| Place | Resolution |
| --- | --- |
| `send_joint_targets` in `desk_core/station_bridge.py` | Took `feat/final`'s rule: use the motor's effective range and refuse a move that cannot bring the motor into range in one step. This branch's older rule (ignore the range when the motor sits outside it) is gone. **The user has not confirmed this choice.** |
| `go_home` | Kept this branch's table-view pose and `look_at_table` argument, added the motion lock. |
| `move_joint` | Kept the smooth stream from this branch. |
| `solve_ik` | Kept the `valid_bounds` check, runs the solver in a worker thread. |
| `pick_at`, `place_at` | Kept this branch's joint-space path. Applied `feat/final`'s checks (stop if the gripper did not open, close or release) to both the joint path and the pose path. A pick now reports `ok` only when the gripper reports `holding`. |

### The two uncommitted fixes

1. `release()` in `desk_core/mcp_server.py`. It compared the commanded step with `motor.position` after sending. The mock bridge changes that same object in place, so the check always read "no progress" and `release()` failed in mock mode. It failed the same way on `feat/final` itself. The fix stores the position before sending.
2. `bounds_rad()` in `desk_core/calib_store.py`. The inverse kinematics (IK) solver was given the motor tick range (motor 4: about 1.7626 rad either way), but `move_joint` validates against the tighter URDF limit (1.7533 rad). A solution in the gap passed IK and was rejected after the arm had already moved to hover. `bounds_rad()` now returns the overlap of the two limits. The `pick_pixel` check in `scripts/test_primitives.py` failed about half the time because of this, and now passes.

### Test results on the mock

| Command | Result |
| --- | --- |
| `pytest` | 99 passed |
| `scripts/test_primitives.py` | 34 checks passed, 6 runs out of 6 |
| `scripts/test_reconnect.py` | passed |
| `scripts/mcp_smoke.py` | passed |

### Environment

- The merge added `open3d` as a development dependency. It is installed, so `uv run --offline` works.
- `import open3d` fails until the system library is installed with `brew install libusb`. Only `scripts/plot_stereo_cloud.py` and `scripts/measure_object.py` import it. Nothing in the workflow needs it.

## 3. Rules

1. **Use the mock unless the user asked for a live run in your session.** Importing `desk_core.mcp_server` reads `.env`, and `.env` here sets `STATION_HOST`, so a plain script talks to the real arm. Set `DESK_MOCK=1 DESK_NO_DOTENV=1` for every script and test you run on your own.
2. **Move the real arm only with the user at the arm.** Run every new motion as a dry run first (`execute=False`), show the plan, and ask before the first real move.
3. Follow `AGENTS.md`: start with `observe()`, register obstacles before moving, use the IK tools and not raw tick guesses, and call `estop()` if anything shifts or stalls.
4. Do not hand-edit calibrated values in `calib/rig_calib.json`. Put Hive settings in a new file, `calib/hive_layout.json`.
5. Keep the four commands in section 9 passing after every change. Do not loosen a safety check to make a test pass.
6. Commit only when the user asks. Never push unless asked.
7. Stay inside `AGENT_BOUNDS` in `desk_core/motion.py` for all speed settings.
8. Stop and ask the user when a question in section 6 blocks your stage, when a hardware experiment fails twice, or when the fix would be a physical change to the rig.

## 4. Code map

| File | What to use from it |
| --- | --- |
| `desk_core/mcp_server.py` | The tools the agent and the TUI call. `_motion_tool` lets one motion tool run at a time. `_traced` logs every call with `duration_ms` to `runs/<timestamp>/calls.jsonl`. `_SEQ_TOOLS` lists what `execute_sequence` may call. `move_joint` streams one smooth move. `go_home` and `move_to_home` still move in stop-and-go steps. |
| `desk_core/main_control.py` | `MainControl`. `_motor_state()` returns a fresh copy of telemetry or raises. `_checked_path()` checks a joint path against the table and registered obstacles. `_move_checked()` runs a checked path as a smooth stream and confirms the arm settled. `FAST_LIMITS` is the speed profile it uses. `_jaw_bounds()` gives the jaw collision corners. |
| `desk_core/tolerant_grasp.py` | `plan_tolerant_grasp(...)`. Plans top-down grasp and hover joint angles from an object's centre, heading and box size. It needs no camera. |
| `desk_core/motion.py` | `MotionLimits`, `AGENT_BOUNDS`, `stream_joint_path`, `move_joints_smooth`, `correct_final_pose`, `TCP_IN_FLANGE`, `lowest_point_m`. |
| `desk_core/station_bridge.py` | `MockStationBridge`, `LiveStationBridge`, the per-command step cap, the range rule, the emergency-stop latch, `grasp_with_verify`. |
| `desk_core/calib_store.py` | `pixel_to_world(x, y, z_m, camera, q_rad)` turns a pixel into table coordinates at a height you give. `project_world_to_pixel` goes the other way. |
| `desk_core/stereo_sweep.py` | `aim_wrist`, `plan_view_pose`, `look_at_pose` plan a pose that points the wrist camera at a spot. `joint_path` builds a joint path. |
| `desk_core/vision_3d.py` | Calls to the vision models. The prompt and parser handle one object per call. |
| `desk_core/command_planner.py` | Templates that turn a typed or spoken command into tool steps. |
| `scripts/agent_tui.py` | The terminal UI. `MOTION_TOOLS` lists tools behind the motion interlock. `_execute_command` routes a command. |
| `scripts/wrist_stereo_sweep.py` | `centre_object()` is a working example of "move, look, re-centre". |
| `robot_setup.md` | The log of real-arm runs. Read it before any hardware stage. |
| `order.md` | The order of steps in the existing pick-and-place workflow. |

## 5. What is known about the rig

| Fact | Source |
| --- | --- |
| The arm has 7 joints plus a gripper (motor 8). Positions are in metres in the `base_link` frame. The table is at z = 0.00237. | Verified, `calib/rig_calib.json` |
| The table surface is white. The scene camera (id `top`, 1280x720) looks at the arm from the front at an angle, not straight down. | Verified, `calib_frames/scene_1/scene.jpg` |
| The Hive set is dual colour: off-white tiles (slightly yellow) and black tiles. At most about 12 tiles are used. Sizes: about 38 mm across the flats, 44 mm across the corners, 12 mm thick, as the user believes. | User, not measured |
| The work area in front of the arm is at negative y. Positive x is to the right in the scene camera image. | Derived from logged positions. Confirm with one `pixel_to_world` call. |
| The scene camera places objects about 2 cm off. One object was read 5 cm apart in two detections. | From the log |
| The wrist camera (id `wrist`) is on the gripper. Closed jaws hide the lower half of its image, so open the gripper before looking. Even with the jaws open, a target close to them can be half hidden at the bottom of the image. | From the log |
| The jaws open to 54 mm. Gripper ticks: 2864 open, 1232 closed. The jaw-tip midpoint is 103 mm along the flange +Y axis (`TCP_IN_FLANGE`). | Verified in files; 54 mm is from the log |
| `calib/rig_calib.json` has no `grasp_tool` entry, so `master_pick_place(execute=True)` stops at `validate_setup` unless a tool is passed in. The values used are in `robot_setup.md`. | Verified |
| Wrist stereo depth often found no matches on plain, glossy objects. | From the log |
| Each `send_joint_targets` command moves a motor at most `ROBOT_MAX_STEP_TICKS`. The code default is 400. The local `.env` sets 250. | Verified |
| Servo speed and acceleration come from `ROBOT_GOAL_SPEED` (600) and `ROBOT_GOAL_ACCEL` (20). They are written when torque is enabled. | Verified |
| `.env` points `STATION_HOST` at a remote address, not `127.0.0.1`. The log says to run on the Station machine. | Verified |
| Saved poses: home in `calib/fixed_positions.json`, drop point in `calib/drop_location.json`. | Verified |

## 6. Questions for the user

### Answered by the user

- **Edition and count.** Dual-colour edition: off-white and black tiles. At most about 12 tiles are used. The user believes the sizes in stage 1 (about 38 mm across the flats, 44 mm across the corners, 12 mm thick) are right. They are not measured yet.
- **Colour.** The light tiles are slightly yellow, not white. Do not treat them as white.
- **Flipping.** Out of scope. Stage 6 is kept only as a record.
- **Layout.** The two rows do not need the same insect opposite each other.
- **Server.** The MCP server runs on the Station machine.

### Still open (ask before the stage that needs them)

1. **A mat** (stages 2, 4). May a soft, plain, mid-tone mat go under the tiles? Off-white tiles have little contrast against the white table, and a soft surface protects the jaws.
2. **Tiles with no image face** (stages 1, 5). Are there any tiles showing their back? Without flipping, such a tile cannot be sorted. The plan skips it and reports it.
3. **The range rule** from the merge (section 2). Keep the stricter rule?

## 7. Work plan

Stages 0 to 2 need no arm. Stages 3 to 5 need the arm and the user. If the arm is free first, do stage 3 first, because speed is the user's main complaint.

### Stage 0: confirm the baseline

Run the four commands in section 9. All must pass before you change anything. If one fails, stop and report it.

### Stage 1: tile model and sorting plan (proposed)

New file `desk_core/hive.py`, pure Python, no arm and no camera. Tests in `tests/test_hive.py`.

- `Tile`: `color` (`"black"` or `"offwhite"`), `insect` (a name, or `None` when the tile cannot be read), `x` and `y` (tile centre in metres), `yaw` (direction of a corner-to-corner line in radians, reduced to the range 0 to 60 degrees). Add a `face` field only if question 2 says tiles with no image face exist.
- `calib/hive_layout.json`: tile size, the y value of each colour's row, the x of the first pile, the gap between piles, and the insect order that fixes each insect's slot in a row. Every number in it is a placeholder until the reach check below passes. Both rows can use the same insect order; the user has not asked for matching across rows.
- `pile_spot(layout, color, insect)`: the row comes from the colour and the slot from the insect order.
- `next_action(tiles, piles, layout)`: `piles` counts the tiles already stacked at each spot, because the camera sees only the top tile of a pile. Rules in order: stack the next sorted-but-not-yet-stacked tile at its pile spot, at height `piles[key]`; an unreadable tile is reported and not stacked; otherwise finish. A tile counts as stacked when its centre is within 2 cm of its pile spot. Refuse a plan with more than 12 tiles.

Spacing: the jaws need room. Close them along the direction that runs from one row to the other, so the open jaws do not reach toward the neighbouring piles in the same row. Keep about 6 cm between pile centres and at least 10 cm between the rows.

Reach check: write a script that calls `plan_tolerant_grasp` for every pile spot at every stack height, starting from the home pose, and prints which spots have a solution. Move the layout until all do.

Done when: the tests cover slot assignment, pile heights, unreadable tiles, the 12-tile limit, the finish condition, and the reach check passes for the layout.

### Stage 2: finding and reading tiles (proposed)

New file `desk_core/hive_vision.py`. Build it against real images, not against guesses.

1. With the user, save at least 3 scene frames and 10 wrist close-ups under `calib_frames/hive/`. Cover both colours and every insect in the set. Write the true colour and insect for each tile in a JSON file next to the images.
2. `find_tiles_top(image)`: a rough pixel centre and colour for every tile in a scene frame.
3. `inspect_tile(image, expected_pixel)`: for a wrist close-up, the tile nearest the expected pixel, with its pixel centre, heading, colour and insect.
4. Turn pixels into table positions with `calib.pixel_to_world(u, v, z_m=..., camera=..., q_rad=...)`. Pass the height of the tile's top face: table height plus thickness times the number of tiles in the stack. The thickness is known, so stereo depth is not needed.

Accuracy needed: the jaws leave about 6 mm of slack around a 38 mm tile, and a neat pile needs about 3 mm. The scene camera is about 2 cm off, so use it only to find tiles roughly and take the exact position from a wrist close-up.

Method: with a mat, separate tiles from the mat by colour, keep shapes of the right size, and tell black from off-white by brightness, comparing each tile with the table around it (the off-white tiles are slightly yellow, so do not assume a pure white threshold). Read the insect from the engraving colour and pattern. If these rules fail on the saved images, ask a vision model. That needs a function that sends your own prompt and returns the raw text, because the existing calls in `desk_core/vision_3d.py` build a one-object prompt inside. Add it without changing what the existing calls return.

Testing without the arm: the mock bridge serves a saved image when you call `bridge.set_frame_override(camera, jpeg_bytes)` or set `DESK_MOCK_TOP_IMAGE` and `DESK_MOCK_WRIST_IMAGE`.

Done when: on the saved images every tile is found, colour is right for all of them, the insect is right for every readable tile, and wrist centres are within 3 pixels of centres marked by hand.

### Stage 3: speed and smoothness (arm and user)

Measure before changing anything. Run three fixed moves and record, for each, the duration from `runs/<timestamp>/calls.jsonl` and the `run` result of the stream (`cycles`, `elapsed_s`, `worst_lag_ticks`). `move_joint` returns only true or false today, so log the `run` result while you measure. `cycles / elapsed_s` should be close to 10. If it is well below 10, the loop is stalling on input or output, and the network path to Station is the first suspect.

Causes visible in the code, in the order I would test them. Change one thing at a time.

1. **`go_home` stops and starts.** It sends a step, waits for the arm to settle, and repeats up to 10 times. `place_at` calls it after every place by default, and the `pick_and_place` command template uses that default. In the calibrated branch `go_home` already has the pose in radians (`q_list`), so it can call `move_joint(joints=q_list)`. Keep the old loop for the uncalibrated branch.
2. **`move_joint` is capped at 25 degrees per second** (`MOVE_JOINT_SPEED_DEG_S`). I chose that from mock runs, and that was a mistake. The mock jumps to each target at once, so its `worst_lag_ticks` is not servo lag. On the arm, raise the speed in steps (25, 35, 45) while the lag stays under about half the limit. In the mock that number still grows with speed (73 ticks at 30 degrees per second and over 80 at 40 in my runs). If a higher default trips the guard in the mock tests, keep the tests at 25 through a setting. Do not loosen the guard.
3. **Every smooth move ends with a correction.** `move_joints_smooth` always calls `correct_final_pose`, which nudges and waits up to 4 times. Add a `correct` argument and use it only for the last approach to a tile.
4. **Servo settings.** `ROBOT_GOAL_ACCEL=20` and `ROBOT_GOAL_SPEED=600` are conservative. An older example in `old_solutions` used speeds from 300 to 3300. Try acceleration 40, then 60, then speed 1000. The values are written when the server first enables torque, so restart the server for a change to apply.
5. **Stream rate.** `MotionLimits.rate_hz` is 10. Try 20.
6. **Network.** The server runs on the Station machine (confirmed), so the network is not in the path. Check the link status (`rx_age_ms` from `observe()`) before blaming the code.

Agree a target with the user. A proposal: a 60 degree joint move finishes in 3 seconds or less with no stop in the middle, and `go_home` shows no steps.

### Stage 4: pick one tile and stack it (arm and user, proposed)

The limit to know first: `plan_tolerant_grasp` puts the jaw tips 12 to 22 mm above the table by default and keeps the jaws 8 mm above it, and `MainControl._checked_path` refuses anything under 5 mm. A tile lying on the table is about 12 mm thick, so with the defaults the jaws close above it. A tile on top of another tile is high enough.

- Wrap `plan_tolerant_grasp` in a tile planner. Pass the tile's width across the flats, its length across the corners, `tcp_in_flange=TCP_IN_FLANGE`, the jaw corners from `_jaw_bounds`, `max_tilt_deg=15` and `long_tol_mm=2`. `tip_height_mm` is measured from the table, so for a tile resting on `n` other tiles use `(n * thickness + 4, n * thickness + 8)`, and pass `height_mm=(n + 1) * thickness`.
- For a tile on the table, lower `jaw_floor_clearance_m` to 3 mm. Do this only on a soft mat (question 1). Add a `table_clearance_m` argument to `_checked_path` and `_move_checked` with the current value as the default. Do not edit the shared constants.
- To place, plan a grasp at the pile spot as if the tile were already there, with `opening_mm` set to the width plus 4 so the planner allows no sideways slack. Then move to hover, move down, release, and move back to hover.
- To pick: release, move to hover, move down, grasp, check `holding`, move back to hover.

Experiments, each as a dry run first: pick a tile from the mat five times, then place one tile on another and measure the offset.

Done when: five picks out of five succeed and the stacked tile is within 3 mm of the one below.

Stage 4 also needs the answer to question 1 (the mat) before the table-contact step.

### Stage 5: the full loop, the tool and the command (proposed)

New file `desk_core/hive_workflow.py` with a `HiveSorter` class and one `run(execute=False)` method:

1. `observe()`.
2. Find tiles roughly in a scene frame.
3. Ask `next_action`. If there is none, finish.
4. Point the wrist camera at the tile, read it exactly, and update the tile.
5. Stack the tile at its pile spot.
6. Repeat. Stop after three actions per tile so a failure cannot loop forever. Skip a tile that fails three times and report it.

A dry run returns the planned actions from the first scene frame and moves nothing.

Wiring:

- A tool `hive_sort(execute: bool = False)` in `desk_core/mcp_server.py` with the decorators `@mcp.tool()`, `@_traced`, `@_motion_tool` in that order. Add it to `_SEQ_TOOLS`.
- Add `"hive_sort"` to `MOTION_TOOLS` in `scripts/agent_tui.py`.
- In `desk_core/command_planner.py`, add a `sort_hive` entry to `TEMPLATES`, return `{}` for it in `extract_slots`, and return one `hive_sort` step in `_steps`. Add a test to `tests/test_command_planner.py`. `plan_command` asks an outside service (Jev, key `TYPE_SAFE_API_KEY`) to choose the template. Check what the TUI does with the command when that service is unavailable.
- For tests, follow `tests/test_main_control.py`: the mock bridge, saved images, and a stand-in for `_move_checked`.

Done when: a dry run on saved images lists a correct plan, the mock test runs a whole sort, and one typed command starts it in the TUI.

### Stage 6: flipping (removed from scope)

The user has ruled out flipping. Nothing in stages 1 to 5 depends on it. The options below are kept only as a record.

- **Option A, tilt and release.** Pick the tile, lift it, tilt the gripper past vertical about the line between the two jaws, and release it just above a raised block so it falls onto its other face. It needs no second grasp but may bounce. The loop reads the tile again afterwards, so a failed flip is retried.
- **Option B, re-grip and roll.** Place the tile on the block with half of it over the edge. Grip the overhanging half from the side with one jaw above and one below. Lift, turn the last joint (the wrist roll, joint 7) by 180 degrees, and set it down. It is more certain but needs two grasps and a sideways approach.
- **Fallback.** Move the tile to a marked spot and ask a person to turn it, using `speak_status`.

Before either option, check with the IK solver, off the arm, that the poses can be reached with the gripper clear of the table.

## 8. Traps

- **`.env` makes scripts live.** See rule 1.
- **Telemetry objects change under you.** `get_state()` returns the bridge's own motor objects on both the mock and the live bridge. Copy a value before you `await` anything. `MainControl._motor_state()` returns copies.
- **The emergency-stop latch.** After `estop()`, or after a stream aborts on lag, every motion raises `EStopLatched` until `observe()` runs. `MainControl.observe()` also clears it. Do not call `observe()` inside a retry loop without deciding that it is safe to continue.
- **One motion tool at a time.** A motion tool called from a second task while one is running raises "Another desk-core motion operation is running." Inside a tool, call other tools directly.
- **The mock does not show servo lag.** See stage 3, cause 2.
- **Vision can fall back without telling you.** `localize_3d` returns `backend == "opencv-fallback"` when the vision model call fails. `MainControl.detect_requested_object` raises in that case.
- **A failed grasp leaves the arm down.** `pick_at` returns without retracting when the gripper reports an error.
- **Step cap.** The local `.env` caps each command at 250 ticks. That also slows the gripper.

## 9. Commands

Run from the repository root.

```bash
uv run --offline --managed-python python -m pytest -q
uv run --offline --managed-python python scripts/test_primitives.py
uv run --offline --managed-python python scripts/test_reconnect.py
uv run --offline --managed-python python scripts/mcp_smoke.py
```

A script of your own, on the mock:

```bash
DESK_MOCK=1 DESK_NO_DOTENV=1 uv run --offline --managed-python python your_script.py
```
