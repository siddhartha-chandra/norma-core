# DESK-CORE hackathon handoff

**21 November 2026 · ElRobot follower · metric depth and hand-eye calibration**

NormaCore Station is running the arm and both camera streams. DESK-CORE adds the Python interface for a desk concierge to observe the table, locate targets and command the follower. The current calibration task is to express points seen by the wrist camera in the robot base frame as the arm moves.

**Where the team left off.** The wrist and scene camera intrinsics have been measured. Live capture of wrist images and seven encoder readings works. At the documentation check, session `calib_frames/hand_eye_raw_05/` contained three 1280 × 720 JPEGs paired with seven ticks each. An offline OpenCV check found all 70 inner corners in each image. The solver requires at least five usable poses and rotation about different axes. A measured wrist mount and its held-out validation are still pending.

## Repository map and implementation history

| Area | What is implemented |
| --- | --- |
| NormaCore platform | Station, the viewer, device drivers, protobuf definitions and Station clients are supplied by the upstream repository. The follower URDF defines the seven arm joints and gripper flange. [Platform overview](../README.md) · [Follower URDF](../hardware/elrobot/simulation/elrobot_follower.urdf) |
| Station bridge | Live and mock bridges share an interface. Live positions and current come from ST3215 RX, motor ranges from inference, and JPEGs from USB video queues. The bridge discovers cameras, rebuilds subscriptions after reconnect, checks freshness and limits each motor command. Hand-eye capture also uses local receipt times for each camera and each motor on its bus. [Bridge](../desk_core/station_bridge.py) |
| Agent interface | FastMCP exposes observation, camera frames, joint angles, IK, motion, gripper control, obstacle registration and emergency stop. Composite tools include move-to-pose, pick, place and multi-step execution. Calls and frames can be recorded and inspected with the replay script. [MCP server](../desk_core/mcp_server.py) · [Run logger](../desk_core/runlog.py) · [Replay](../scripts/replay_run.py) |
| Kinematics and perception | The calibration store reads camera models, undistorts rays, converts ticks to radians and uses URDF forward kinematics for the wrist. Table-plane projection and a custom metric depth hook are supported. The IK adapter accepts an HTTP Pink service or a local Pink module and otherwise uses a nominal damped solver. Values from the nominal model are suitable for testing the software path. [Calibration store](../desk_core/calib_store.py) · [Forward kinematics](../desk_core/urdf_fk.py) · [IK adapter](../desk_core/ik_adapter.py) |
| Calibration experiments | Camera capture and solve, board-relative camera pose, checkerboard-based coin measurement, scene-to-base recovery, paired hand-eye capture and wrist mount solving. [Scripts](../scripts/) |

The branch reviewed was `feat/metric_depth_estimation` at `758c52f`. Commit `fa6db35` added reconnect, replay and smoke coverage. Commit `8d786fb` added the arm primitives and composite skills. Commit `33adc44` connected wrist FK, lens undistortion and the calibration verifier. Subsequent commits added session 1 camera calibration and loading of its nested scene/wrist format. At this snapshot, the five scripts for coin measurement, camera pose recovery, scene-to-base recovery, wrist mount solving and wrist capture were untracked working-tree files. The bridge receipt-time changes were also local. A clean checkout at the cited commit needs those working-tree changes to reproduce the capture workflow.

## Frames, units and measured inputs

Transform names follow `T_A_B`: multiply coordinates in frame B by this matrix to obtain coordinates in frame A. Translations use meters, joint angles use radians, and encoder readings use 12-bit ticks. Both optical frames use OpenCV axes (+X right, +Y down, +Z forward).

| Input or transform | Meaning and current evidence |
| --- | --- |
| Camera intrinsics | Session 1 provides the matrix and distortion for each 1280 × 720 camera. Reprojection RMS was 0.181 px for wrist and 0.240 px for scene. The physical board has 10 × 7 inner corners and 15 mm squares. [Session 1 calibration](../calib_frames/session1/camera_calibration.json) |
| `T_base_ee(q)` | The URDF chain gives the pose of `Gripper_Base_v1_1` in `base_link` for seven joint angles. Motor 8 operates the gripper. Physical accuracy depends on the measured zero tick and direction of each arm motor. The tracked rig file contains placeholders. |
| `T_ee_cam_wrist` | Maps wrist optical coordinates into the flange. The follower URDF has no wrist optical frame. This is the fixed transform being estimated. At runtime, `T_base_cam_wrist(q) = T_base_ee(q) @ T_ee_cam_wrist`. |
| `T_world_cam_top`, `H_table`, `z_table` | The fixed scene camera is called **top** in DESK-CORE. Its base-frame pose, or a tabletop homography for planar mapping, and the tabletop height are separate calibration inputs. A board-relative camera pose alone does not establish the board’s location in the robot base. |

The calibration store chooses an explicit `DESK_CALIB_PATH` first, then a rig file marked calibrated, then the newest session camera calibration by file modification time, then the nominal rig file. It loads one selected file. To deploy multiple measurements together, assemble the measured fields in the same active file. [Calibration contract](../calib/README.md)

## Live hand-eye capture

Keep Station and the web viewer running. The capture window subscribes to Station and sends no motor or torque commands. The verified local endpoint was TCP port 8888, the follower bus was `5B8E115307`, and the wrist camera ID was `11391:19040:1:12`. The scene camera was `11391:19040:1:10`. Check the routing after reconnecting or replacing USB devices.

From the repository root, with the Station Python client dependencies, generated Python protobufs and desktop OpenCV available, launch a **new** session directory:

```bash
python3 scripts/capture_wrist_mount.py calib_frames/hand_eye_next \
  --host 127.0.0.1 --port 8888 \
  --bus 5B8E115307 --camera 11391:19040:1:12
```

1. Fix the board relative to the arm base. Pick one recognizable physical corner of the inner grid as the origin for the entire session. Check that the camera is rigidly mounted.
2. Manually pose the follower, keeping the full grid in the wrist image. Pause until **READY - press SPACE to save** appears. Focus the window and press **Space**. Confirm the saved count increases.
3. Collect different flange orientations, including left/right and forward/backward tilts. Aim for 15–20 samples to allow for unusable images and held-out checks. Translation alone gives poor rotational coverage. Keep the board fixed throughout.
4. Press lowercase **q** or **Esc** to finish. To append more poses after closing the window, launch with the same directory and add `--resume`. Keep the physical board in the same position. If combining sessions later, verify the board stayed fixed relative to the base and preserve the correct image path for every record.

**What READY checks.** Capture requires all seven arm motors on the selected bus with clear status, fresh receipts for wrist video and every joint (under 0.35 s), and a steady pose for 0.7 s within three ticks per joint. The frame must arrive during the hold. Compared with every saved pose, at least one joint must differ by 25 ticks. Pairing uses host receipt times, so hold still through the keypress. A low-resolution background check advises on framing; READY refers to image and joint pairing. The offline solver checks the full-resolution board.

**Why capture changed.** The original preview waited for a full-resolution checkerboard search each frame. In a five-run check on a saved wrist image without a detected board, the median search took 243 ms. Capture now saves the original unannotated JPEG and defers corner detection. The READY fix tracks joint stillness independently of short video delays and shows joint/video receipt ages when data is stale. Live checks confirmed the READY signal with both Station camera streams active.

| Symptom | What to check |
| --- | --- |
| Waiting for fresh data | Read the joint and video ages in the overlay. Check the selected follower bus and wrist stream in Station. A clear board cannot compensate for a stale stream. |
| Pose already saved | Tilt the wrist to a visibly different orientation, then hold still. |
| READY, but no file | Give the preview keyboard focus, press Space while READY is visible and check the on-screen count. The manifest is created after the first save. |
| Resolution error | Match the live stream to the 1280 × 720 wrist intrinsics. The capture tool rejects a mismatch. |
| Missing board offline | Inspect the original JPEG for clipped corners, blur, glare or occlusion. The solver skips images for which it cannot detect the full grid. |

## What is on disk

Each capture writes `wrist-NNN.jpg` and updates `samples.json` using a temporary file and rename. Each manifest record includes the image filename, seven raw `joint_ticks` ordered by motor ID 1–7, derived `joints_rad`, the derived `T_base_ee`, camera and bus IDs, and one monotonic receipt time for the frame and for each joint. The raw ticks allow the offline solve to use corrected encoder offsets without recapturing images.

| Local directory | Contents at review |
| --- | --- |
| `calib_frames/session1/` | Measured wrist/scene intrinsics. The camera calibration JSON is tracked. |
| `calib_frames/coin_smoke/` | Board pose and coin-measurement experiments, including a scene coin estimate of 24.33 mm and a scene-to-base experiment. These demonstrate the geometry workflow. Physical base-frame accuracy still needs independent validation. |
| `calib_frames/wrist_handeye/` | Three earlier diagnostic image/encoder pairs in a different JSON layout. The capture solver expects a list in the newer manifest format. |
| `calib_frames/hand_eye_live_01/`, `hand_eye_raw_01/` through `04/` | Earlier preview launches. No saved JPEGs at review. |
| `calib_frames/hand_eye_raw_05/` | First verified capture-only manifest with three detectable grids at review. Capture count may grow. |

Session output is generally Git-ignored. When handing data to a teammate, copy the manifest **and its JPEGs together**, along with the intrinsics and the measured joint mapping used for the solve.

## Solve, validate and load the result

The board remains fixed, so each sample should give the same value for `T_base_board = T_base_ee(q) @ T_ee_cam_wrist @ T_cam_board`. The solver detects the board using the wrist intrinsics, obtains its pose in the camera, computes flange poses from the URDF and solves for the common wrist mount with OpenCV’s Park hand-eye method.

Once there are at least five usable, distinct poses, run the following from a graphical desktop. Replace the mapping path with an actual measured rig JSON containing `joint_zero_ticks` and `joint_signs` keyed by motor IDs "1" through "7". If the mapping argument is omitted, the solver uses the angles recorded at capture, which can reflect nominal defaults.

```bash
session=calib_frames/hand_eye_raw_05
python3 scripts/calibrate_wrist_mount.py "$session/samples.json" \
  --calibration calib_frames/session1/camera_calibration.json \
  --joint-calibration /path/to/measured_rig.json \
  --output "$session/wrist_mount.json"
```

For each detected grid without a stored origin, click the **same physical origin** among the two highlighted diagonal endpoints and press Enter. The solver checks the available rotational diversity. Its output contains the 4 × 4 mount, parent and child frame names, per-image reprojection diagnostics and pairwise fixed-board position and orientation RMS in mm and degrees.

Inspect those metrics, then test additional poses excluded from the fit. For each held-out pose, compose the three transforms above and compare the predicted position of the marked corner in the base frame. Evaluate the discrepancy against the grasp task’s tolerance. The reported fit consistency uses the training samples; the capture workflow has yet to demonstrate held-out accuracy on this arm.

After validation, copy the measured `T_ee_cam_wrist` into the selected rig calibration alongside the correct intrinsics and joint mapping. The standalone solver output is a partial calibration. Use MCP `reload_config()` to reread the active calibration, then inspect `get_calib_status()` for its path, transforms and missing fields. [Solver](../scripts/calibrate_wrist_mount.py) · [Calibration verifier](../scripts/verify_calib.py)

## Verification and operating boundaries

The live capture window, both camera identifiers, fresh follower telemetry, saved image/encoder pairs and offline board detections were observed. A synthetic 12-pose test recovered a known wrist mount. Physical mount accuracy, measured encoder mapping and end-to-end calibrated pick-and-place remain open.

The repository’s offline verification commands are:

```bash
uv run --offline --managed-python python scripts/test_reconnect.py
uv run --offline --managed-python python scripts/mcp_smoke.py
uv run --offline --managed-python python scripts/test_primitives.py
uv run --offline --managed-python python scripts/verify_calib.py
```

During the repository review, the first three commands stopped before test execution because the managed interpreter pinned in `.python-version` was unavailable offline on this host. The desktop capture was separately exercised with system Python. For future **motion** sessions, start with MCP `observe()`, check RX freshness and both camera views, register tabletop obstacles and use the calibrated pose pipeline. On an unexpected shift or stall, call `estop()` and observe again. [Operator guide](../AGENTS.md)
