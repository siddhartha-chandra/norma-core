# Wrist camera stereo test

Use two images from the same ElRobot wrist camera to estimate the depth of a static target. Move the camera sideways between captures so the target shifts in the image. The module uses the paired encoder ticks, ElRobot forward kinematics, and the measured hand-eye transform to recover the camera poses. It then matches features along epipolar lines and triangulates their positions in `base_link`. [Stereo module](desk_core/wrist_stereo.py)

The stereo command works on saved captures and can run on any teammate's laptop with the files below. Only the capture step needs a connection to Station. Run the commands from the repo root on `feat/epipolar`.

## 1. Get the required files

| File | Where it comes from |
| --- | --- |
| `desk_core/wrist_stereo.py` | The standalone module on `feat/epipolar`. |
| `calib_frames/session1/camera_calibration.json` | The tracked camera intrinsics and distortion file. Both commands default to the `wrist` calibration for **1280 × 720** images. |
| `/tmp/measured_rig.json` | The measured calibration for the actual arm, with `T_ee_cam_wrist`, `joint_zero_ticks`, and `joint_signs`. The commands below create this path from the calibration branch. You can also use your own measured rig file and substitute its path. |
| `calib_frames/wrist_sweep_01/samples.json` and `wrist-*.jpg` | The capture script writes these files. The manifest pairs each JPEG with the seven arm encoder readings. Keep the JPEGs next to the manifest. |

If you're using the team's measured rig from `origin/feat/metric_depth_estimation`, extract it while staying on `feat/epipolar`:

```bash
git fetch origin
git show origin/feat/metric_depth_estimation:calib/rig_calib.json > /tmp/measured_rig.json
```

The `calib/rig_calib.json` checked into `feat/epipolar` has nominal values. Use the measured file for the same follower and wrist mount that captured your frames. The stereo command recomputes joint angles from the manifest's raw ticks with that file.

Use the repo's Python environment. For a fresh setup with `uv`, run `uv sync --python 3.12` and `source .venv/bin/activate`. The capture preview needs desktop OpenCV, and the live Station connection needs generated Python protobufs. If `target/gen_python` is missing, generate the bindings with `make protobuf`. [Repo dependencies](pyproject.toml) · [Protobuf target](Makefile)

## 2. Capture two settled wrist views

Keep Station running with its TCP endpoint and both camera streams active. Use `127.0.0.1` when the capture script runs on the Station host. From another machine, use the Station host's reachable address. The usual TCP port is `8888`.

If you need to identify the current follower bus and wrist camera, run the probe with your Station address. It prints the selected bus and discovered camera IDs and saves `cam_*.jpg` snapshots. Use the full follower bus serial and the ID for the snapshot that shows the wrist view. [Station probe](scripts/probe_station.py)

```bash
STATION_HOST=127.0.0.1 STATION_PORT=8888 python scripts/probe_station.py
```

When the correct `STATION_HOST`, `STATION_BUS_SERIAL`, and `CAMERA_WRIST` are already set in `.env`, start the capture window with:

```bash
python scripts/capture_wrist_mount.py calib_frames/wrist_sweep_01 \
  --joint-calibration /tmp/measured_rig.json
```

You can also pass `--host`, `--port`, `--bus`, and `--camera` directly. Use a new session directory for each sweep. The capture window reads video and encoders while the operator positions the arm. [Capture script](scripts/capture_wrist_mount.py)

1. Before moving, call `observe()` and check `link.rx_age_ms` and both camera views. Register tabletop obstacles and use the calibrated Pink IK or pose pipeline. If the arm stalls or anything shifts, call `estop()` and observe again. [Operator guide](AGENTS.md)
2. Place a stationary, textured target in view. Hold the arm at the reference pose until **READY**, focus the capture window, and press **Space**. Confirm the saved count increases.
3. At a safe clearance, aim for about **40 mm** of sideways translation across the wrist camera's line of sight. A robot-base X move works when that axis is mostly sideways in the wrist view. Keep wrist orientation fixed as closely as practical and keep the same target visible. Wait for **READY** again and press **Space** for the second capture.
4. Press lowercase **q** or **Esc** to finish. The session directory should contain `samples.json`, `wrist-000.jpg`, and `wrist-001.jpg`. Share the directory as a unit if a teammate will run the estimate. New capture sessions are Git-ignored.

**READY** depends on fresh video and seven encoder readings, a 0.7-second hold within three ticks per joint, and at least one joint reading 25 ticks away from each saved pose. The checkerboard text in the preview is a framing hint. You can capture a textured object once **READY** appears. [Capture conditions](scripts/capture_wrist_mount.py#L70-L109)

## 3. Set the ROI and run stereo

Open `calib_frames/wrist_sweep_01/wrist-000.jpg` at full resolution and pick a rectangle around a textured patch on the surface you want to measure. Use coordinates from the original JPEG, with the origin at the top left. The module detects features in that patch and searches for their matches in the second image.

```bash
python -m desk_core.wrist_stereo calib_frames/wrist_sweep_01/samples.json \
  --rig /tmp/measured_rig.json --pair 0 1 --roi 480 260 250 180
```

- `--pair 0 1` selects the first two manifest entries. The ROI always applies to the first entry in the selected pair.
- `--roi 480 260 250 180` means **x, y, width, height** in pixels. In this example the rectangle starts at `(480, 260)` and extends 250 pixels right and 180 pixels down. Replace the four values with your target rectangle and keep it inside the frame.
- Use `--intrinsics FILE` on the stereo command and `--calibration FILE` on the capture command if your rig uses another camera calibration. Use the same file for both, and match the wrist camera's calibrated image size.

## 4. Read and verify the result

| JSON field | Meaning |
| --- | --- |
| `baseline_m` | Distance between the two camera centers, computed from encoder-derived poses, in meters. |
| `match_count` | Number of corresponding features that passed the geometric checks. |
| `median_depth_m` | Median optical Z in the first wrist camera, in meters. Use a patch on a single surface to interpret this value. |
| `points_base_m` | The reconstructed `[x, y, z]` for each retained feature in robot `base_link`, in meters. |
| `depth_a_m`, `pixels_a`, `pixels_b`, `reprojection_error_px`, `parallax_deg` | Per-match depth, pixel coordinates in both saved images, the larger reprojection error, and the angle between the viewing rays. |

For the first accuracy check, use a textured target at independently measured heights and repeat the sweep in both directions. Compare reconstructed base-frame Z with the known height in that frame. If measuring height above the table, subtract the calibrated `z_table` from base-frame Z. A small reprojection error means the point fits the two images and the supplied camera poses. Use the physical reference to check metric accuracy.

| Symptom | What to check |
| --- | --- |
| No **READY** signal | Check the live camera and follower bus IDs, current encoder and video ages, and whether the pose differs enough from the first capture. |
| Image resolution error | Set the raw wrist stream to the calibrated resolution and keep the original JPEGs uncropped. |
| Missing manifest or JPEG | Check the session path and make sure the selected manifest and both referenced images are together. |
| `nominal rig` error | Point `--rig` to the measured JSON with the wrist hand-eye and encoder mapping. |
| `match_count` is `0` or the median is `null` | Check target overlap and camera translation. Choose a matte, textured patch on the target and verify the calibration belongs to this rig. The CLI exits with status `1` for an empty result. |

## Optional check with saved Station frames

After extracting `/tmp/measured_rig.json`, use this saved pair to check the command and file layout before collecting your own sweep:

```bash
saved_wrist_pair=$(mktemp -d /tmp/deskcore-wrist-pair.XXXXXX)
for artifact in samples.json wrist-003.jpg wrist-010.jpg; do
  git show "origin/feat/metric_depth_estimation:calib_frames/hand_eye_1/$artifact" \
    > "$saved_wrist_pair/$artifact"
done
python -m desk_core.wrist_stereo "$saved_wrist_pair/samples.json" \
  --rig /tmp/measured_rig.json --pair 3 10 --roi 340 160 600 380
```

This pair gave a baseline near **51 mm** and retained matches in the offline smoke check. The fresh fixed-orientation sweep with a physical reference is the Station accuracy test.
