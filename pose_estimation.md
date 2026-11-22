# Grasp pose from wrist stereo

The grasp estimator takes two settled wrist captures, the measured rig calibration, and a detector box in the first image. It estimates grasp position and object yaw in `base_link` and returns a solver-independent IK target. The pose has six degrees of freedom. It is serialized as XYZ and an `xyzw` quaternion, which uses four numbers to represent three rotational degrees of freedom. For upright tabletop objects, the object roll and pitch are set to zero. The IK quaternion combines the estimated yaw with the gripper's verified approach orientation. [Grasp estimator](desk_core/pose_estimation.py) · [Wrist stereo test](epipolar.md)

## Run the grasp estimator

From the repo root, follow the wrist stereo guide to capture a pair. This example uses `calib_frames/wrist_sweep_01/samples.json` with its JPEGs alongside it, the team's measured `calib/rig_calib.json`, and the default `calib_frames/session1/camera_calibration.json` for wrist intrinsics. Run in the repo's Python environment with NumPy, OpenCV, Pillow, and SciPy. [Capture and calibration setup](epipolar.md) · [Dependencies](pyproject.toml)

```bash
python -m desk_core.pose_estimation calib_frames/wrist_sweep_01/samples.json \
  --rig calib/rig_calib.json --pair 0 1 --roi 480 260 250 180 \
  --tool-quat QX QY QZ QW --reference-jaw-yaw-deg REF_YAW \
  --tcp-offset-m TCP_X TCP_Y TCP_Z
```

Replace the example ROI with the target box in the original reference JPEG. The capitalized values are numeric inputs from the tool calibration:

- `QX QY QZ QW` is the IK link's orientation at a verified approach pose. For a `Gripper_Base_v1_1` target, read the settled arm ticks at that pose, convert them with the measured rig, and use the link's forward kinematics. Convert the resulting rotation matrix with `Rotation.from_matrix(R).as_quat()` to get `xyzw` ordering.
- `REF_YAW` is the jaw-closing direction at that same pose, measured in degrees from base +X toward +Y. Use the direction of jaw travel projected into base XY.
- `TCP_X TCP_Y TCP_Z` is the offset from the IK link origin to the fingertip contact point, in link coordinates and meters. Use `0 0 0` when the IK link origin is already at your chosen tool center point.

| Optional flag | What it controls |
| --- | --- |
| `--box-corners X0 Y0 X1 Y1 X2 Y2 X3 Y3` | Four convex corners of a rotated box around the top surface. The estimator uses the polygon to filter stereo matches and determines its long axis after projecting the corners into base XY. |
| `--axis X0 Y0 X1 Y1` | Two reference-image pixels along the physical object axis. Takes precedence over the rotated box for orientation. |
| `--grasp-pixel X Y` | The chosen contact pixel. The default is the box center. |
| `--jaw-offset-deg ANGLE` | Jaw-closing direction relative to object yaw. The default is `90`, across the object's long axis. |
| `--fixed-jaw-yaw-deg ANGLE` | Explicit base-frame jaw direction for a symmetric target or a deliberate fixed-heading grasp. |
| `--grasp-drop-m METERS` | Vertical distance the TCP should sit below the observed top. The default `0` keeps it on the top surface. |
| `--table-z-m METERS` | Override the measured rig's `z_table` with the known support-surface height in `base_link`. |
| `--surface-tolerance-m METERS` | Half-width of the height band used to group stereo points. The default is `0.01`. |
| `--intrinsics FILE` | Calibrated intrinsics and distortion JSON for the original wrist images. |

The command prints JSON. `grasp_point_base_m` is the contact point and `ik_target` is the corresponding link pose with `x`, `y`, `z`, `qx`, `qy`, `qz`, and `qw`. `object_rpy_rad` reports `[0, 0, yaw]`, with a `null` yaw allowed when you supply a fixed heading. `jaw_yaw_rad`, `surface_z_m`, `surface_match_count`, and `baseline_m` help diagnose the estimate. The command works offline on saved captures. Downstream code can call `estimate_grasp_pose(...)` and pass its `ik_target` to whichever quaternion IK interface is deployed. [Python API](desk_core/pose_estimation.py)

## Inputs and outputs

| Input | Use |
| --- | --- |
| Detector box in the reference wrist image | Selects the target region in pixels. A mask, rotated box, or a pair of labeled points on the object can also supply an axis. |
| `points_base_m` and `pixels_a` from wrist stereo | Each row pairs a metric 3D point with its pixel in the reference image. Use these to estimate the target surface and grasp position. |
| Wrist intrinsics, distortion, and `T_base_cam_a` | Map reference-image pixels to rays in `base_link` using the arm pose at the time the image was captured. |
| Gripper approach orientation and fingertip offset | Convert the chosen contact position and jaw direction to the pose of the link controlled by IK. Confirm the reference configuration on the actual rig. |

Keep the pose estimate and the IK request explicit. The estimate contains `grasp_point_base_m`, `object_yaw_rad` when available, and enough diagnostics to judge the measurement. The IK request contains `x`, `y`, `z` in meters and `qx`, `qy`, `qz`, `qw` in that order. The IK position is the origin of the controlled link. [Stereo result fields](desk_core/wrist_stereo.py#L36-L51) · [Calibration frames](calib/README.md#conventions)

Use the detection from the same wrist frame as `pixels_a`. If the fixed scene camera finds the object first, associate it with a fresh wrist detection before using the wrist ROI. A second option is to project the stereo points into the scene image with the calibrated scene camera pose and test them against the scene box. The detector format `[ymin, xmin, ymax, xmax]` uses a different order from `--roi`. If normalized to 0–1000, multiply X values by `image_width / 1000` and Y values by `image_height / 1000` before computing `(x, y, width, height)`. [Detection schema](desk_core/interfaces.py#L39-L47) · [World-to-pixel projection](desk_core/calib_store.py#L391-L415)

## 1. Estimate `x`, `y`, `z`

1. Select the stereo matches inside the ROI using `pixels_a` and apply the oriented box polygon when you supply one. For a box-only detector, prefer an inset patch around the intended surface to reduce background and table matches.
2. Group the remaining `points_base_m` by height. The implementation keeps the best-supported height band, with at least four distinct feature locations by default, and takes its median base-frame Z as `z_surface`. When table height is available, points within 2 mm of the table or below it are discarded. The test requires a coherent patch on a flat top above the support surface.
3. Choose a grasp pixel on that surface. Undistort its pixel to a camera ray, transform the ray into `base_link`, and intersect it with `z = z_surface`. This gives the requested point even when the stereo features lie away from the center of the patch. Set the actual fingertip height using the object's grasp geometry and the measured table clearance.

For a wrist pixel `p`, `undistort(p)` below gives normalized optical coordinates `(u_n, v_n)`. Write `T_base_cam_a` as rotation `R` and camera center `C`, both expressed in `base_link`.

```text
ray = R @ [u_n, v_n, 1]
s = (z_surface - C.z) / ray.z
point_base = C + s * ray
```

Require a positive intersection with a ray that crosses the plane at a useful angle. An object with a nonplanar grasp surface needs a reliable 3D match near the intended contact or a surface model. `median_depth_m` from stereo is camera optical Z. The `z_m` input to `pixel_to_world` specifies a base-frame plane height. Use `points_base_m` when estimating that height. [Stereo depth convention](desk_core/wrist_stereo.py#L36-L51) · [Pixel-to-world convention](desk_core/calib_store.py#L358-L382)

## 2. Estimate yaw around `base_link` +Z

A standard axis-aligned box provides the region to search. Its orientation is tied to the image axes, so the object can rotate inside the box while the box edges remain horizontal and vertical. For an elongated object, get its axis from an **oriented box**, the top-surface mask or contour, or two points along a physical edge. An image-plane angle can also supply the direction if the detector defines its image axes and the axis endpoints can be placed on the same object surface.

1. Pick two distinct pixels on the long axis of the top surface. For an oriented box fitted to that surface, use the midpoints of its two short edges. If the detector gives true stereo-matched axis keypoints, use their reconstructed 3D points directly.
2. Project both pixels onto the same `z_surface` plane using the reference camera transform and the ray calculation above. This accounts for the camera's angle and perspective. If foreshortening makes the image's long edge ambiguous, project the top-surface contour and fit its oriented rectangle in base XY.
3. For endpoints `P0` and `P1` in `base_link`, compute `object_yaw_rad = atan2(P1.y - P0.y, P1.x - P0.x)`. The angle is measured from base +X toward +Y. An undirected long axis is equivalent at `yaw` and `yaw + π`.

With a plain box, the implementation tries PCA on the filtered 3D points projected into base XY. It requires at least six locations, sufficient image coverage, and a clear dominant axis. Unevenly textured objects can still bias this estimate, so prefer an oriented top-surface box or physical axis when one is available. A near-square rotated box or a weak PCA result leaves yaw unavailable. Use an explicitly configured jaw heading for a symmetric target if its shape allows it.

## 3. Build the quaternion for IK

**Object yaw describes the object. Tool yaw describes where the jaws should close.** For an elongated object gripped across its width, choose a closing direction perpendicular to its long axis. Other grasp strategies can choose a different offset. For an undirected object axis, the implementation chooses the 180°-equivalent heading closest to the supplied reference jaw direction. The downstream IK stack can check reachability and joint limits.

Record a verified reference pose in which the fingers approach the table at the desired angle. Let `q_ref` be the IK link's orientation in `base_link`, and let `jaw_yaw_ref` be the direction in which its jaws close, measured in base XY at that pose. Rotate the reference about **base +Z** to align the jaws with the desired direction.

```text
delta = desired_jaw_yaw - jaw_yaw_ref
q_z = [0, 0, sin(delta / 2), cos(delta / 2)]
q_base_ee = normalize(q_z ⊗ q_ref)
```

Here `⊗` means quaternion multiplication with the base-frame rotation on the left. This keeps the reference approach angle. A pure yaw quaternion `[0, 0, sin(yaw / 2), cos(yaw / 2)]` is the full IK orientation only if the solver's identity pose already gives the required tool approach. The ElRobot solver targets `Gripper_Base_v1_1`. The modeled fingers extend approximately along local +Y, and jaw travel is nearly along local X. A 90° rotation about Y leaves local +Y horizontal when applied directly to that link. Verify the reference pose on the deployed solver and the actual rig. [ElRobot jaw joints and mesh origins](hardware/elrobot/simulation/elrobot_follower.urdf#L370-L408) · [Joint axes](hardware/elrobot/simulation/elrobot_follower.urdf#L487-L520)

If the selected position is the fingertip or tool-center point, apply the measured link-to-TCP offset before sending the pose. For a fixed vector `t_ee_tcp` from the IK link origin to the desired TCP, expressed in the link frame, use `p_base_ee = p_base_tcp - R(q_base_ee) @ t_ee_tcp`. Send `p_base_ee` and `q_base_ee` together.

### Which IK interface to call

The merged MCP server accepts `solve_ik(x, y, z, qx, qy, qz, qw)`. Pass the fields in `ik_target` as its pose arguments to check the solution. For a verified pick, `pick_at` accepts the same XYZ and quaternion fields. Pass the estimated quaternion explicitly so the gripper uses the calibrated approach and jaw yaw. The pose estimator remains independent of the configured IK backend. [MCP IK](desk_core/mcp_server.py#L369-L394) · [Pick path](desk_core/mcp_server.py#L752-L772)

## First validation

Use a flat, elongated block with a marked axis and independently measured position and height. Test at several known headings, including an oblique angle, and repeat the wrist sweep from the other side. Compare the reported grasp point and the base yaw modulo 180°. Try a round target with a fixed jaw heading and a weak stereo pair to exercise the failure handling. Once the estimate is stable, check the solved approach and jaw direction at a clear hover before any descent. [Station capture guide](epipolar.md) · [Operator guide](AGENTS.md)

### Draw a depth cuboid on the reference image

Use the same capture pair and ROI. The preview script gets depth from the two images, fits a rectangular footprint at the observed top height, and draws its projection on image A. It uses the rig's `z_table` for the bottom. Supply `--table-z-m` if your rig JSON lacks that value or the object sits on a different measured support. The script needs camera and rig calibration, with no tool calibration inputs. [Preview script](scripts/preview_wrist_cuboid.py)

```bash
python scripts/preview_wrist_cuboid.py calib_frames/wrist_sweep_01/samples.json \
  --rig calib/rig_calib.json --pair 0 1 --roi 480 260 250 180 \
  --output /tmp/wrist-cuboid.png
```

Open `/tmp/wrist-cuboid.png` to inspect the wireframe, median optical depth, height above the support, and number of surface matches. Pass the same `--box-corners` as the grasp command for a tighter preview of a rotated object. The cuboid is a visual check for a rectangular footprint on a known support. Use a fresh target with a measured height to check metric accuracy.
