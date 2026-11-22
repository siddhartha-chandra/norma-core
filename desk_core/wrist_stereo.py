"""Sparse metric stereo from two settled images of the same wrist camera.

Poses map OpenCV optical coordinates into the robot base frame, in meters.
For an offline check of a capture_wrist_mount.py session, run
``python -m desk_core.wrist_stereo --help``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class StereoResult:
    """Quality-filtered matches and geometry, with one row per retained match.

    Depth is optical Z in camera A. Reprojection error is the larger of the
    two pixel errors. Match indices refer to the input correspondences for
    triangulate_matches, or to the descriptor matches for estimate_wrist_depth.
    """

    points_base_m: np.ndarray
    depth_a_m: np.ndarray
    pixels_a: np.ndarray
    pixels_b: np.ndarray
    reprojection_error_px: np.ndarray
    parallax_deg: np.ndarray
    match_indices: np.ndarray
    baseline_m: float

    @property
    def median_depth_m(self) -> float | None:
        """Median optical depth for an ROI covering a single target surface."""
        return float(np.median(self.depth_a_m)) if self.depth_a_m.size else None


def _camera_pose(value: ArrayLike) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Each camera pose must be a finite 4x4 transform.")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("Each camera pose must contain a proper rigid transform.")
    return pose


def _geometry(camera_matrix, distortion, T_base_cam_a, T_base_cam_b):
    K = np.asarray(camera_matrix, dtype=np.float64)
    if (
        K.shape != (3, 3) or not np.isfinite(K).all()
        or K[0, 0] <= 0 or K[1, 1] <= 0
        or not np.allclose(K[2], [0, 0, 1]) or not np.allclose([K[0, 1], K[1, 0]], 0)
    ):
        raise ValueError("Expected a finite OpenCV 3x3 camera matrix with positive focal lengths.")
    dist = None if distortion is None else np.asarray(distortion, dtype=np.float64).ravel()
    if dist is not None and (dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all()):
        raise ValueError("Expected valid OpenCV distortion coefficients, or None.")
    pose_a = _camera_pose(T_base_cam_a)
    pose_b = _camera_pose(T_base_cam_b)
    pose_b_from_a = np.linalg.solve(pose_b, pose_a)
    baseline = float(np.linalg.norm(pose_b_from_a[:3, 3]))
    if baseline < 1e-6:
        raise ValueError("Stereo requires camera translation between the two views.")
    return K, dist, pose_a, pose_b_from_a, baseline


def _pixels(value: ArrayLike) -> np.ndarray:
    pixels = np.asarray(value, dtype=np.float64)
    if pixels.shape == (0,):
        return pixels.reshape(0, 2)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("Expected finite (x, y) pixel coordinates in an Nx2 array.")
    return np.ascontiguousarray(pixels)


def triangulate_matches(
    camera_matrix: ArrayLike,
    distortion: ArrayLike | None,
    T_base_cam_a: ArrayLike,
    T_base_cam_b: ArrayLike,
    pixels_a: ArrayLike,
    pixels_b: ArrayLike,
    *,
    depth_range_m: tuple[float, float] | None = None,
    max_reprojection_error_px: float = 2.0,
    min_parallax_deg: float = 0.5,
) -> StereoResult:
    """Triangulate corresponding original-image pixels from two camera poses.

    Reject points behind either camera, with poor parallax, outside the
    optional camera-A depth range, or with excessive reprojection error.
    Returns empty arrays when no correspondence passes the checks.
    """
    K, dist, pose_a, pose_b_from_a, baseline = _geometry(
        camera_matrix, distortion, T_base_cam_a, T_base_cam_b,
    )
    a, b = _pixels(pixels_a), _pixels(pixels_b)
    if a.shape != b.shape:
        raise ValueError("The two views must provide the same number of pixels.")
    if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0:
        raise ValueError("Maximum reprojection error must be positive and finite.")
    if not 0 <= min_parallax_deg < 180:
        raise ValueError("Minimum parallax must be in [0, 180) degrees.")
    if depth_range_m is None:
        near, far = 0.0, np.inf
    else:
        near, far = depth_range_m
        if not np.isfinite([near, far]).all() or not 0 < near < far:
            raise ValueError("Depth range must contain two increasing positive distances in meters.")

    indices = np.arange(len(a))
    points_a = np.empty((0, 3))
    angles = errors = np.empty(0)
    if len(indices):
        rays_a = cv2.undistortPoints(a.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        rays_b = cv2.undistortPoints(b.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        homogeneous = cv2.triangulatePoints(np.eye(3, 4), pose_b_from_a[:3], rays_a.T, rays_b.T).T
        valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1e-12)
        indices = indices[valid]
        homogeneous = homogeneous[valid]
        points_a = homogeneous[:, :3] / homogeneous[:, 3:]
        rotation, translation = pose_b_from_a[:3, :3], pose_b_from_a[:3, 3]
        points_b = points_a @ rotation.T + translation
        valid = (points_a[:, 2] > 0) & (points_b[:, 2] > 0)
        valid &= (points_a[:, 2] >= near) & (points_a[:, 2] <= far)
        indices, points_a, points_b = indices[valid], points_a[valid], points_b[valid]

    if len(indices):
        rays_b_in_a = points_b @ rotation
        angles = np.degrees(np.arctan2(
            np.linalg.norm(np.cross(points_a, rays_b_in_a), axis=1),
            np.sum(points_a * rays_b_in_a, axis=1),
        ))
        zero = np.zeros(3)
        projected_a, _ = cv2.projectPoints(points_a, zero, zero, K, dist)
        projected_b, _ = cv2.projectPoints(points_b, zero, zero, K, dist)
        errors = np.maximum(
            np.linalg.norm(projected_a.reshape(-1, 2) - a[indices], axis=1),
            np.linalg.norm(projected_b.reshape(-1, 2) - b[indices], axis=1),
        )
        valid = (angles >= min_parallax_deg) & (errors <= max_reprojection_error_px)
        indices, points_a, angles, errors = indices[valid], points_a[valid], angles[valid], errors[valid]

    return StereoResult(
        points_base_m=points_a @ pose_a[:3, :3].T + pose_a[:3, 3],
        depth_a_m=points_a[:, 2],
        pixels_a=a[indices],
        pixels_b=b[indices],
        reprojection_error_px=errors,
        parallax_deg=angles,
        match_indices=indices,
        baseline_m=baseline,
    )


def _gray(frame: np.ndarray) -> np.ndarray:
    if isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.size:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a nonempty uint8 grayscale or BGR wrist frame.")


def _distinct_matches(matcher, descriptors_a, descriptors_b, allowed):
    matches = {}
    for neighbors in matcher.knnMatch(descriptors_a, descriptors_b, k=2, mask=allowed):
        if not neighbors:
            continue
        best = neighbors[0]
        # A lone geometric candidate still needs a strong descriptor match.
        limit = 0.75 * neighbors[1].distance if len(neighbors) == 2 else 160.0
        if best.distance < limit:
            matches[best.queryIdx] = best.trainIdx
    return matches


def estimate_wrist_depth(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    camera_matrix: ArrayLike,
    distortion: ArrayLike | None,
    T_base_cam_a: ArrayLike,
    T_base_cam_b: ArrayLike,
    roi: tuple[int, int, int, int],
    *,
    depth_range_m: tuple[float, float] = (0.1, 0.8),
    max_reprojection_error_px: float = 2.0,
    min_parallax_deg: float = 0.5,
) -> StereoResult:
    """Match target features and estimate sparse depth from two wrist images.

    Supply two frames at the calibrated resolution and the measured pose at
    each capture. ROI is (x, y, width, height) in the first image. Use a tight
    ROI on the intended surface if reporting its median depth. Frames must
    show the same static scene. Textureless or ambiguous pairs can be empty.
    """
    gray_a, gray_b = _gray(frame_a), _gray(frame_b)
    if gray_a.shape != gray_b.shape:
        raise ValueError("Both wrist frames must have the same resolution.")
    if len(roi) != 4 or not all(isinstance(v, (int, np.integer)) for v in roi):
        raise ValueError("ROI must contain integer x, y, width and height.")
    x, y, width, height = roi
    rows, cols = gray_a.shape
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > cols or y + height > rows:
        raise ValueError("ROI must lie inside the first wrist frame.")
    K, dist, _, pose_b_from_a, _ = _geometry(camera_matrix, distortion, T_base_cam_a, T_base_cam_b)

    roi_mask = np.zeros(gray_a.shape, dtype=np.uint8)
    roi_mask[y:y + height, x:x + width] = 255
    keys_a, descriptors_a = cv2.SIFT_create().detectAndCompute(gray_a, roi_mask)
    pixels_a, pixels_b = [], []
    if descriptors_a is not None:
        keys_b, descriptors_b = cv2.SIFT_create(nfeatures=1500).detectAndCompute(gray_b, None)
        if descriptors_b is not None:
            a = np.array([key.pt for key in keys_a], dtype=np.float64)
            b = np.array([key.pt for key in keys_b], dtype=np.float64)
            ideal_a = cv2.undistortPoints(a.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
            ideal_b = cv2.undistortPoints(b.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
            K_inv = np.linalg.inv(K)
            rotation, translation = pose_b_from_a[:3, :3], pose_b_from_a[:3, 3]
            essential = np.cross(translation, rotation.T).T
            fundamental = K_inv.T @ essential @ K_inv
            lines_b = ideal_a @ fundamental[:, :2].T + fundamental[:, 2]
            line_residual = np.abs(lines_b[:, :2] @ ideal_b.T + lines_b[:, 2, None])
            line_norm = np.linalg.norm(lines_b[:, :2], axis=1, keepdims=True)
            allowed = np.uint8(line_residual <= 2 * max_reprojection_error_px * line_norm)

            matcher = cv2.BFMatcher(cv2.NORM_L2)
            forward = _distinct_matches(matcher, descriptors_a, descriptors_b, allowed)
            backward = _distinct_matches(matcher, descriptors_b, descriptors_a, allowed.T.copy())
            for i, j in forward.items():
                if backward.get(j) == i and x <= a[i, 0] < x + width and y <= a[i, 1] < y + height:
                    pixels_a.append(a[i])
                    pixels_b.append(b[j])

    return triangulate_matches(
        K, dist, T_base_cam_a, T_base_cam_b, pixels_a, pixels_b,
        depth_range_m=depth_range_m, max_reprojection_error_px=max_reprojection_error_px,
        min_parallax_deg=min_parallax_deg,
    )


def _main() -> int:
    import argparse
    import json
    from pathlib import Path

    from .calib_store import CalibrationStore
    from .station_bridge import MAX_ANGLE_STEP, MotorState

    default_intrinsics = Path(__file__).resolve().parents[1] / "calib_frames/session1/camera_calibration.json"
    parser = argparse.ArgumentParser(description="Estimate wrist stereo depth from two saved Station captures.")
    parser.add_argument("samples", type=Path, help="samples.json from scripts/capture_wrist_mount.py")
    parser.add_argument("--rig", required=True, type=Path, help="Measured wrist mount and encoder mapping JSON")
    parser.add_argument("--intrinsics", type=Path, default=default_intrinsics, help="Wrist camera calibration JSON")
    parser.add_argument("--pair", type=int, nargs=2, default=(0, 1), metavar=("A", "B"), help="Zero-based sample indices")
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()

    try:
        if not args.rig.is_file():
            raise ValueError(f"Missing measured rig calibration: {args.rig}")
        rig = CalibrationStore(args.rig)
        if rig.data.get("calibrated") is False:
            raise ValueError("Supply a measured rig calibration or wrist mount, not the nominal rig.")
        mount = _camera_pose(rig.data["T_ee_cam_wrist"])
        count = len(rig.joint_names())
        if rig.data.setdefault("num_arm_joints", count) != count:
            raise ValueError("Arm joint count does not match the rig's joint names.")
        for field in ("joint_zero_ticks", "joint_signs"):
            mapping = rig.data[field]
            entries = [mapping[str(i)] for i in range(1, count + 1)] if isinstance(mapping, dict) else mapping
            values = np.asarray(entries, dtype=float)
            if field == "joint_zero_ticks":
                valid = (values >= 0) & (values <= MAX_ANGLE_STEP)
            else:
                valid = np.isin(values, [-1, 1])
            if values.shape != (count,) or not valid.all():
                raise ValueError(f"{field} must have a valid entry for every arm motor.")

        wrist = json.loads(args.intrinsics.read_text(encoding="utf-8"))["wrist"]
        samples = json.loads(args.samples.read_text(encoding="utf-8"))
        if not isinstance(samples, list) or len(set(args.pair)) != 2 or not all(0 <= i < len(samples) for i in args.pair):
            raise ValueError("Choose two distinct indices in the capture manifest.")
        capture_a, capture_b = (samples[i] for i in args.pair)
        if any(capture_a.get(key) != capture_b.get(key) for key in ("camera", "bus")):
            raise ValueError("Both captures must use the same wrist camera and follower bus.")
        frames, poses = [], []
        for capture in (capture_a, capture_b):
            image = args.samples.parent / capture["image"]
            frame = cv2.imread(str(image))
            if frame is None or list(frame.shape[1::-1]) != wrist["image_size"]:
                raise ValueError(f"{image}: expected a readable image at the calibrated resolution.")
            ticks = capture["joint_ticks"]
            if len(ticks) != count or not all(type(tick) is int and 0 <= tick <= MAX_ANGLE_STEP for tick in ticks):
                raise ValueError(f"{image}: expected {count} raw encoder ticks.")
            motors = [MotorState(i, tick, 0, 0, MAX_ANGLE_STEP) for i, tick in enumerate(ticks, 1)]
            poses.append(rig.fk_ee_pose(rig.ticks_to_rad(motors)) @ mount)
            frames.append(frame)

        result = estimate_wrist_depth(
            frames[0], frames[1], wrist["camera_matrix"], wrist["distortion_coefficients"],
            poses[0], poses[1], tuple(args.roi),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    print(json.dumps({
        "baseline_m": result.baseline_m,
        "match_count": len(result.depth_a_m),
        "median_depth_m": result.median_depth_m,
        "points_base_m": result.points_base_m.tolist(),
        "depth_a_m": result.depth_a_m.tolist(),
        "pixels_a": result.pixels_a.tolist(),
        "pixels_b": result.pixels_b.tolist(),
        "reprojection_error_px": result.reprojection_error_px.tolist(),
        "parallax_deg": result.parallax_deg.tolist(),
    }, indent=2))
    return 0 if result.depth_a_m.size else 1


if __name__ == "__main__":
    raise SystemExit(_main())
