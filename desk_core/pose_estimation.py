"""Estimate a tabletop grasp pose from calibrated wrist stereo and a target box."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi

import cv2
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .wrist_stereo import StereoResult, _camera_pose


@dataclass(frozen=True)
class TopSurface:
    """Stereo points supporting a horizontal surface in robot base coordinates."""

    z_m: float
    points_base_m: np.ndarray
    pixels_a: np.ndarray


@dataclass(frozen=True)
class GraspPose:
    """TCP grasp point and the corresponding IK link pose in base_link."""

    grasp_point_base_m: np.ndarray
    object_yaw_rad: float | None
    jaw_yaw_rad: float
    surface: TopSurface
    ik_target: dict[str, float]


def _box(roi: ArrayLike) -> np.ndarray:
    box = np.asarray(roi, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[:2] < 0) or np.any(box[2:] <= 0):
        raise ValueError("ROI must contain x, y, width and height in reference-image pixels.")
    return box


def _oriented_box(corners: ArrayLike) -> np.ndarray:
    pixels = np.asarray(corners, dtype=np.float32)
    if pixels.shape != (4, 2) or not np.isfinite(pixels).all():
        raise ValueError("Supply four finite (x, y) corners for the oriented box.")
    hull = cv2.convexHull(pixels).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) <= 0:
        raise ValueError("The oriented box must have four distinct convex corners.")
    return hull


def estimate_top_surface(
    stereo: StereoResult,
    roi: ArrayLike,
    *,
    table_z_m: float | None = None,
    surface_tolerance_m: float = 0.01,
    min_matches: int = 4,
    box_corners_px: ArrayLike | None = None,
) -> TopSurface:
    """Find the best-supported horizontal height among stereo features in a box."""
    box = _box(roi)
    if not np.isfinite(surface_tolerance_m) or surface_tolerance_m <= 0:
        raise ValueError("Surface tolerance must be positive and finite.")
    if not isinstance(min_matches, (int, np.integer)) or min_matches < 3:
        raise ValueError("At least three distinct matches are required to estimate a surface.")

    pixels = np.asarray(stereo.pixels_a, dtype=float)
    points = np.asarray(stereo.points_base_m, dtype=float)
    if pixels.shape != (len(points), 2) or points.shape != (len(pixels), 3):
        raise ValueError("Stereo points and reference pixels must have matching rows.")
    valid = np.isfinite(points).all(axis=1) & np.isfinite(pixels).all(axis=1)
    valid &= np.all((box[:2] <= pixels) & (pixels < box[:2] + box[2:]), axis=1)
    if box_corners_px is not None:
        hull = _oriented_box(box_corners_px)
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] = [cv2.pointPolygonTest(hull, tuple(pixels[i]), False) >= 0 for i in valid_indices]
    if table_z_m is not None:
        if not np.isfinite(table_z_m):
            raise ValueError("Table height must be finite.")
        valid &= points[:, 2] > table_z_m + 0.002
    pixels, points = pixels[valid], points[valid]

    # SIFT can report several orientations at one pixel. Count it only once.
    _, distinct = np.unique(np.rint(pixels), axis=0, return_index=True)
    ordered = distinct[np.argsort(points[distinct, 2])]
    heights = points[ordered, 2]
    if len(heights) < min_matches:
        raise ValueError(f"Need at least {min_matches} distinct target matches, got {len(heights)}.")

    lower = np.searchsorted(heights, heights - surface_tolerance_m, side="left")
    upper = np.searchsorted(heights, heights + surface_tolerance_m, side="right")
    best = len(heights) - 1 - int(np.argmax((upper - lower)[::-1]))
    inliers = ordered[lower[best]:upper[best]]
    if len(inliers) < min_matches:
        raise ValueError(f"Need at least {min_matches} matches on one horizontal surface, got {len(inliers)}.")
    return TopSurface(float(np.median(points[inliers, 2])), points[inliers], pixels[inliers])


def pixels_on_plane(
    pixels: ArrayLike,
    camera_matrix: ArrayLike,
    distortion: ArrayLike | None,
    T_base_cam_a: ArrayLike,
    z_m: float,
) -> np.ndarray:
    """Intersect raw wrist-image rays with a horizontal base-frame plane."""
    uv = np.ascontiguousarray(pixels, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not len(uv) or not np.isfinite(uv).all():
        raise ValueError("Expected one or more finite (x, y) pixels in an Nx2 array.")
    if not np.isfinite(z_m):
        raise ValueError("Plane height must be finite.")
    pose = _camera_pose(T_base_cam_a)
    K = np.asarray(camera_matrix, dtype=np.float64)
    dist = None if distortion is None else np.asarray(distortion, dtype=np.float64).ravel()
    normalized = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = np.column_stack((normalized, np.ones(len(uv)))) @ pose[:3, :3].T
    if np.any(np.abs(rays[:, 2]) < 0.02 * np.linalg.norm(rays, axis=1)):
        raise ValueError("The target rays are too close to parallel to the surface plane.")
    scale = (z_m - pose[2, 3]) / rays[:, 2]
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("The surface must intersect all target rays in front of the camera.")
    return pose[:3, 3] + scale[:, None] * rays


def _wrap(angle: float, period: float) -> float:
    return float((angle + period / 2) % period - period / 2)


def _cloud_yaw(surface: TopSurface, box: np.ndarray) -> float | None:
    xy = surface.points_base_m[:, :2]
    if len(xy) < 6 or np.max(np.ptp(surface.pixels_a, axis=0) / box[2:]) < 0.5:
        return None
    eigenvalues, axes = np.linalg.eigh(np.cov(xy.T))
    axis = axes[:, -1]
    if eigenvalues[-1] <= 2.25 * eigenvalues[0] or np.ptp(xy @ axis) < 0.015:
        return None
    return atan2(axis[1], axis[0])


def estimate_grasp_pose(
    stereo: StereoResult,
    camera_matrix: ArrayLike,
    distortion: ArrayLike | None,
    T_base_cam_a: ArrayLike,
    roi: ArrayLike,
    *,
    tool_quaternion_xyzw: ArrayLike,
    reference_jaw_yaw_rad: float,
    tcp_offset_m: ArrayLike,
    box_corners_px: ArrayLike | None = None,
    axis_pixels: ArrayLike | None = None,
    grasp_pixel: ArrayLike | None = None,
    fixed_jaw_yaw_rad: float | None = None,
    jaw_offset_rad: float = pi / 2,
    grasp_drop_m: float = 0.0,
    table_z_m: float | None = None,
    surface_tolerance_m: float = 0.01,
    min_matches: int = 4,
) -> GraspPose:
    """Return XYZ and an xyzw quaternion for a grasp on a tabletop object's top.

    Provide a verified link orientation and its jaw direction in base XY.
    The TCP offset is expressed in the IK link frame. Axis pixels describe
    two points on the same surface in wrist image A. The object axis is
    undirected and the chosen jaw angle is the equivalent nearest the reference.
    A fixed jaw yaw explicitly sets the approach heading for a symmetric target.
    """
    box = _box(roi)
    corners = None if box_corners_px is None else _oriented_box(box_corners_px)
    surface = estimate_top_surface(
        stereo, box, table_z_m=table_z_m, surface_tolerance_m=surface_tolerance_m,
        min_matches=min_matches, box_corners_px=corners,
    )
    pixel = box[:2] + box[2:] / 2 if corners is None else corners.mean(axis=0)
    if grasp_pixel is not None:
        pixel = np.asarray(grasp_pixel, dtype=float)
    if pixel.shape != (2,) or not np.all((box[:2] <= pixel) & (pixel < box[:2] + box[2:])):
        raise ValueError("The grasp pixel must be inside the reference-image ROI.")
    if corners is not None and cv2.pointPolygonTest(corners, tuple(float(v) for v in pixel), False) < 0:
        raise ValueError("The grasp pixel must be inside the oriented box.")
    if not np.isfinite(grasp_drop_m) or grasp_drop_m < 0:
        raise ValueError("Grasp drop must be nonnegative and finite.")
    grasp = pixels_on_plane([pixel], camera_matrix, distortion, T_base_cam_a, surface.z_m)[0]
    grasp[2] -= grasp_drop_m
    if table_z_m is not None and grasp[2] <= table_z_m:
        raise ValueError("The requested grasp point is at or below the table.")

    if axis_pixels is not None:
        endpoints = np.asarray(axis_pixels, dtype=float)
        if endpoints.shape != (2, 2) or not np.all((box[:2] <= endpoints) & (endpoints <= box[:2] + box[2:])):
            raise ValueError("The object axis must have two endpoints in the reference-image ROI.")
        axis_base = pixels_on_plane(endpoints, camera_matrix, distortion, T_base_cam_a, surface.z_m)
        dx, dy = axis_base[1, :2] - axis_base[0, :2]
        yaw = atan2(dy, dx) if np.hypot(dx, dy) >= 0.005 else None
    elif corners is not None:
        xy = pixels_on_plane(corners, camera_matrix, distortion, T_base_cam_a, surface.z_m)[:, :2]
        rect = cv2.boxPoints(cv2.minAreaRect(np.ascontiguousarray(xy, dtype=np.float32)))
        edges = np.roll(rect, -1, axis=0) - rect
        lengths = np.linalg.norm(edges, axis=1)
        dx, dy = edges[np.argmax(lengths)]
        yaw = atan2(dy, dx) if lengths.max() >= 0.005 and lengths.max() > 1.2 * lengths.min() else None
    else:
        yaw = _cloud_yaw(surface, box)
    if yaw is not None:
        yaw = _wrap(yaw, pi)

    reference = np.asarray(tool_quaternion_xyzw, dtype=float)
    offset = np.asarray(tcp_offset_m, dtype=float)
    if reference.shape != (4,) or not np.isfinite(reference).all() or np.linalg.norm(reference) < 1e-12:
        raise ValueError("Supply a finite nonzero tool reference quaternion in xyzw order.")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Supply a finite tool-local XYZ TCP offset in meters.")
    if not np.isfinite([reference_jaw_yaw_rad, jaw_offset_rad]).all():
        raise ValueError("Tool reference yaw and jaw offset must be finite angles.")
    if fixed_jaw_yaw_rad is not None:
        if not np.isfinite(fixed_jaw_yaw_rad):
            raise ValueError("Fixed jaw yaw must be finite.")
        delta = _wrap(fixed_jaw_yaw_rad - reference_jaw_yaw_rad, 2 * pi)
    elif yaw is not None:
        delta = _wrap(yaw + jaw_offset_rad - reference_jaw_yaw_rad, pi)
    else:
        raise ValueError("Object yaw is ambiguous. Supply an object axis, oriented box, or an intentional fixed jaw yaw.")

    tool = Rotation.from_rotvec([0.0, 0.0, delta]) * Rotation.from_quat(reference)
    position = grasp - tool.apply(offset)
    target = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), map(float, (*position, *tool.as_quat()))))
    return GraspPose(grasp, yaw, _wrap(reference_jaw_yaw_rad + delta, 2 * pi), surface, target)


def _main() -> int:
    import argparse
    import json
    from pathlib import Path

    from .wrist_stereo import DEFAULT_WRIST_INTRINSICS, estimate_wrist_depth, load_wrist_pair

    parser = argparse.ArgumentParser(description="Estimate a grasp pose from two saved wrist captures.")
    parser.add_argument("samples", type=Path, help="Wrist capture manifest")
    parser.add_argument("--rig", type=Path, required=True, help="Measured wrist mount and encoder mapping JSON")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_WRIST_INTRINSICS)
    parser.add_argument("--pair", type=int, nargs=2, default=(0, 1), metavar=("A", "B"))
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--box-corners", type=float, nargs=8, metavar="PX", help="Four oriented-box x/y pairs")
    parser.add_argument("--axis", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Two pixels along the object axis")
    parser.add_argument("--grasp-pixel", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--tool-quat", type=float, nargs=4, required=True, metavar=("QX", "QY", "QZ", "QW"))
    parser.add_argument("--reference-jaw-yaw-deg", type=float, required=True)
    parser.add_argument("--tcp-offset-m", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--jaw-offset-deg", type=float, default=90.0)
    parser.add_argument("--fixed-jaw-yaw-deg", type=float)
    parser.add_argument("--grasp-drop-m", type=float, default=0.0)
    parser.add_argument("--table-z-m", type=float, help="Override the measured rig table height")
    parser.add_argument("--surface-tolerance-m", type=float, default=0.01)
    args = parser.parse_args()

    try:
        capture = load_wrist_pair(args.samples, args.rig, args.intrinsics, tuple(args.pair))
        stereo = estimate_wrist_depth(
            capture.frame_a, capture.frame_b, capture.camera_matrix, capture.distortion,
            capture.T_base_cam_a, capture.T_base_cam_b, tuple(args.roi),
        )
        pose = estimate_grasp_pose(
            stereo, capture.camera_matrix, capture.distortion, capture.T_base_cam_a, args.roi,
            tool_quaternion_xyzw=args.tool_quat,
            reference_jaw_yaw_rad=np.deg2rad(args.reference_jaw_yaw_deg),
            tcp_offset_m=args.tcp_offset_m,
            box_corners_px=None if args.box_corners is None else np.reshape(args.box_corners, (4, 2)),
            axis_pixels=None if args.axis is None else np.reshape(args.axis, (2, 2)),
            grasp_pixel=args.grasp_pixel,
            jaw_offset_rad=np.deg2rad(args.jaw_offset_deg),
            fixed_jaw_yaw_rad=None if args.fixed_jaw_yaw_deg is None else np.deg2rad(args.fixed_jaw_yaw_deg),
            grasp_drop_m=args.grasp_drop_m,
            table_z_m=capture.table_z_m if args.table_z_m is None else args.table_z_m,
            surface_tolerance_m=args.surface_tolerance_m,
        )
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        parser.error(str(error))

    print(json.dumps({
        "grasp_point_base_m": pose.grasp_point_base_m.tolist(),
        "object_rpy_rad": [0.0, 0.0, pose.object_yaw_rad],
        "jaw_yaw_rad": pose.jaw_yaw_rad,
        "surface_z_m": pose.surface.z_m,
        "surface_match_count": len(pose.surface.points_base_m),
        "baseline_m": stereo.baseline_m,
        "ik_target": pose.ik_target,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
