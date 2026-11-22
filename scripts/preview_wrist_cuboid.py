#!/usr/bin/env python3
"""Draw a stereo-derived object cuboid on the first saved wrist image."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk_core.pose_estimation import estimate_top_surface, pixels_on_plane  # noqa: E402
from desk_core.wrist_stereo import (  # noqa: E402
    DEFAULT_WRIST_INTRINSICS,
    estimate_wrist_depth,
    load_wrist_pair,
)


def _cuboid_pixels(pair, roi, top_z_m, table_z_m, box_corners_px=None):
    if box_corners_px is None:
        x, y, width, height = roi
        right, bottom = x + width - 1, y + height - 1
        box_corners_px = [(x, y), (right, y), (right, bottom), (x, bottom)]
    top_points = pixels_on_plane(
        box_corners_px, pair.camera_matrix, pair.distortion, pair.T_base_cam_a, top_z_m,
    )
    rectangle = cv2.minAreaRect(top_points[:, :2].astype(np.float32))
    if min(rectangle[1]) <= 1e-5:
        raise ValueError("Choose a target ROI with a nonzero footprint on the top plane.")
    footprint = cv2.boxPoints(rectangle).astype(np.float64)
    top = np.column_stack((footprint, np.full(4, top_z_m)))
    bottom = np.column_stack((footprint, np.full(4, table_z_m)))
    corners_base = np.vstack((top, bottom))
    rotation = pair.T_base_cam_a[:3, :3]
    camera_center = pair.T_base_cam_a[:3, 3]
    corners_camera = (corners_base - camera_center) @ rotation
    if np.any(corners_camera[:, 2] <= 0):
        raise ValueError("The estimated cuboid must be in front of the wrist camera.")
    zero = np.zeros(3, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        corners_camera, zero, zero, pair.camera_matrix, pair.distortion,
    )
    if not np.isfinite(projected).all() or np.max(np.abs(projected)) > 1e7:
        raise ValueError("Cuboid projection is too far outside the image.")
    return np.rint(projected.reshape(8, 2)).astype(np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="Capture manifest with both wrist images")
    parser.add_argument("--rig", type=Path, required=True, help="Measured wrist rig JSON")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_WRIST_INTRINSICS)
    parser.add_argument("--pair", type=int, nargs=2, default=(0, 1), metavar=("A", "B"))
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--box-corners", type=float, nargs=8, metavar="PX", help="Four oriented-box x/y pairs")
    parser.add_argument("--output", type=Path, required=True, help="Output image path, such as cuboid.png")
    parser.add_argument("--table-z-m", type=float, help="Object bottom height in base_link, defaults to rig z_table")
    parser.add_argument(
        "--surface-tolerance-m", type=float, default=0.01, help="Tolerance for grouping top-surface stereo points",
    )
    args = parser.parse_args()

    try:
        pair = load_wrist_pair(args.samples, args.rig, args.intrinsics, tuple(args.pair))
        table_z = pair.table_z_m if args.table_z_m is None else args.table_z_m
        if table_z is None or not np.isfinite(table_z):
            raise ValueError("Supply --table-z-m or use a rig with calibrated z_table.")

        roi = tuple(args.roi)
        corners = None if args.box_corners is None else np.asarray(args.box_corners).reshape(4, 2)
        stereo = estimate_wrist_depth(
            pair.frame_a, pair.frame_b, pair.camera_matrix, pair.distortion,
            pair.T_base_cam_a, pair.T_base_cam_b, roi,
        )
        surface = estimate_top_surface(
            stereo, roi, table_z_m=table_z, surface_tolerance_m=args.surface_tolerance_m,
            box_corners_px=corners,
        )
        height_m = surface.z_m - table_z
        if height_m <= 0:
            raise ValueError("The estimated surface must be above the cuboid bottom.")
        vertices = _cuboid_pixels(pair, roi, surface.z_m, table_z, box_corners_px=corners)

        rotation = pair.T_base_cam_a[:3, :3]
        camera_center = pair.T_base_cam_a[:3, 3]
        depth_m = float(np.median((surface.points_base_m - camera_center) @ rotation[:, 2]))
        inliers = len(surface.points_base_m)
        preview = pair.frame_a.copy()
        cv2.polylines(preview, [vertices[4:]], True, (54, 160, 84), 2, cv2.LINE_AA)
        for i in range(4):
            cv2.line(preview, tuple(vertices[i]), tuple(vertices[i + 4]), (57, 202, 106), 2, cv2.LINE_AA)
        cv2.polylines(preview, [vertices[:4]], True, (68, 242, 131), 3, cv2.LINE_AA)
        label = f"Wrist depth {depth_m * 1000:.0f} mm | height {height_m * 1000:.0f} mm | {inliers} matches"
        cv2.putText(preview, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(args.output), preview):
            raise ValueError(f"Could not write preview to {args.output}.")
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        parser.error(str(error))

    print(f"{args.output} | {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
