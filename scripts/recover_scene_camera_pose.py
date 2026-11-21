#!/usr/bin/env python3
"""Estimate a scene or wrist camera pose from the calibrated 15 mm checkerboard."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from measure_coin import DEFAULT_CALIBRATION, board_correspondences


def inverse_pose(transform):
    rotation = transform[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def recover_pose(points_mm, pixels, matrix, distortion):
    points = np.ascontiguousarray(points_mm / 1000.0)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(points, pixels, matrix, distortion)
    if not ok:
        raise ValueError("Could not solve for the checkerboard pose.")
    rotation, _ = cv2.Rodrigues(rvec)
    camera_from_board = np.eye(4)
    camera_from_board[:3, :3] = rotation
    camera_from_board[:3, 3] = tvec.ravel()
    if np.any((points @ rotation.T + tvec.ravel())[:, 2] <= 0):
        raise ValueError("The estimated checkerboard lies behind the camera.")
    projected, _ = cv2.projectPoints(points, rvec, tvec, matrix, distortion)
    error = projected.reshape(-1, 2) - pixels
    return inverse_pose(camera_from_board), float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))


def align_board_origin(pixels, origin_pixel):
    origin = np.asarray(origin_pixel, dtype=float)
    if not np.isfinite(origin).all():
        raise ValueError("Origin pixel must be finite.")
    distance = np.linalg.norm(pixels[[0, -1]] - origin, axis=1)
    if distance.min() > 12:
        raise ValueError("Mark the physical origin near one of the two detected diagonal endpoints.")
    return pixels[::-1].copy() if distance[1] < distance[0] else pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Uncropped camera image with the reference board.")
    parser.add_argument("--camera", choices=("scene", "wrist"), default="scene")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--board-roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--manual-board", action="store_true",
                        help="Click the four extreme inner corners in board order.")
    parser.add_argument("--origin-pixel", nargs=2, type=float, metavar=("U", "V"),
                        help="Marked physical origin's approximate pixel for automatic detection.")
    parser.add_argument("--board-origin-m", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Measured inner-corner origin in robot base_link meters.")
    parser.add_argument("--board-yaw-deg", type=float,
                        help="Angle from robot +X to board +X, toward robot +Y in degrees. Board must be horizontal.")
    parser.add_argument("--z-table-m", type=float, help="Height of the bare tabletop in base_link meters.")
    parser.add_argument("--coin-result", type=Path, help="Optional independent coin measurement JSON for this camera.")
    parser.add_argument("--coin-ground-truth-mm", type=float, default=23.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        camera = json.loads(args.calibration.read_text())[args.camera]
        image = cv2.imread(str(args.image))
        if image is None or list(image.shape[1::-1]) != camera["image_size"]:
            raise ValueError(f"Use a readable, uncropped {args.camera} image at {camera['image_size']}.")
        has_base = args.board_origin_m is not None and args.board_yaw_deg is not None
        if not has_base and any(v is not None for v in (args.board_origin_m, args.board_yaw_deg, args.z_table_m)):
            raise ValueError("Supply both --board-origin-m and --board-yaw-deg for robot-frame output.")
        if has_base and not np.isfinite([*args.board_origin_m, args.board_yaw_deg]).all():
            raise ValueError("The board origin and yaw must be finite.")
        if args.z_table_m is not None and not np.isfinite(args.z_table_m):
            raise ValueError("The table height must be finite.")
        if not np.isfinite(args.coin_ground_truth_mm) or args.coin_ground_truth_mm <= 0:
            raise ValueError("Coin ground truth must be finite and positive.")
        points, pixels, method = board_correspondences(
            image, camera["board"], args.manual_board, args.board_roi
        )
        if method == "automatic" and args.origin_pixel is not None:
            pixels = align_board_origin(pixels, args.origin_pixel)
        elif has_base and method == "automatic":
            raise ValueError("Provide --origin-pixel to identify the measured physical corner, or use --manual-board.")
        elif has_base and not args.manual_board:
            raise ValueError("To use manually selected reference corners in base_link, pass --manual-board explicitly.")
        matrix = np.asarray(camera["camera_matrix"], dtype=float)
        distortion = np.asarray(camera["distortion_coefficients"], dtype=float)
        pose, rmse = recover_pose(points, pixels, matrix, distortion)
        result = {
            "camera": args.camera,
            "image": str(args.image),
            "translation_units": "meters",
            f"T_board_cam_{'top' if args.camera == 'scene' else 'wrist'}": pose.tolist(),
            "board_origin_explicit": args.origin_pixel is not None or args.manual_board,
            "board_origin_pixel": pixels[0].tolist(),
            "board_positive_x_pixel": pixels[camera["board"]["inner_columns"] - 1 if method == "automatic" else 1].tolist(),
            "checkerboard_method": method,
            "checkerboard_points": len(points),
            "checkerboard_reprojection_rmse_px": rmse,
        }
        if has_base:
            yaw = np.deg2rad(args.board_yaw_deg)
            c, s = np.cos(yaw), np.sin(yaw)
            world_from_board = np.eye(4)
            # For a printed grid ordered clockwise in the image, +Z points into
            # the board. Use an upward robot Z and map board +Y accordingly.
            normal = np.sign(pose[2, 3])
            world_from_board[:3, :3] = [[c, -normal * s, 0], [s, normal * c, 0], [0, 0, normal]]
            world_from_board[:3, 3] = args.board_origin_m
            result[f"T_world_cam_{'top' if args.camera == 'scene' else 'wrist'}"] = (world_from_board @ pose).tolist()
            if args.z_table_m is not None:
                result["z_table"] = args.z_table_m
        if args.coin_result:
            coin = json.loads(args.coin_result.read_text())
            if coin.get("camera", "scene") != args.camera:
                raise ValueError(f"Use a {args.camera} camera coin measurement.")
            measured = float(coin["diameter_mm"])
            if not np.isfinite(measured) or measured <= 0:
                raise ValueError("Measured coin diameter must be finite and positive.")
            error = measured - args.coin_ground_truth_mm
            result["coin_check"] = {
                "estimated_diameter_mm": measured,
                "ground_truth_diameter_mm": args.coin_ground_truth_mm,
                "signed_error_mm": error,
                "absolute_percent_error": abs(error) / args.coin_ground_truth_mm * 100,
            }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Checkerboard reprojection RMSE: {rmse:.3f} px ({len(points)} corners)")
        print(f"Camera height above checkerboard: {abs(pose[2, 3]) * 1000:.1f} mm")
        print(f"Saved {'robot-frame and ' if has_base else ''}checkerboard-frame pose to {args.output}")
        if "coin_check" in result:
            check = result["coin_check"]
            print(f"Coin diameter error: {check['signed_error_mm']:+.2f} mm ({check['absolute_percent_error']:.2f}%)")
    except (OSError, ValueError, KeyError, cv2.error) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
