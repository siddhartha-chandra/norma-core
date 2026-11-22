#!/usr/bin/env python3
"""Recover the scene-camera pose from paired checkerboard views and ElRobot wrist FK."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from measure_coin import DEFAULT_CALIBRATION, board_correspondences, select_pixels
from recover_scene_camera_pose import align_board_origin, inverse_pose, recover_pose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desk_core.urdf_fk import DEFAULT_ELROBOT_URDF, load_urdf

BASE_LINK = "base_link"
GRIPPER_FLANGE = "Gripper_Base_v1_1"


def load_transform(value, name):
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix.")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1]) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError(f"{name} must be a rigid homogeneous transform.")
    return transform


def pose_from_image(path, name, calibration, origin, roi):
    image = cv2.imread(str(path))
    if image is None or list(image.shape[1::-1]) != calibration["image_size"]:
        raise ValueError(f"Use a readable uncropped {name} image at {calibration['image_size']}.")
    points, pixels, method = board_correspondences(
        image, calibration["board"], False, roi, interactive_fallback=False
    )
    if origin is None:
        for pt in pixels[[0, -1]]:
            cv2.circle(image, tuple(pt.astype(int)), 9, (0, 255, 255), 2)
        origin = select_pixels(image, "Board origin", "Click the same physical origin at one of the two circled corners, then press Enter.", 1, 1)[0]
    pixels = align_board_origin(pixels, origin)
    transform, rmse = recover_pose(
        points, pixels,
        np.asarray(calibration["camera_matrix"], dtype=float),
        np.asarray(calibration["distortion_coefficients"], dtype=float),
    )
    return transform, {
        "image": str(path), "board_origin_pixel": pixels[0].tolist(),
        "checkerboard_method": method, "checkerboard_points": len(points),
        "checkerboard_reprojection_rmse_px": rmse,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_pose", type=Path, help="Scene pose JSON, or scene image with --images.")
    parser.add_argument("wrist_pose", type=Path, help="Wrist pose JSON, or wrist image with --images.")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    for camera in ("scene", "wrist"):
        parser.add_argument(f"--{camera}-origin", nargs=2, type=float, metavar=("U", "V"),
                            help="Pixel near the same physical inner corner in both images.")
        parser.add_argument(f"--{camera}-roi", nargs=4, type=int,
                            metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--wrist-mount", type=Path,
                        help=f"Measured wrist optical frame to {GRIPPER_FLANGE}, as a JSON matrix or T_ee_cam_wrist entry.")
    parser.add_argument("--joints-rad", type=float, nargs=7,
                        help="Measured URDF joint angles 1–7 in radians at wrist image capture.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_ELROBOT_URDF)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = {"translation_units": "meters"}
        if args.images:
            if args.scene_origin is None or args.wrist_origin is None:
                raise ValueError("Identify the same physical inner corner with --scene-origin and --wrist-origin.")
            calibration = json.loads(args.calibration.read_text())
            if calibration["scene"]["board"] != calibration["wrist"]["board"]:
                raise ValueError("The two calibrations specify different checkerboards.")
            board_from_scene, result["scene"] = pose_from_image(
                args.scene_pose, "scene", calibration["scene"], args.scene_origin, args.scene_roi
            )
            board_from_wrist, result["wrist"] = pose_from_image(
                args.wrist_pose, "wrist", calibration["wrist"], args.wrist_origin, args.wrist_roi
            )
            result["board"] = calibration["scene"]["board"]
            result["T_board_cam_top"] = board_from_scene.tolist()
            result["T_board_cam_wrist"] = board_from_wrist.tolist()
        else:
            scene = json.loads(args.scene_pose.read_text())
            wrist = json.loads(args.wrist_pose.read_text())
            if not scene.get("board_origin_explicit") or not wrist.get("board_origin_explicit"):
                raise ValueError("Identify the same physical board corner and +X edge in both pose solves.")
            if scene.get("camera") != "scene" or wrist.get("camera") != "wrist":
                raise ValueError("Provide the scene pose first and wrist pose second.")
            board_from_scene = load_transform(scene["T_board_cam_top"], "T_board_cam_top")
            board_from_wrist = load_transform(wrist["T_board_cam_wrist"], "T_board_cam_wrist")
        wrist_from_scene = inverse_pose(board_from_wrist) @ board_from_scene
        result["T_cam_wrist_cam_top"] = wrist_from_scene.tolist()
        missing = []
        if args.wrist_mount is None:
            missing.append("measured T_ee_cam_wrist")
        else:
            mount = json.loads(args.wrist_mount.read_text())
            if isinstance(mount, dict):
                if mount.get("calibrated") is False:
                    raise ValueError("The wrist mount file is marked uncalibrated. Supply a measured transform.")
                parent = mount.get("frames", {}).get("T_ee_cam_wrist", {}).get(
                    "parent_frame", mount.get("wrist_mount_link", GRIPPER_FLANGE)
                )
                if parent != GRIPPER_FLANGE:
                    raise ValueError(f"Wrist mount parent must be {GRIPPER_FLANGE}, got {parent}.")
                mount = mount["T_ee_cam_wrist"]
            flange_from_wrist = load_transform(mount, "T_ee_cam_wrist")
            result["T_ee_cam_wrist"] = flange_from_wrist.tolist()
            result["frames"] = {"T_ee_cam_wrist": {
                "parent_frame": GRIPPER_FLANGE, "child_frame": "wrist_camera_optical_frame",
            }}
        if args.joints_rad is None:
            missing.append("seven URDF joint angles at wrist image capture")
        elif not np.isfinite(args.joints_rad).all():
            raise ValueError("The seven joint angles must be finite.")
        if missing:
            result["missing_for_T_world_cam_top"] = missing
        else:
            base_from_flange = load_urdf(args.urdf).fk(
                args.joints_rad, target_link=GRIPPER_FLANGE, base_link=BASE_LINK
            )
            result["T_world_cam_top"] = (base_from_flange @ flange_from_wrist @ wrist_from_scene).tolist()
            result["frames"]["T_world_cam_top"] = {
                "parent_frame": BASE_LINK, "child_frame": "scene_camera_optical_frame",
            }
            result["joints_rad_at_capture"] = args.joints_rad
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Saved pose data to {args.output}")
        if missing:
            print("Still needed for T_world_cam_top: " + ", ".join(missing))
    except (OSError, ValueError, KeyError, cv2.error) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
