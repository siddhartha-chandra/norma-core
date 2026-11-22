#!/usr/bin/env python3
"""Solve T_base_cam_scene from one scene view of the fixed board and the wrist-mount fit.

T_base_cam_scene = T_base_board @ inv(T_cam_scene_board). A wrist view captured at the
same time checks that the board has not moved since the wrist-mount session.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from auto_board_origin import outer_square_mean
from fit_joint_offsets import TICKS_PER_RAD, detect_corners, to_pose
from recover_scene_camera_pose import inverse_pose
from recover_scene_to_base import BASE_LINK, GRIPPER_FLANGE, load_urdf


def board_pose(image_path, calibration):
    board = calibration["board"]
    columns, rows = board["inner_columns"], board["inner_rows"]
    gray = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
    pixels, method = detect_corners(gray, columns, rows)
    if pixels is None:
        raise SystemExit(f"{image_path}: board not detected")
    first = outer_square_mean(gray, pixels[0], pixels[columns + 1])
    last = outer_square_mean(gray, pixels[-1], pixels[-(columns + 2)])
    if last < first:
        pixels = pixels[::-1].copy()
    grid = np.zeros((columns * rows, 3))
    grid[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * board["square_mm"] / 1000
    matrix = np.asarray(calibration["camera_matrix"], dtype=float)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=float)
    _, rvec, tvec = cv2.solvePnP(grid, pixels, matrix, distortion)
    projected, _ = cv2.projectPoints(grid, rvec, tvec, matrix, distortion)
    rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1))))
    return to_pose(np.concatenate([rvec.ravel(), tvec.ravel()])), rmse, method


def pose_difference(a, b):
    cosine = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return 1000 * float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="capture.json from capture_scene_board.py")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--wrist-mount", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    capture = json.loads(args.capture.read_text())
    calibration = json.loads(args.calibration.read_text())
    mount_fit = json.loads(args.wrist_mount.read_text())
    base_board = np.asarray(mount_fit["T_base_board"])
    mount = np.asarray(mount_fit["T_ee_cam_wrist"])

    scene_board, scene_rmse, scene_method = board_pose(args.capture.parent / capture["scene_image"], calibration["scene"])
    base_scene = base_board @ inverse_pose(scene_board)

    zeros = np.array([mount_fit["joint_zero_ticks"][str(j)] for j in range(1, 8)], dtype=float)
    signs = np.array([mount_fit["joint_signs"][str(j)] for j in range(1, 8)], dtype=float)
    q = signs * (np.asarray(capture["joint_ticks"], dtype=float) - zeros) / TICKS_PER_RAD
    base_flange = load_urdf().fk(q, target_link=GRIPPER_FLANGE, base_link=BASE_LINK)
    wrist_board, wrist_rmse, _ = board_pose(args.capture.parent / capture["wrist_image"], calibration["wrist"])
    moved_mm, moved_deg = pose_difference(base_flange @ mount @ wrist_board, base_board)

    result = {
        "_notes": {
            "transform_convention": "T_A_B maps points in frame B into frame A (p_A = T_A_B @ p_B), row-major, meters.",
            "frames": {
                "T_base_cam_scene": {
                    "parent": "base_link",
                    "child": "scene_camera_optical_frame",
                    "why_key_not_renamed": "Matches the T_world_cam_top role in calib/rig_calib.json, where world_frame is base_link and 'top' is this scene camera.",
                },
            },
            "non_urdf_frames": {
                "scene_camera_optical_frame": "Not in elrobot_follower.urdf. OpenCV optical convention: +X right, +Y down, +Z forward out of the lens.",
            },
            "depends_on": str(args.wrist_mount),
        },
        "T_base_cam_scene": base_scene.tolist(),
        "translation_units": "meters",
        "scene_camera": capture.get("scene_camera"),
        "scene_reprojection_rmse_px": round(scene_rmse, 3),
        "scene_detector": scene_method,
        "board_moved_check": {
            "wrist_view_vs_T_base_board_mm": round(moved_mm, 1),
            "wrist_view_vs_T_base_board_deg": round(moved_deg, 2),
            "wrist_reprojection_rmse_px": round(wrist_rmse, 3),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.set_printoptions(precision=4, suppress=True)
    print("T_base_cam_scene:\n", base_scene)
    print(f"scene camera position in base (mm): {np.round(base_scene[:3, 3] * 1000, 1)}")
    print(f"scene board reprojection {scene_rmse:.2f}px ({scene_method})")
    print(f"board-moved check from wrist view: {moved_mm:.1f} mm, {moved_deg:.2f} deg")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
