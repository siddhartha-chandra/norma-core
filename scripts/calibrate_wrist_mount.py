#!/usr/bin/env python3
"""Calibrate the wrist optical frame against the ElRobot flange using a fixed board."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from recover_scene_to_base import (
    BASE_LINK, DEFAULT_CALIBRATION, DEFAULT_ELROBOT_URDF, GRIPPER_FLANGE,
    inverse_pose, load_transform, load_urdf, pose_from_image,
)


def solve_hand_eye(base_from_flange, camera_from_board):
    rotations = [
        cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0].ravel()
        for a, b in combinations(base_from_flange, 2)
    ]
    singular_values = np.linalg.svd(rotations, compute_uv=False)
    if singular_values[1] < np.deg2rad(5):
        raise ValueError("Too little rotation about distinct axes. Tilt the wrist in different directions.")
    rotation, translation = cv2.calibrateHandEye(
        [t[:3, :3] for t in base_from_flange],
        [t[:3, 3:] for t in base_from_flange],
        [t[:3, :3] for t in camera_from_board],
        [t[:3, 3:] for t in camera_from_board],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    mount = np.eye(4)
    mount[:3, :3] = rotation
    mount[:3, 3] = translation.ravel()
    return load_transform(mount, "T_ee_cam_wrist")


def board_consistency(mount, base_from_flange, camera_from_board):
    poses = [a @ mount @ b for a, b in zip(base_from_flange, camera_from_board)]
    translation = []
    rotation = []
    for a, b in combinations(poses, 2):
        translation.append(1000 * np.linalg.norm(a[:3, 3] - b[:3, 3]))
        cosine = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
        rotation.append(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
    return {
        "board_origin_pairwise_rms_mm": float(np.sqrt(np.mean(np.square(translation)))),
        "board_rotation_pairwise_rms_deg": float(np.sqrt(np.mean(np.square(rotation)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="JSON list of wrist images and paired URDF joint angles.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--joint-calibration", type=Path, help="Recompute angles from captured ticks using measured joint_zero_ticks and joint_signs")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_ELROBOT_URDF)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        samples = json.loads(args.samples.read_text())
        if not isinstance(samples, list) or len(samples) < 5:
            raise ValueError("Provide at least five wrist images from distinct poses.")
        wrist = json.loads(args.calibration.read_text())["wrist"]
        robot = load_urdf(args.urdf)
        arm_poses, camera_poses, diagnostics = [], [], []
        mapping = json.loads(args.joint_calibration.read_text()) if args.joint_calibration else None
        if mapping is not None:
            ids = [str(i) for i in range(1, 8)]
            zeros = np.array([mapping["joint_zero_ticks"][i] for i in ids])
            signs = np.array([mapping["joint_signs"][i] for i in ids])
            if not np.isin(signs, [-1, 1]).all():
                raise ValueError("Each joint sign must be +1 or -1")
        for sample in samples:
            image = args.samples.parent / sample["image"]
            angles = np.asarray(sample["joints_rad"], dtype=float)
            if mapping is not None:
                ticks = np.asarray(sample["joint_ticks"], dtype=float)
                if ticks.shape != (7,):
                    raise ValueError("Expected seven raw joint ticks")
                angles = signs * (ticks - zeros) * (2 * np.pi / 4096)
            if angles.shape != (7,) or not np.isfinite(angles).all():
                raise ValueError(f"{image}: supply seven finite URDF joint angles in radians.")
            try:
                board_from_camera, details = pose_from_image(
                    image, "wrist", wrist, sample.get("origin_pixel"), sample.get("board_roi")
                )
            except ValueError as error:
                if str(error).startswith("Full checkerboard detection failed"):
                    print(f"Skipping {image}: {error}", file=sys.stderr)
                    continue
                raise
            arm_poses.append(robot.fk(angles, target_link=GRIPPER_FLANGE, base_link=BASE_LINK))
            camera_poses.append(inverse_pose(board_from_camera))
            diagnostics.append(details)
        if len(arm_poses) < 5:
            raise ValueError(f"Only {len(arm_poses)} usable poses. Capture more distinct wrist tilts.")
        mount = solve_hand_eye(arm_poses, camera_poses)
        result = {
            "translation_units": "meters",
            "T_ee_cam_wrist": mount.tolist(),
            "frames": {"T_ee_cam_wrist": {
                "parent_frame": GRIPPER_FLANGE, "child_frame": "wrist_camera_optical_frame",
            }},
            "fit_consistency": board_consistency(mount, arm_poses, camera_poses),
            "samples": diagnostics,
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Saved wrist-camera mount to {args.output}")
        stats = result["fit_consistency"]
        print(f"Fixed-board consistency: {stats['board_origin_pairwise_rms_mm']:.2f} mm position RMS, "
              f"{stats['board_rotation_pairwise_rms_deg']:.2f} deg rotation RMS (pairwise, in-sample)")
    except (OSError, ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
