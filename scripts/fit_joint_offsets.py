#!/usr/bin/env python3
"""Jointly refine the wrist-camera mount and servo zero offsets from a fixed-board session.

The board stays fixed on the table, so every view must agree on one base-to-board
pose. Bundle adjustment over all corner reprojections solves for the wrist mount
(T_ee_cam_wrist), the board pose (T_base_board) and zero offsets for joints 2-6.
Joint 1 offsets are absorbed by T_base_board and joint 7 offsets by the mount, so
both stay fixed. Leave-one-out prediction checks that the offsets generalise.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from auto_board_origin import outer_square_mean
from calibrate_wrist_mount import board_consistency
from recover_scene_camera_pose import inverse_pose
from recover_scene_to_base import BASE_LINK, GRIPPER_FLANGE, load_urdf

TICKS_PER_RAD = 4096 / (2 * np.pi)
FREE_JOINTS = [1, 2, 3, 4, 5]


def detect_corners(gray, columns, rows):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, (columns, rows), flags)
    if found:
        return corners.reshape(-1, 2), "sb"
    boosted = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    upscaled = cv2.resize(boosted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    found, corners = cv2.findChessboardCornersSB(upscaled, (columns, rows), flags)
    if found:
        return corners.reshape(-1, 2) / 2 - 0.25, "sb_clahe_up2"
    return None, None


def to_pose(vec):
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(vec[:3])[0]
    pose[:3, 3] = vec[3:]
    return pose


def to_vec(pose):
    return np.concatenate([cv2.Rodrigues(pose[:3, :3])[0].ravel(), pose[:3, 3]])


def viewer_mapping(robot, ranges):
    """Return (zero_ticks, rad_per_tick) matching the station viewer.

    The viewer maps each motor's calibrated tick range linearly onto the URDF
    joint limits; extrapolate the same line outside the range.
    """
    zeros, scales = np.zeros(7), np.zeros(7)
    for j, (lower, upper) in enumerate(robot.joint_limits()):
        low_ticks, high_ticks = ranges[str(j + 1)]
        scales[j] = (upper - lower) / (high_ticks - low_ticks)
        zeros[j] = low_ticks - lower / scales[j]
    return zeros, scales


class Problem:
    def __init__(self, views, robot, intrinsics, grid, zeros, scales, prior_deg):
        self.views, self.robot, self.grid = views, robot, grid
        self.matrix, self.distortion = intrinsics
        self.zeros, self.scales = zeros, scales
        self.prior = np.radians(prior_deg)

    def angles(self, view, offsets):
        q = self.scales * (view["ticks"] - self.zeros)
        q[FREE_JOINTS] += offsets
        return q

    def arm(self, view, offsets):
        return self.robot.fk(self.angles(view, offsets), target_link=GRIPPER_FLANGE, base_link=BASE_LINK)

    def unpack(self, x):
        return to_pose(x[:6]), to_pose(x[6:12]), x[12:]

    def project(self, view, mount, board, offsets):
        cam_from_board = inverse_pose(mount) @ inverse_pose(self.arm(view, offsets)) @ board
        rvec = cv2.Rodrigues(cam_from_board[:3, :3])[0]
        pixels, _ = cv2.projectPoints(self.grid, rvec, cam_from_board[:3, 3], self.matrix, self.distortion)
        return pixels.reshape(-1, 2) - view["pixels"]

    def residuals(self, x, indices, fit_offsets):
        mount, board, offsets = self.unpack(x)
        if not fit_offsets:
            offsets = np.zeros(len(FREE_JOINTS))
        parts = [self.project(self.views[i], mount, board, offsets).ravel() for i in indices]
        if fit_offsets:
            parts.append(offsets / self.prior)
        return np.concatenate(parts)

    def initial(self, indices):
        arm = [self.arm(self.views[i], np.zeros(len(FREE_JOINTS))) for i in indices]
        cams = [self.views[i]["cam_from_board"] for i in indices]
        R, t = cv2.calibrateHandEye([a[:3, :3] for a in arm], [a[:3, 3:] for a in arm],
                                    [c[:3, :3] for c in cams], [c[:3, 3:] for c in cams],
                                    method=cv2.CALIB_HAND_EYE_PARK)
        mount = np.eye(4)
        mount[:3, :3], mount[:3, 3] = R, t.ravel()
        boards = [a @ mount @ c for a, c in zip(arm, cams)]
        board = boards[0].copy()
        board[:3, 3] = np.mean([b[:3, 3] for b in boards], axis=0)
        return np.concatenate([to_vec(mount), to_vec(board), np.zeros(len(FREE_JOINTS))])

    def fit(self, indices, fit_offsets):
        x0 = self.initial(indices)
        result = least_squares(self.residuals, x0, args=(indices, fit_offsets), x_scale="jac")
        if not fit_offsets:
            result.x[12:] = 0
        return result.x

    def rmse(self, x, index):
        mount, board, offsets = self.unpack(x)
        error = self.project(self.views[index], mount, board, offsets)
        return float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))


def load_views(samples_path, wrist, columns, rows):
    views = []
    for sample in json.loads(samples_path.read_text()):
        image = cv2.imread(str(samples_path.parent / sample["image"]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixels, method = detect_corners(gray, columns, rows)
        if pixels is None:
            print(f"{sample['image']}: no board, skipped", file=sys.stderr)
            continue
        first = outer_square_mean(gray, pixels[0], pixels[columns + 1])
        last = outer_square_mean(gray, pixels[-1], pixels[-(columns + 2)])
        if last < first:
            pixels = pixels[::-1].copy()
        views.append({"image": sample["image"], "method": method, "pixels": pixels,
                      "ticks": np.asarray(sample["joint_ticks"], dtype=float)})
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--zero-ticks", type=float, default=2048)
    parser.add_argument("--ranges", type=Path,
                        help='JSON {"1": [range_min, range_max], ...}; use the station viewer mapping.')
    parser.add_argument("--prior-deg", type=float, default=10.0,
                        help="Soft prior (1 sigma) keeping offsets near zero.")
    parser.add_argument("--rig-zeros", action="store_true",
                        help="With --ranges, keep only the rounded viewer zeros and the 2*pi/4096 scale "
                             "that desk_core/calib_store.py supports.")
    parser.add_argument("--mount-output", type=Path,
                        help="Write the mount-only result as a wrist-mount file for export_rig_calib.py.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rig_zeros and not args.ranges:
        parser.error("--rig-zeros needs --ranges")

    wrist = json.loads(args.calibration.read_text())["wrist"]
    board_spec = wrist["board"]
    columns, rows = board_spec["inner_columns"], board_spec["inner_rows"]
    grid = np.zeros((columns * rows, 3))
    grid[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * board_spec["square_mm"] / 1000
    matrix = np.asarray(wrist["camera_matrix"], dtype=float)
    distortion = np.asarray(wrist["distortion_coefficients"], dtype=float)

    views = load_views(args.samples, wrist, columns, rows)
    for view in views:
        ok, rvec, tvec = cv2.solvePnP(grid, view["pixels"], matrix, distortion)
        view["cam_from_board"] = to_pose(np.concatenate([rvec.ravel(), tvec.ravel()]))
    robot = load_urdf()
    if args.ranges:
        zeros, scales = viewer_mapping(robot, json.loads(args.ranges.read_text()))
        if args.rig_zeros:
            zeros, scales = np.round(zeros), np.full(7, 1 / TICKS_PER_RAD)
    else:
        zeros, scales = np.full(7, args.zero_ticks), np.full(7, 1 / TICKS_PER_RAD)
    problem = Problem(views, robot, (matrix, distortion), grid, zeros, scales, args.prior_deg)
    every = list(range(len(views)))

    report = {}
    for label, fit_offsets in (("mount_only", False), ("mount_and_offsets", True)):
        x = problem.fit(every, fit_offsets)
        mount, board, offsets = problem.unpack(x)
        arm = [problem.arm(v, offsets) for v in views]
        held_out = []
        for i in every:
            x_loo = problem.fit([j for j in every if j != i], fit_offsets)
            held_out.append(problem.rmse(x_loo, i))
        report[label] = {
            "T_ee_cam_wrist": mount.tolist(),
            "T_base_board": board.tolist(),
            "offsets_deg": dict(zip([f"joint_{j + 1}" for j in FREE_JOINTS], np.degrees(offsets).round(2).tolist())),
            "offsets_ticks": dict(zip([f"joint_{j + 1}" for j in FREE_JOINTS], (offsets * TICKS_PER_RAD).round(1).tolist())),
            "in_sample_rmse_px": [round(problem.rmse(x, i), 2) for i in every],
            "leave_one_out_rmse_px": [round(e, 2) for e in held_out],
            "fit_consistency": board_consistency(mount, arm, [v["cam_from_board"] for v in views]),
        }
        r = report[label]
        print(f"\n== {label} ==")
        print("mount t (mm):", np.round(mount[:3, 3] * 1000, 1))
        if fit_offsets:
            print("offsets deg:", r["offsets_deg"])
            print("offsets ticks:", r["offsets_ticks"])
        print(f"in-sample reproj median {np.median(r['in_sample_rmse_px']):.1f}px, "
              f"leave-one-out median {np.median(held_out):.1f}px max {max(held_out):.1f}px")
        print(f"board consistency: {r['fit_consistency']['board_origin_pairwise_rms_mm']:.1f} mm, "
              f"{r['fit_consistency']['board_rotation_pairwise_rms_deg']:.2f} deg")

    report["images"] = [{"image": v["image"], "detector": v["method"]} for v in views]
    report["zero_ticks"] = zeros.round(1).tolist()
    report["rad_per_tick"] = scales.tolist()
    report["free_joints"] = [j + 1 for j in FREE_JOINTS]
    report["translation_units"] = "meters"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {args.output}")
    if args.mount_output:
        if not np.allclose(scales, 1 / TICKS_PER_RAD):
            sys.exit("--mount-output needs a uniform 2*pi/4096 scale (use --rig-zeros or no --ranges).")
        best = report["mount_only"]
        mount_file = {
            "_notes": mount_notes(args),
            "T_ee_cam_wrist": best["T_ee_cam_wrist"],
            "T_base_board": best["T_base_board"],
            "translation_units": "meters",
            "joint_zero_ticks": {str(j + 1): int(z) for j, z in enumerate(zeros)},
            "joint_signs": {str(j + 1): 1 for j in range(7)},
            "fit_consistency": best["fit_consistency"],
            "reprojection_rmse_px": dict(zip([v["image"] for v in views], best["in_sample_rmse_px"])),
            "leave_one_out_rmse_px": dict(zip([v["image"] for v in views], best["leave_one_out_rmse_px"])),
            "method": "bundle adjustment over all views, zeros from station viewer range-to-URDF-limit mapping",
        }
        args.mount_output.write_text(json.dumps(mount_file, indent=2) + "\n")
        print(f"Saved {args.mount_output}")


def mount_notes(args):
    return {
        "transform_convention": "T_A_B is a 4x4 row-major matrix mapping points in frame B into frame A (p_A = T_A_B @ p_B), same as calib/rig_calib.json.",
        "frames": {
            "T_ee_cam_wrist": {
                "parent": "Gripper_Base_v1_1",
                "child": "wrist_camera_optical_frame",
                "why_key_not_renamed": "'ee' is the URDF link Gripper_Base_v1_1 (the ee_link / wrist_mount_link in rig_calib.json). Key kept because desk_core/calib_store.py, scripts/recover_scene_to_base.py and scripts/verify_calib.py read 'T_ee_cam_wrist'.",
            },
            "T_base_board": {
                "parent": "base_link",
                "child": "calib_board",
                "why_key_not_renamed": "'base' is the URDF link base_link. Key kept to match the T_<parent>_<child> short names used across the scripts.",
            },
        },
        "non_urdf_frames": {
            "wrist_camera_optical_frame": "Not in elrobot_follower.urdf. OpenCV optical convention: +X right, +Y down, +Z forward out of the lens. Suggested name if added to the URDF as a fixed child of Gripper_Base_v1_1.",
            "calib_board": "Not in the URDF. Origin at the board inner corner whose outer square is dark; +X along the inner columns, +Y along the inner rows, +Z into the table.",
        },
        "joint_mapping": "q_rad[j] = joint_signs[j] * (ticks[j] - joint_zero_ticks[j]) * 2*pi/4096 for URDF joints rev_motor_01..07 (keys '1'..'7'). The transforms are only valid with this mapping.",
        "reproduce": f"scripts/fit_joint_offsets.py {args.samples} --calibration {args.calibration} --ranges {args.ranges} --rig-zeros --mount-output <this file>",
    }


if __name__ == "__main__":
    main()
