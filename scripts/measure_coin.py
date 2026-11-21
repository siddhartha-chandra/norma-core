#!/usr/bin/env python3
"""Measure a coin on the checkerboard in a saved camera image."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parents[1] / "calib_frames/session1/camera_calibration.json"
)


def pixel(value):
    try:
        x, y = map(float, value.split(","))
        if np.isfinite([x, y]).all():
            return (x, y)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Expected a pixel as u,v.")


def select_pixels(image, title, prompt, minimum, maximum):
    points = []
    message = ""
    window = "Coin measurement"

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < maximum:
            points.append((x, y))

    print(prompt, flush=True)
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            preview = image.copy()
            for index, (x, y) in enumerate(points, 1):
                cv2.circle(preview, (x, y), 4, (0, 255, 255), 1)
                cv2.putText(preview, str(index), (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            caption = f"{title} | {len(points)} selected | Enter accept | Backspace undo | Esc quit"
            cv2.putText(preview, caption, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.57, (0, 0, 0), 5)
            cv2.putText(preview, caption, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.57, (255, 255, 255), 1)
            if message:
                cv2.putText(preview, message, (12, 53), cv2.FONT_HERSHEY_SIMPLEX,
                            0.57, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(40) & 255
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_AUTOSIZE) < 0:
                raise ValueError("Measurement cancelled.")
            if key in (8, 127) and points:
                points.pop()
            if key in (10, 13):
                if len(points) >= minimum:
                    return points
                message = f"Select at least {minimum} points."
    finally:
        cv2.destroyAllWindows()


def board_correspondences(image, board, manual, roi=None, *, interactive_fallback=True):
    columns = board["inner_columns"]
    rows = board["inner_rows"]
    square_mm = board["square_mm"]
    grid = np.zeros((columns * rows, 3), dtype=np.float64)
    grid[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_mm
    if not manual:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, y = 0, 0
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if roi is not None:
            x, y, width, height = roi
            if min(x, y) < 0 or min(width, height) <= 0 or x + width > gray.shape[1] or y + height > gray.shape[0]:
                raise ValueError("The checkerboard region must fit inside the reference image.")
            gray = gray[y:y + height, x:x + width]
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(
            gray, (columns, rows), flags
        )
        if found:
            return grid, corners.reshape(-1, 2) + np.array([x, y]), "automatic"
        if not interactive_fallback:
            raise ValueError("Full checkerboard detection failed. Try a board ROI or a clearer reference image.")
        print("Full grid detection failed. The coin may cover an inner corner.", file=sys.stderr)
    pixels = select_pixels(
        image, "Checkerboard reference",
        "Click the four EXTREME INNER corners: top-left, top-right, bottom-right, bottom-left. "
        "Orient the image so the long grid edge runs left to right. "
        "If a reference corner is hidden, take a second image with the coin removed.",
        4, 4,
    )
    pixels = np.asarray(pixels, dtype=np.float64)
    if not cv2.isContourConvex(pixels.astype(np.float32)):
        raise ValueError("The four reference corners must form a convex quadrilateral in order.")
    return grid[[0, columns - 1, -1, -columns]], pixels, "manual_four_corners"


def measure(image_points, board_points, rim_pixels, matrix, distortion, rim_height):
    ok, rvec, tvec = cv2.solvePnP(board_points, image_points, matrix, distortion)
    if not ok:
        raise ValueError("Could not estimate the checkerboard pose.")
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(board_points, rvec, tvec, matrix, distortion)
    residuals = projected.reshape(-1, 2) - image_points
    board_rmse = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))

    camera_in_board = -rotation.T @ tvec.ravel()
    if np.any((board_points @ rotation.T + tvec.ravel())[:, 2] <= 0):
        raise ValueError("The recovered board is behind the camera.")
    # Positive height means toward the camera, independent of corner ordering.
    rim_z = np.copysign(rim_height, camera_in_board[2])
    normalized = cv2.undistortPoints(
        rim_pixels.reshape(-1, 1, 2), matrix, distortion
    ).reshape(-1, 2)
    rays = np.column_stack((normalized, np.ones(len(normalized)))) @ rotation
    if np.any(np.abs(rays[:, 2]) < 1e-9):
        raise ValueError("A rim ray runs parallel to the checkerboard.")
    scales = (rim_z - camera_in_board[2]) / rays[:, 2]
    if np.any(scales <= 0):
        raise ValueError("The selected rim plane is behind the camera.")
    points = (camera_in_board + scales[:, None] * rays)[:, :2]

    origin = points.mean(axis=0)
    centered = points - origin
    system = np.column_stack((2 * centered, np.ones(len(centered))))
    solution, _, rank, _ = np.linalg.lstsq(system, (centered ** 2).sum(axis=1), rcond=None)
    if rank < 3:
        raise ValueError("Select points distributed around the rim.")
    center = solution[:2] + origin
    delta = points - center
    angles = np.sort(np.arctan2(delta[:, 1], delta[:, 0]))
    if np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() >= np.pi:
        raise ValueError("Select rim points around the full coin, including opposite sides.")
    radii = np.linalg.norm(delta, axis=1)
    radius = radii.mean()
    return {
        "diameter_mm": float(2 * radius),
        "rim_fit_rmse_mm": float(np.sqrt(np.mean((radii - radius) ** 2))),
        "checkerboard_reprojection_rmse_px": float(board_rmse),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Full-resolution image containing the coin.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--camera", choices=("scene", "wrist"), default="scene")
    parser.add_argument("--board-image", type=Path,
                        help="Reference image with the coin removed and camera and board in the same positions.")
    parser.add_argument("--manual-board", action="store_true", help="Click four checkerboard corners.")
    parser.add_argument("--board-roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Search only this region of the checkerboard reference image.")
    parser.add_argument("--rim-point", type=pixel, action="append",
                        help="Rim pixel as u,v. Repeat five or more times to skip rim selection.")
    parser.add_argument("--rim-height-mm", type=float, default=0,
                        help="Height of the selected rim above the printed board surface.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    args = parser.parse_args()
    try:
        calibration = json.loads(args.calibration.read_text())[args.camera]
        matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
        if not np.isfinite(args.rim_height_mm) or args.rim_height_mm < 0:
            raise ValueError("Rim height must be finite and nonnegative.")
        image = cv2.imread(str(args.image))
        reference = cv2.imread(str(args.board_image)) if args.board_image else image
        for frame in (image, reference):
            if frame is None or list(frame.shape[1::-1]) != calibration["image_size"]:
                width, height = calibration["image_size"]
                raise ValueError(f"Use readable, uncropped {args.camera}-camera images at {width}×{height}.")
        board_points, board_pixels, method = board_correspondences(
            reference, calibration["board"], args.manual_board, args.board_roi
        )
        rim = args.rim_point or select_pixels(
            image, "Coin rim", "Click 8–12 points around the outer rim, then press Enter.", 5, 32
        )
        rim = np.asarray(rim, dtype=np.float64)
        if len(np.unique(rim, axis=0)) < 5 or np.any(rim < 0) or np.any(rim >= calibration["image_size"]):
            raise ValueError("Provide at least five distinct rim pixels inside the image.")
        result = measure(board_pixels, board_points, rim, matrix, distortion, args.rim_height_mm)
        result.update({
            "camera": args.camera,
            "image": str(args.image),
            "board_image": str(args.board_image or args.image),
            "checkerboard_method": method,
            "checkerboard_points": len(board_points),
            "rim_points": len(rim),
            "rim_height_mm": args.rim_height_mm,
        })
        print(f"Estimated coin diameter: {result['diameter_mm']:.2f} mm")
        print(f"Rim fit RMSE: {result['rim_fit_rmse_mm']:.3f} mm")
        print(f"Checkerboard reprojection RMSE: {result['checkerboard_reprojection_rmse_px']:.3f} px")
        if method == "manual_four_corners":
            print("Board RMSE uses the four selected corners. Automatic grid detection offers a stronger check.")
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
