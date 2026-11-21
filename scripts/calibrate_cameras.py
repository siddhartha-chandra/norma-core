#!/usr/bin/env python3
"""Capture checkerboard images from V4L2 and fit per-camera OpenCV intrinsics."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import tempfile

import cv2
import numpy as np


WINDOW = "Camera calibration"
MIN_VIEWS = 25


def write_json(path, data):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    Path(stream.name).replace(path)


def station_running():
    for path in Path("/proc").glob("[0-9]*/comm"):
        try:
            if path.read_text().strip() == "station":
                return True
        except OSError:
            continue
    return False


def list_cameras():
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        print(f"/dev/{node.name}  {(node / 'name').read_text().strip()}")
    for alias in sorted(Path("/dev/v4l/by-path").glob("*video-index0")):
        print(f"{alias} -> {alias.resolve()}")


def coverage(corners, size):
    points = corners.reshape(-1, 2) / np.asarray(size)
    center = points.mean(axis=0)
    area = cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
    return center.tolist(), area


def capture(args):
    if station_running():
        raise ValueError("Station is running. Stop it before opening the cameras.")
    cameras = dict(args.camera)
    if len(cameras) != len(args.camera) or len(cameras) > 2:
        raise ValueError("Supply one or two distinctly named cameras.")
    if len({Path(device).resolve() for device in cameras.values()}) != len(cameras):
        raise ValueError("Each camera needs a different device.")
    if args.inner_columns < 3 or args.inner_rows < 3 or not math.isfinite(args.square_mm) or args.square_mm <= 0:
        raise ValueError("Use at least 3 inner corners per axis and a positive square size.")
    if min(args.width, args.height, args.fps) <= 0:
        raise ValueError("Width, height, and FPS must be positive.")
    args.session.mkdir(parents=True, exist_ok=True)
    manifest_path = args.session / "session.json"
    board = {
        "inner_columns": args.inner_columns,
        "inner_rows": args.inner_rows,
        "square_mm": args.square_mm,
    }
    manifest = {"board": board, "cameras": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["board"] != board:
            raise ValueError("Board settings differ from the existing session.")
    for name, device in cameras.items():
        existing = manifest["cameras"].setdefault(name, {"views": []})
        if existing["views"] and Path(existing["device"]).resolve() != Path(device).resolve():
            raise ValueError(f"{name} was captured from a different device. Use a new session.")
        existing["device"] = device
        (args.session / name).mkdir(exist_ok=True)
    write_json(manifest_path, manifest)

    def open_camera(name):
        source = cv2.VideoCapture(cameras[name], cv2.CAP_V4L2)
        if not source.isOpened():
            source.release()
            raise ValueError(f"Cannot open {name} at {cameras[name]}. Close Cheese or other camera apps and check the device path.")
        source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        source.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        source.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        source.set(cv2.CAP_PROP_FPS, args.fps)
        return source

    names = list(cameras)
    name = names[0]
    source = open_camera(name)
    message = ""
    pattern = (args.inner_columns, args.inner_rows)
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = source.read()
            if not ok:
                raise ValueError(f"Camera {name} stopped delivering frames. Saved captures can be resumed.")
            stamp = datetime.now(timezone.utc).isoformat()
            size = [frame.shape[1], frame.shape[0]]
            if size != [args.width, args.height]:
                raise ValueError(f"{name} delivered {size[0]}x{size[1]}. Set --width and --height to the intended capture mode.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
            views = manifest["cameras"][name]["views"]
            warning = ""
            if found:
                center, area = coverage(corners, size)
                margin = cv2.boundingRect(corners)
                x, y, width, height = margin
                board_gray = gray[y:y + height, x:x + width]
                if cv2.Laplacian(board_gray, cv2.CV_64F).var() < 50:
                    warning = "Possible blur. Hold still."
                elif any(
                    view["image_size"] == size and
                    np.mean(np.linalg.norm(
                        corners.reshape(-1, 2) - np.asarray(view["corners"]), axis=1
                    )) < 0.025 * min(size)
                    for view in views
                ):
                    warning = "Similar view. Move or tilt the board."
            else:
                warning = "Show the full inner grid."

            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, pattern, corners, found)
            instruction = "Center and vary distance" if len(views) < 5 else (
                "Visit all four edges" if len(views) < 13 else
                "Visit all four corners" if len(views) < 21 else
                "Tilt in different directions"
            )
            lines = [
                f"{name} | {size[0]}x{size[1]} | {len(views)}/{MIN_VIEWS} saved",
                f"{instruction} | {warning or 'Board detected'}",
                "1/2 camera   Space capture   Backspace undo   Q quit",
                message,
            ]
            for index, line in enumerate(lines):
                point = (10, 24 + 24 * index)
                cv2.putText(preview, line, point, cv2.FONT_HERSHEY_SIMPLEX, .56, (0, 0, 0), 4)
                cv2.putText(preview, line, point, cv2.FONT_HERSHEY_SIMPLEX, .56, (255, 255, 255), 1)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 255
            # OpenCV's GTK backend returns -1 for WND_PROP_VISIBLE even while open.
            try:
                window_closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0
            except cv2.error:
                window_closed = True
            if key in (ord("q"), ord("Q"), 27) or window_closed:
                break
            if key in (ord("1"), ord("2")):
                index = key - ord("1")
                if index < len(names) and names[index] != name:
                    source.release()
                    name = names[index]
                    source = open_camera(name)
                    message = ""
            elif key == ord(" "):
                if not found:
                    message = "Skipped. No complete checkerboard."
                elif views and views[0]["image_size"] != size:
                    message = "Skipped. Resolution changed. Restore it or start a new session."
                else:
                    filename = f"{name}/{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.png"
                    if not cv2.imwrite(str(args.session / filename), frame):
                        raise OSError(f"Could not save {filename}")
                    views.append({
                        "file": filename, "timestamp": stamp, "image_size": size,
                        "corners": corners.reshape(-1, 2).tolist(), "center": center,
                        "area_fraction": area,
                    })
                    write_json(manifest_path, manifest)
                    message = f"Saved view {len(views)}"
            elif key in (8, 127) and views:
                removed = views.pop()
                write_json(manifest_path, manifest)
                (args.session / removed["file"]).unlink(missing_ok=True)
                message = "Last view removed"
    finally:
        source.release()
        cv2.destroyAllWindows()


def solve(args):
    manifest = json.loads((args.session / "session.json").read_text())
    board = manifest["board"]
    columns, rows = board["inner_columns"], board["inner_rows"]
    grid = np.zeros((columns * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * board["square_mm"]
    names = args.name or list(manifest["cameras"])
    for name in names:
        if name not in manifest["cameras"]:
            raise ValueError(f"No camera named {name}.")
        count = len(manifest["cameras"][name]["views"])
        if count < MIN_VIEWS:
            raise ValueError(f"{name} has {count} views. Capture at least {MIN_VIEWS}.")
    for name in names:
        camera = manifest["cameras"][name]
        views = camera["views"]
        size = tuple(views[0]["image_size"])
        if any(tuple(view["image_size"]) != size for view in views):
            raise ValueError(f"{name} has mixed resolutions.")
        points = [np.asarray(view["corners"], np.float32).reshape(-1, 1, 2) for view in views]
        rms, matrix, distortion, *_, errors = cv2.calibrateCameraExtended(
            [grid] * len(views), points, size, None, None
        )
        per_view = [
            {"file": view["file"], "rms_px": float(error)}
            for view, error in zip(views, errors.ravel())
        ]
        per_view.sort(key=lambda view: view["rms_px"], reverse=True)
        observed = np.concatenate(points).reshape(-1, 2) / np.asarray(size)
        edge_coverage = {
            "left": bool(observed[:, 0].min() <= .15),
            "right": bool(observed[:, 0].max() >= .85),
            "top": bool(observed[:, 1].min() <= .15),
            "bottom": bool(observed[:, 1].max() >= .85),
        }
        output = args.session / name
        write_json(output / "intrinsics.json", {
            "camera": name, "device": camera["device"], "image_size": list(size),
            "board": board, "camera_matrix": matrix.tolist(),
            "distortion_coefficients": distortion.ravel().tolist(),
        })
        write_json(output / "report.json", {
            "views": len(views), "rms_px": float(rms), "per_view": per_view,
            "edge_coverage": edge_coverage,
        })
        for index in sorted({0, len(views) // 2, len(views) - 1}):
            frame = cv2.imread(str(args.session / views[index]["file"]))
            if frame is None:
                raise OSError(f"Could not read {views[index]['file']}")
            preview = cv2.undistort(frame, matrix, distortion)
            if not cv2.imwrite(str(output / f"undistorted_{index + 1:03d}.png"), preview):
                raise OSError("Could not save an undistorted preview.")
        missing = [edge for edge, covered in edge_coverage.items() if not covered]
        print(f"{name}: {rms:.3f} px RMS across {len(views)} views. Worst view: {per_view[0]['file']} ({per_view[0]['rms_px']:.3f} px).")
        if missing:
            print(f"Add views near these edges: {', '.join(missing)}")


def camera_spec(value):
    name, sep, device = value.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or not device.startswith("/dev/"):
        raise argparse.ArgumentTypeError("Use NAME=/dev/videoN or a /dev/v4l alias.")
    return name, device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List Linux camera devices without opening them.")
    grab = sub.add_parser("capture", help="Capture or resume a checkerboard session.")
    grab.add_argument("session", type=Path)
    grab.add_argument("--camera", action="append", required=True, type=camera_spec)
    grab.add_argument("--inner-columns", type=int, default=10)
    grab.add_argument("--inner-rows", type=int, default=7)
    grab.add_argument("--square-mm", type=float, default=15)
    grab.add_argument("--width", type=int, default=1280)
    grab.add_argument("--height", type=int, default=720)
    grab.add_argument("--fps", type=int, default=30)
    fit = sub.add_parser("solve", help="Fit intrinsics from the saved session.")
    fit.add_argument("session", type=Path)
    fit.add_argument("--name", action="append", help="Fit only this camera. Repeat for multiple.")
    args = parser.parse_args()
    try:
        if args.action == "list":
            list_cameras()
        elif args.action == "capture":
            capture(args)
        else:
            solve(args)
    except (ValueError, OSError, cv2.error) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
