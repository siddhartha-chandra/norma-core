#!/usr/bin/env python3
"""Measure an object from a wrist stereo sweep: segment it, fit a table-aligned 3D cuboid, check the gripper.

The object is segmented in view A with GrabCut, initialised from ``--box`` (a pixel box around the object,
e.g. from a vision model or a person); a morphological opening removes thin attachments such as cables.
Dense-cloud points (``cloud.ply``, already levelled to the table) that project inside the mask and sit above
the table form the object. The cuboid's footprint is their minimum-area rectangle on the table and its height
the ``--top-percentile`` height. Saves ``object_mask.png``, ``cuboid_overlay.jpg``, ``cuboid_3d.png`` and
``object_measurement.json`` in the attempt directory.

    python scripts/measure_object.py calib_frames/stereo_earbud/sweep/attempt_1 --box 545 287 161 103
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_stereo_cloud import build, render_png, view  # noqa: E402

from desk_core.calib_store import CalibrationStore  # noqa: E402
from desk_core.station_bridge import MotorState  # noqa: E402
from desk_core.stereo_sweep import camera_pose, project, refine_relative_pose  # noqa: E402
from desk_core.wrist_stereo import DEFAULT_WRIST_INTRINSICS  # noqa: E402

# hardware/pgripper/README.md: 27 mm stroke per jaw, 54 mm total opening.
GRIPPER_MAX_OPENING_MM = 54.0
CUBOID_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


def segment(image, box, open_px, dark_v_max=0):
    """GrabCut mask of the object inside ``box`` (x, y, w, h), with thin parts opened away.

    ``dark_v_max`` > 0 (a dark object) also requires HSV value below it, with holes (glossy highlights)
    filled: GrabCut alone confuses a black object with adjacent dark checkerboard squares.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, tuple(int(v) for v in box), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    if dark_v_max > 0:
        dark = (cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2] < dark_v_max).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        fg &= dark
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(fg)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return out.astype(bool)


def object_points(points, T_base_cam, K, dist, mask, table_z, min_height_m):
    """Cloud points above the table whose projection into view A falls inside the mask."""
    above = points[:, 2] - table_z > min_height_m
    candidates = points[above]
    T_cam_base = np.linalg.inv(T_base_cam)
    in_cam = candidates @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    front = in_cam[:, 2] > 0
    pixels = cv2.projectPoints(in_cam[front], np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2).round().astype(int)
    h, w = mask.shape
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    keep = np.zeros(len(pixels), bool)
    keep[inside] = mask[pixels[inside, 1], pixels[inside, 0]]
    return candidates[front][keep]


def inside_mask(points, T_base_cam, K, dist, mask):
    T_cam_base = np.linalg.inv(T_base_cam)
    in_cam = points @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    pixels = cv2.projectPoints(in_cam, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2).round().astype(int)
    h, w = mask.shape
    ok = (in_cam[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    out = np.zeros(len(points), bool)
    out[ok] = mask[pixels[ok, 1], pixels[ok, 0]]
    return out


def carve(views, centre_xy, table_z, half_m=0.06, top_m=0.09, step_m=0.001, slack_px=4):
    """Visual hull above the table: voxels inside every view's (slightly dilated) silhouette."""
    xs = np.arange(centre_xy[0] - half_m, centre_xy[0] + half_m, step_m)
    ys = np.arange(centre_xy[1] - half_m, centre_xy[1] + half_m, step_m)
    zs = np.arange(table_z + step_m / 2, table_z + top_m, step_m)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = np.ones(len(grid), bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * slack_px + 1, 2 * slack_px + 1))
    for T, K, dist, mask in views:
        grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        keep[keep] = inside_mask(grid[keep], T, K, dist, grown)
    return grid[keep]


def extra_view(d, result, calib, K, dist, image_a, args):
    """Another wrist view (image + joint ticks), registered to view A from image matches.

    The arm-only pose of a distant view is a few degrees off; the image-refined A->view motion
    (metric scale from the arm), mapped through the sweep's table alignment, keeps silhouettes consistent.
    """
    image_path, ticks_path = args.extra_view
    ticks_a = json.loads((d / "samples.json").read_text())[0]["joint_ticks"]
    ticks_v = json.loads(Path(ticks_path).read_text())["joint_ticks"]
    T_a_arm = camera_pose(calib, motors_from_ticks(ticks_a))
    T_v_arm = camera_pose(calib, motors_from_ticks(ticks_v))
    image_v = cv2.imread(str(image_path))
    rel, info = refine_relative_pose(image_a, image_v, K, dist, np.linalg.solve(T_v_arm, T_a_arm))
    print(f"extra view registration: {info['source']} (rotation diff {info.get('rotation_diff_deg')} deg, "
          f"direction diff {info.get('direction_diff_deg')} deg)")
    rel = rel.copy()
    rel[:3, 3] *= result["table_alignment"].get("scale", 1.0)
    mask_v = segment(image_v, args.extra_box, args.open_px, args.dark_v_max)
    cv2.imwrite(str(d / "object_mask_extra.png"), mask_v.astype(np.uint8) * 255)
    return np.array(result["T_base_cam_a"]) @ np.linalg.inv(rel), K, dist, mask_v


def motors_from_ticks(ticks):
    return [MotorState(i, int(t), 0, 0, 4095) for i, t in enumerate(list(ticks) + [2048], start=1)]


def fit_cuboid(points, table_z, trim_pct, top_pct):
    """Table-aligned cuboid: footprint = min-area rectangle of the trimmed XY points, from the table up."""
    xy = points[:, :2]
    centre = np.median(xy, axis=0)
    radius = np.linalg.norm(xy - centre, axis=1)
    core = points[radius <= np.percentile(radius, 100 - trim_pct)]
    (cx, cy), (a, b), angle = cv2.minAreaRect((core[:, :2] * 1000).astype(np.float32))
    top = float(np.percentile(core[:, 2], top_pct))
    footprint = cv2.boxPoints(((cx, cy), (a, b), angle)) / 1000
    corners = np.vstack([np.column_stack([footprint, np.full(4, table_z)]), np.column_stack([footprint, np.full(4, top)])])
    length, width = max(a, b), min(a, b)
    yaw = angle if a >= b else angle + 90
    return corners, {"length_mm": round(length, 1), "width_mm": round(width, 1),
                     "height_mm": round((top - table_z) * 1000, 1), "yaw_deg": round(yaw % 180, 1),
                     "centre_base_mm": [round(cx, 1), round(cy, 1), round((table_z + top) / 2 * 1000, 1)],
                     "points_used": int(len(core))}


def draw_overlay(image, mask, corners, T_base_cam, K, dist, dims, path):
    out = image.copy()
    tint = out.copy()
    tint[mask] = (0, 200, 255)
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
    pixels = np.array([project(T_base_cam, K, dist, c) for c in corners]).round().astype(int)
    for i, j in CUBOID_EDGES:
        color = (0, 255, 0) if i >= 4 and j >= 4 else (255, 160, 0)
        cv2.line(out, tuple(pixels[i]), tuple(pixels[j]), color, 2, cv2.LINE_AA)
    label = f"{dims['length_mm']:.0f} x {dims['width_mm']:.0f} x {dims['height_mm']:.0f} mm"
    x, y = pixels[:, 0].min(), max(20, pixels[:, 1].min() - 12)
    cv2.putText(out, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="sweep attempt directory (wrist-000.jpg, cloud.ply, stereo_result.json)")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True,
                        help="pixel box around the object in view A (wrist-000.jpg)")
    parser.add_argument("--open-px", type=int, default=15, help="opening kernel that removes cables and thin parts")
    parser.add_argument("--dark-v-max", type=int, default=0,
                        help="dark object: also require HSV value below this (e.g. 45); 0 = GrabCut only")
    parser.add_argument("--min-height-mm", type=float, default=4.0, help="ignore points this close to the table")
    parser.add_argument("--trim-pct", type=float, default=3.0, help="drop the farthest XY points (stray matches)")
    parser.add_argument("--top-percentile", type=float, default=98.0)
    parser.add_argument("--height-percentile", type=float, default=90.0,
                        help="with a visual hull: stereo height percentile inside the hull footprint")
    parser.add_argument("--gripper-mm", type=float, default=GRIPPER_MAX_OPENING_MM, help="maximum jaw opening")
    parser.add_argument("--top-image", type=Path, help="scene (top) camera frame of the same scene for silhouette carving")
    parser.add_argument("--top-box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="pixel box around the object in --top-image; the cuboid then comes from the visual hull")
    parser.add_argument("--extra-view", nargs=2, metavar=("IMAGE", "TICKS_JSON"),
                        help="another wrist view (e.g. centre_0.jpg and centre_0.json from a side-on centring)")
    parser.add_argument("--extra-box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="pixel box around the object in --extra-view")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_WRIST_INTRINSICS)
    args = parser.parse_args()

    d = args.directory
    result = json.loads((d / "stereo_result.json").read_text())
    wrist = json.loads(args.intrinsics.read_text())["wrist"]
    K, dist = np.array(wrist["camera_matrix"]), np.array(wrist["distortion_coefficients"])
    T_a, table_z = np.array(result["T_base_cam_a"]), result["table_z_m"]
    image = cv2.imread(str(d / "wrist-000.jpg"))

    mask = segment(image, args.box, args.open_px, args.dark_v_max)
    cv2.imwrite(str(d / "object_mask.png"), mask.astype(np.uint8) * 255)
    cloud = o3d.io.read_point_cloud(str(d / "cloud.ply"))
    points = object_points(np.asarray(cloud.points), T_a, K, dist, mask, table_z, args.min_height_mm / 1000)
    if len(points) < 50:
        raise SystemExit(f"only {len(points)} object points above the table inside the mask")
    corners, dims = fit_cuboid(points, table_z, args.trim_pct, args.top_percentile)
    dims["source"] = "stereo"

    if args.top_box is not None:
        cams = json.loads(args.intrinsics.read_text())
        calib = CalibrationStore()
        top = cams.get("scene") or {}
        K_top = np.array(top.get("camera_matrix", calib.data["K_top"]))
        dist_top = np.array(top.get("distortion_coefficients", calib.data["dist_top"]))
        top_image = cv2.imread(str(args.top_image))
        top_mask = segment(top_image, args.top_box, max(5, args.open_px // 2), args.dark_v_max)
        cv2.imwrite(str(d / "object_mask_top.png"), top_mask.astype(np.uint8) * 255)
        views = [(T_a, K, dist, mask), (np.array(calib.T_world_cam("top")), K_top, dist_top, top_mask)]
        if args.extra_view is not None:
            views.append(extra_view(d, result, calib, K, dist, image, args))
        hull = carve(views, np.median(points[:, :2], axis=0), table_z)
        if len(hull) < 100:
            raise SystemExit(f"visual hull has only {len(hull)} voxels; check the boxes and the top camera calibration")
        stereo_dims = dims
        corners, dims = fit_cuboid(hull, table_z, 0.5, 99.5)
        cameras = "wrist A + top camera" + (" + extra wrist view" if args.extra_view is not None else "")
        dims["source"] = f"visual hull ({cameras})"
        dims["stereo_estimate"] = {k: stereo_dims[k] for k in ("length_mm", "width_mm", "height_mm")}
        dims["hull_height_upper_bound_mm"] = dims["height_mm"]
        # Steep views only bound the hull from above; the stereo top surface inside the footprint is tighter.
        occupied = {tuple(c) for c in np.round(hull[:, :2] * 1000).astype(int)}
        inside = np.array([tuple(c) in occupied for c in np.round(points[:, :2] * 1000).astype(int)])
        if inside.sum() >= 50:
            top = min(corners[4:, 2].max(), float(np.percentile(points[inside, 2], args.height_percentile)))
            corners[4:, 2] = top
            dims["height_mm"] = round((top - table_z) * 1000, 1)
            dims["centre_base_mm"][2] = round((table_z + top) / 2 * 1000, 1)
            dims["source"] += f", height from stereo p{args.height_percentile:g} ({int(inside.sum())} points)"
        points = hull[hull[:, 2] <= corners[4, 2]]

    sparse = np.array(result.get("points_base_m") or []).reshape(-1, 3)
    if len(sparse):
        pix = np.array(result["pixels_a"]).round().astype(int)
        on_object = mask[np.clip(pix[:, 1], 0, mask.shape[0] - 1), np.clip(pix[:, 0], 0, mask.shape[1] - 1)]
        heights = (sparse[on_object, 2] - table_z) * 1000
        dims["sparse_matches_on_object"] = int(on_object.sum())
        if on_object.any():
            dims["sparse_height_mm"] = {"median": round(float(np.median(heights)), 1),
                                        "max": round(float(heights.max()), 1)}
    grasp = dims["width_mm"]
    dims["gripper_max_opening_mm"] = args.gripper_mm
    dims["fits_gripper"] = bool(grasp < args.gripper_mm)
    dims["gripper_margin_mm"] = round(args.gripper_mm - grasp, 1)

    draw_overlay(image, mask, corners, T_a, K, dist, dims, d / "cuboid_overlay.jpg")
    geometries, _ = build(d, "height", 0.0015)
    box = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(CUBOID_EDGES))
    box.colors = o3d.utility.Vector3dVector([[0.1, 1.0, 0.2]] * len(CUBOID_EDGES))
    obj = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    obj.paint_uniform_color([1.0, 0.55, 0.0])
    geometries += [box, obj]
    centre = corners.mean(axis=0)
    _, eye, up = view({"T_base_cam_a": result["T_base_cam_a"], "points_base_m": [centre.tolist()]}, geometries,
                      elevation_deg=40.0, distance_m=0.25)
    render_png(geometries, centre, eye, up, d / "cuboid_3d.png")
    (d / "object_measurement.json").write_text(json.dumps({**dims, "corners_base_m": corners.round(5).tolist()},
                                                          indent=2) + "\n")
    print(json.dumps(dims, indent=2))


if __name__ == "__main__":
    main()
