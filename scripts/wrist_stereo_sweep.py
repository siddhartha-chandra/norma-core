#!/usr/bin/env python3
"""Automatic, kinematics-aware wrist-camera stereo sweep with fallbacks and colored point clouds.

Each attempt: plan the slide (direction and distance from the camera pose, the target depth and the
joint swings), capture view A, slide the flange with fixed orientation, capture view B, then
  1. refine the A->B camera motion from whole-image matches (encoders give the metric scale),
  2. run the epipolar matcher (desk_core.wrist_stereo) on the target ROI,
  3. level and scale the result so the fitted table plane is horizontal at the arm-derived camera height,
  4. build a dense colored cloud (rectified SGBM) and save cloud.ply / sparse.ply / stereo_result.json.
If the plan is refused or the stereo result fails the quality checks, the next attempt plans a shorter
slide from the new pose, with the target re-projected into the new view.

When the object location is known (--object in any robot frame, or a top/wrist camera pixel), the arm
first moves to a reachable view that puts the object at the wrist image centre and leaves room for a
good slide, then checks the image (optionally detecting the object) and re-centres before the sweep.

  --look                                   save the current wrist frame (pick the target pixel)
  --target U V                             dry run: show the chosen slide and the fallbacks
  --target U V --execute                   run the sweep
  --object X Y Z [--frame LINK]            dry run: show the centring view pose, its move and slide
  --object X Y Z --detector color --execute   centre the object, then run the sweep
  --offline DIR --roi X Y W H              re-evaluate a saved attempt without the robot

Render the cloud with:  python scripts/plot_stereo_cloud.py <attempt dir> [--show]
"""
import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desk_core.calib_store import CAMERA_ONLY_KEYS, CalibrationStore, _normalize_calib_dict
from desk_core.motion import MotionLimits, correct_final_pose, flange_error, stream_joint_path
from desk_core.station_bridge import LiveStationBridge
from desk_core.stereo_sweep import (align_pose, apply_alignment, camera_pose, dense_cloud, detect_object,
                                    fit_table_plane, object_in_base, path_clearance, pixel_ray, plan_slides,
                                    plan_view_pose, view_grid,
                                    project, rank_slides, refine_relative_pose, table_alignment, table_point,
                                    write_ply)
from desk_core.wrist_stereo import DEFAULT_WRIST_INTRINSICS, estimate_wrist_depth, load_wrist_pair

REPO = Path(__file__).resolve().parents[1]
STILL_TICKS = 3
STILL_S = 0.7


# --- hardware ---------------------------------------------------------------------------------------------------

async def connect(calib, host, port):
    hw = calib.hardware_config()
    bridge = LiveStationBridge(host=host, port=port, target_bus=hw["bus_serial"], cam_top=hw["camera_top"],
                               cam_wrist=hw["camera_wrist"], follow_motor_rx=True)
    await bridge.connect()
    for _ in range(50):
        keys = [k for k in await bridge.list_cameras() if hw["camera_wrist"] in k]
        if keys:
            break
        await asyncio.sleep(0.1)
    if len(keys) != 1:
        raise SystemExit(f"wrist camera {hw['camera_wrist']} not found in {await bridge.list_cameras()}")
    await asyncio.sleep(1.0)
    return bridge, keys[0], hw["bus_serial"]


async def settled_capture(bridge, camera_key, timeout_s=10.0):
    """Wait until the arm joints hold still, then return a wrist frame received after that and the ticks."""
    deadline = time.monotonic() + timeout_s
    baseline, still_since = None, time.monotonic()
    while time.monotonic() < deadline:
        jpeg = await bridge.get_frame("wrist")
        ticks = np.array([m.position for m in (await bridge.get_state()).motors if 1 <= m.id <= 7])
        now = time.monotonic()
        if baseline is None or np.max(np.abs(ticks - baseline)) > STILL_TICKS:
            baseline, still_since = ticks, now
        elif now - still_since >= STILL_S and bridge.frame_received_at.get(camera_key, 0) > still_since:
            return jpeg, ticks.tolist()
        await asyncio.sleep(0.05)
    raise SystemExit("arm did not settle (or no fresh wrist frame) within the timeout")


def save_capture(out, samples, jpeg, ticks, camera_key, bus):
    name = f"wrist-{len(samples):03d}.jpg"
    (out / name).write_bytes(jpeg)
    samples.append({"image": name, "joint_ticks": [int(t) for t in ticks], "camera": camera_key, "bus": bus})
    (out / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")


# --- evaluation -------------------------------------------------------------------------------------------------

def draw_matches(frame_a, frame_b, result, roi, path):
    x, y, w, h = roi
    a, b = frame_a.copy(), frame_b.copy()
    cv2.rectangle(a, (x, y), (x + w, y + h), (0, 200, 255), 2)
    canvas = np.hstack([a, b])
    for pa, pb in zip(result.pixels_a.astype(int), result.pixels_b.astype(int)):
        cv2.circle(canvas, tuple(pa), 4, (0, 255, 0), 1)
        cv2.circle(canvas, (int(pb[0]) + a.shape[1], int(pb[1])), 4, (0, 255, 0), 1)
        cv2.line(canvas, tuple(pa), (int(pb[0]) + a.shape[1], int(pb[1])), (0, 255, 0), 1)
    cv2.imwrite(str(path), canvas)


def evaluate(directory, roi, args):
    """Refine the pose, triangulate the ROI, rescale to the table, build the dense cloud and save everything."""
    pair = load_wrist_pair(directory / "samples.json", args.rig, args.intrinsics, (0, 1))
    K, dist, Ta = pair.camera_matrix, pair.distortion, pair.T_base_cam_a
    rel_arm = np.linalg.solve(pair.T_base_cam_b, Ta)
    rel, refine = refine_relative_pose(pair.frame_a, pair.frame_b, K, dist, rel_arm)

    Tb = Ta @ np.linalg.inv(rel)
    points, colors, dense = dense_cloud(pair.frame_a, pair.frame_b, K, dist, Ta, rel)
    result = estimate_wrist_depth(pair.frame_a, pair.frame_b, K, dist, Ta, Tb, roi)
    sparse = result.points_base_m

    scale, R_fix, align = 1.0, np.eye(3), {"reason": "disabled"}
    if args.table_align and pair.table_z_m is not None and len(points) > 1000:
        plane = fit_table_plane(points)
        align = {"reason": "no table plane found"}
        if plane is not None:
            s, R, align = table_alignment(plane[0], plane[1], Ta[:3, 3], pair.table_z_m)
            align["plane_points"] = int(plane[2].sum())
            if 0.8 < s < 1.25 and align["tilt_corrected_deg"] < 15:
                scale, R_fix = s, R
            else:
                align["reason"] = f"rejected implausible correction (scale {s:.3f})"
    align["scale"] = round(scale, 4)
    centre = Ta[:3, 3]
    points = apply_alignment(points, centre, scale, R_fix)
    sparse = apply_alignment(sparse, centre, scale, R_fix)
    Ta_plot, Tb_plot = align_pose(Ta, centre, scale, R_fix), align_pose(Tb, centre, scale, R_fix)

    depth = result.depth_a_m * scale
    summary = {"roi": list(roi), "pose_source": refine["source"], "refine": refine, "table_alignment": align,
               "dense": dense, "baseline_mm": round(result.baseline_m * scale * 1000, 1),
               "match_count": int(len(depth)),
               "median_depth_mm": round(float(np.median(depth)) * 1000, 1) if depth.size else None}
    if depth.size:
        heights = (sparse[:, 2] - (pair.table_z_m or 0.0)) * 1000
        summary.update({
            "median_point_base_mm": (np.median(sparse, axis=0) * 1000).round(1).tolist(),
            "height_above_table_mm": {"median": round(float(np.median(heights)), 1),
                                      "p90": round(float(np.percentile(heights, 90)), 1),
                                      "max": round(float(heights.max()), 1)},
            "median_reprojection_px": round(float(np.median(result.reprojection_error_px)), 2),
            "median_parallax_deg": round(float(np.median(result.parallax_deg)), 1),
        })
    problems = []
    if summary["match_count"] < args.min_matches:
        problems.append(f"{summary['match_count']} matches < {args.min_matches}")
    elif summary["median_reprojection_px"] > 1.5:
        problems.append(f"median reprojection {summary['median_reprojection_px']} px > 1.5")
    elif summary["median_parallax_deg"] < 1.5:
        problems.append(f"parallax {summary['median_parallax_deg']} deg < 1.5")
    summary["ok"], summary["problems"] = not problems, problems

    draw_matches(pair.frame_a, pair.frame_b, result, roi, directory / "matches.jpg")
    if len(points):
        write_ply(directory / "cloud.ply", points, colors)
    if depth.size:
        write_ply(directory / "sparse.ply", sparse, np.tile([1.0, 0.0, 1.0], (len(depth), 1)))
    (directory / "stereo_result.json").write_text(json.dumps({
        **summary, "table_z_m": pair.table_z_m, "T_base_cam_a": Ta_plot.tolist(), "T_base_cam_b": Tb_plot.tolist(),
        "points_base_m": sparse.tolist(), "depth_a_m": depth.tolist(), "pixels_a": result.pixels_a.tolist(),
        "pixels_b": result.pixels_b.tolist(), "reprojection_error_px": result.reprojection_error_px.tolist(),
    }, indent=2, default=float) + "\n")
    return summary, result


# --- planning ---------------------------------------------------------------------------------------------------

def square_roi(centre, half, size):
    w, h = size
    x, y = int(np.clip(centre[0] - half, 0, w - 2 * half)), int(np.clip(centre[1] - half, 0, h - 2 * half))
    return x, y, 2 * half, 2 * half


def choose_slide(calib, motors, wrist, target_px, roi, distance_cap_m, args, limits, target_base=None):
    K, dist = np.array(wrist["camera_matrix"]), np.array(wrist["distortion_coefficients"])
    T_cam = camera_pose(calib, motors)
    if target_base is None:
        target_base, depth = table_point(T_cam, K, dist, target_px, calib.z_table())
        source = "table plane"
    else:
        depth = float((np.linalg.inv(T_cam) @ np.append(target_base, 1.0))[2])
        source = "object location"
    distance = min(distance_cap_m, max(args.min_slide_cm / 100, depth * math.tan(math.radians(args.parallax_deg))))
    candidates = plan_slides(calib, motors, T_cam, roi, wrist["image_size"], depth, K[0, 0], distance,
                             min_distance_m=args.min_slide_cm / 100, limits=limits)
    ranked = rank_slides(candidates, args.max_joint_deg)
    if args.obstacle is not None:
        clear = [c for c in ranked if path_clearance(calib, c.plan["path"], args.obstacle) >= args.clearance_cm / 100]
        if len(clear) < len(ranked):
            print(f"  {len(ranked) - len(clear)} slide(s) dropped: closer than {args.clearance_cm:g} cm to the object")
        ranked = clear
    print(f"target depth guess {depth * 1000:.0f} mm ({source}), wanted slide {distance * 100:.1f} cm "
          f"for {args.parallax_deg:g} deg parallax")
    for c in candidates:
        mark = "*" if ranked and c is ranked[0] else " "
        print(f"  {mark} {c.describe()}")
    return ranked, target_base


# --- centring the object ----------------------------------------------------------------------------------------

def resolve_object(args, calib, motors):
    """The object of interest in base_link, from --object (any frame) or a top/wrist camera pixel."""
    if args.object is not None:
        return object_in_base(calib, motors, args.object, args.frame)
    z = calib.z_table() + args.object_height_cm / 100
    if args.object_top_pixel is not None:
        return np.array(calib.pixel_to_world(*args.object_top_pixel, z_m=z, camera="top"))
    arm = [m for m in motors if 1 <= m.id <= 7]
    return np.array(calib.pixel_to_world(*args.object_wrist_pixel, z_m=z, camera="wrist", q_rad=calib.ticks_to_rad(arm)))


def move_duration(calib, path, limits):
    """Minimum-jerk duration honouring the joint and the flange (Cartesian) speed limits."""
    travel = float(np.max(np.abs(path[-1] - path[0])))
    flange = np.array([calib.fk_ee_pose(q)[:3, 3] for q in path])
    length_cm = float(np.sum(np.linalg.norm(np.diff(flange, axis=0), axis=1))) * 100
    return max(limits.min_duration_s, 1.875 * math.degrees(travel) / limits.joint_speed_deg_s,
               1.875 * length_cm / limits.speed_cm_s)


def print_views(views):
    for i, v in enumerate(views[:4]):
        print(f"  {'*' if i == 0 else ' '} {v.describe()}")


async def centre_object(bridge, calib, camera_key, wrist, target, args, limits):
    """Move so the object sits at the wrist image centre, verify in the image and re-centre.

    Returns ``(target_base, target_px, roi)`` for the sweep, ``"dry run"``, or None (no reachable view).
    """
    K, dist, size = np.array(wrist["camera_matrix"]), np.array(wrist["distortion_coefficients"]), wrist["image_size"]
    centre = np.array(size, dtype=float) / 2
    print(f"object of interest in base_link: {(target * 1000).round(1).tolist()} mm; planning the centring view...")
    plan = dict(K=K, dist=dist, image_size=size, limits=limits, max_joint_deg=args.max_joint_deg,
                slide_parallax_deg=args.parallax_deg, min_tip_clearance_m=args.tip_clearance_cm / 100,
                obstacle=args.obstacle, obstacle_clearance_m=args.clearance_cm / 100)
    if args.obstacle is not None:
        o = args.obstacle
        print(f"keeping {args.clearance_cm:g} cm from the object: upright cylinder r {o['radius'] * 100:.1f} cm, "
              f"z {o['z_low'] * 100:.1f}..{o['z_high'] * 100:.1f} cm")
    # Planning takes seconds; a worker thread keeps the event loop consuming Station telemetry meanwhile.
    motors = (await bridge.get_state()).motors
    if args.min_tilt_deg > 0:
        grid = [p for p in view_grid(camera_pose(calib, motors), target, tilts_deg=(args.min_tilt_deg, args.min_tilt_deg + 7, 60, 70))
                if p["tilt_deg"] >= args.min_tilt_deg - 1e-6]
        plan["grid"] = grid
    views = [v for v in await asyncio.to_thread(plan_view_pose, calib, motors, target, **plan)
             if v.slide or args.no_sweep]
    if not views:
        print("no reachable view centres the object with room for a slide; move the arm closer and retry")
        return None
    print("centring view candidates (best first):")
    print_views(views)
    view = views[0]
    if not args.execute:
        print(f"dry run: would move to the * view in {move_duration(calib, view.path, limits):.1f} s, re-centre, then sweep")
        return "dry run"

    detected, bbox = None, None
    for step in range(args.centre_iters + 1):
        run_result = await stream_joint_path(bridge, calib, view.path, move_duration(calib, view.path, limits), limits)
        if not run_result["ok"]:
            raise SystemExit(f"centring move aborted (tracking lag): {json.dumps(run_result)}")
        await correct_final_pose(bridge, calib, view.path, limits)
        jpeg, ticks = await settled_capture(bridge, camera_key)
        (args.out / f"centre_{step}.json").write_text(json.dumps({"joint_ticks": [int(t) for t in ticks]}) + "\n")
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        motors = (await bridge.get_state()).motors
        T_cam = camera_pose(calib, motors)
        predicted = project(T_cam, K, dist, target)
        found = detect_object(image, predicted, args.detector, search_px=args.search_px)
        detected, bbox = (found if found else (None, None))
        seen = detected if detected is not None else predicted
        error = float(np.linalg.norm(seen - centre))
        marked = image.copy()
        cv2.drawMarker(marked, tuple(int(v) for v in centre), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.drawMarker(marked, tuple(int(v) for v in predicted), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(args.out / f"centre_{step}.jpg"), marked)
        print(f"centring step {step}: object at {np.round(seen).tolist()} "
              f"({'detected' if detected is not None else 'kinematic projection'}), {error:.0f} px from centre")
        if error <= args.centre_tol_px:
            break
        if detected is not None:
            c = T_cam[:3, 3]
            target = c + np.linalg.norm(target - c) * (T_cam[:3, :3] @ pixel_ray(K, dist, detected))
            print(f"  object re-estimated along the observed ray: {(target * 1000).round(1).tolist()} mm")
            if args.obstacle is not None:
                args.obstacle["axis_xy"] = target[:2].copy()
        if step == args.centre_iters:
            print(f"  still {error:.0f} px off after {step} re-centring moves; sweeping from here")
            break
        redo = await asyncio.to_thread(plan_view_pose, calib, motors, target, fixed=view.params, **plan)
        redo = redo or [v for v in await asyncio.to_thread(plan_view_pose, calib, motors, target, **plan) if v.slide]
        if not redo:
            print("  no reachable re-centring view; sweeping from here")
            break
        view = redo[0]

    target_px = detected if detected is not None else project(camera_pose(calib, motors), K, dist, target)
    if bbox is not None:
        x, y, w, h = bbox
        half = int(max(args.half_px, min(max(w, h) * 0.6, 200)))
    else:
        half = args.half_px
    return target, target_px, square_roi(target_px, half, size)


async def run(args) -> int:
    calib = CalibrationStore(args.rig)
    # An explicit rig path skips the camera-intrinsics overlay; pixel_to_world needs the same intrinsics as the sweep.
    cameras = _normalize_calib_dict(json.loads(args.intrinsics.read_text(encoding="utf-8")))
    for key in CAMERA_ONLY_KEYS:
        if key in cameras:
            calib.data.setdefault(key, cameras[key])
    wrist = json.loads(args.intrinsics.read_text(encoding="utf-8"))["wrist"]
    bridge, camera_key, bus = await connect(calib, args.host, args.port)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.look:
        jpeg, ticks = await settled_capture(bridge, camera_key)
        (args.out / "look.jpg").write_bytes(jpeg)
        print(f"saved {args.out / 'look.jpg'} at ticks {ticks}")
        return 0

    limits = MotionLimits()
    K, dist = np.array(wrist["camera_matrix"]), np.array(wrist["distortion_coefficients"])
    target_base, first_roi = None, tuple(args.roi) if args.roi else None
    if args.target is not None:
        target_px = np.array(args.target, dtype=float)
    else:
        target = resolve_object(args, calib, (await bridge.get_state()).motors)
        if args.object_size_cm is not None:
            diameter, height = (v / 100 for v in args.object_size_cm)
            args.obstacle = dict(axis_xy=target[:2].copy(), z_low=calib.z_table(),
                                 z_high=calib.z_table() + height, radius=diameter / 2)
        centred = await centre_object(bridge, calib, camera_key, wrist, target, args, limits)
        if centred is None or centred == "dry run":
            return 1 if centred is None else 0
        target_base, target_px, first_roi = centred
        if args.no_sweep:
            print(f"object centred at {np.round(target_px).tolist()}; --no-sweep given, stopping")
            return 0
    cap = args.max_slide_cm / 100
    for attempt in range(1, args.attempts + 1):
        motors = (await bridge.get_state()).motors
        if target_base is not None and attempt > 1:
            target_px = project(camera_pose(calib, motors), K, dist, target_base)
        roi = first_roi if first_roi and attempt == 1 else square_roi(target_px, args.half_px, wrist["image_size"])
        print(f"\n=== attempt {attempt}: target pixel {target_px.round(0).tolist()}, ROI {list(roi)}")
        ranked, guess = choose_slide(calib, motors, wrist, target_px, roi, cap, args, limits, target_base)
        target_base = guess if target_base is None else target_base
        if not ranked:
            print(f"no slide under {args.max_joint_deg:g} deg joint swing; shrinking")
            cap *= 0.6
            continue
        slide = ranked[0]
        if not args.execute:
            print(f"dry run: would slide {slide.distance_m * 100:.1f} cm along image angle {slide.image_angle_deg:+.0f} deg "
                  f"(base {np.round(slide.direction_base, 2).tolist()}); fallbacks: "
                  + ", ".join(f"{c.distance_m * 100:.1f} cm @ {c.image_angle_deg:+.0f}" for c in ranked[1:4]))
            return 0

        directory = args.out / f"attempt_{attempt}"
        directory.mkdir(parents=True, exist_ok=True)
        samples = []
        jpeg, ticks = await settled_capture(bridge, camera_key)
        save_capture(directory, samples, jpeg, ticks, camera_key, bus)
        run_result = await stream_joint_path(bridge, calib, slide.plan["path"], slide.plan["duration_s"], limits)
        if run_result.get("aborted"):
            print("move aborted (tracking lag): stopping, no automatic retry", json.dumps(run_result))
            return 2
        await correct_final_pose(bridge, calib, slide.plan["path"], limits)
        moved = flange_error(calib, slide.plan["path"], calib.ticks_to_rad((await bridge.get_state()).motors))
        print(f"moved {moved['flange_moved_cm']} cm, {moved['off_target_mm']} mm / {moved['rotation_off_target_deg']} deg off")
        jpeg, ticks = await settled_capture(bridge, camera_key)
        save_capture(directory, samples, jpeg, ticks, camera_key, bus)

        summary, result = evaluate(directory, roi, args)
        print(json.dumps({k: summary[k] for k in summary if k != "dense"}, indent=2, default=float))
        if summary["ok"]:
            print(f"\nOK on attempt {attempt}. Render: python scripts/plot_stereo_cloud.py {directory}")
            return 0
        print("quality check failed:", "; ".join(summary["problems"]))
        if result.depth_a_m.size >= args.min_matches:
            target_base = np.median(result.points_base_m, axis=0)
        cap = slide.distance_m * 0.6
    print("all attempts failed")
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=float, nargs=2, metavar=("U", "V"), help="target pixel in the current view")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="ROI (default: square at target)")
    parser.add_argument("--half-px", type=int, default=80, help="half size of the square ROI around the target")
    parser.add_argument("--object", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="object of interest in metres, in --frame; centred in the wrist image before the sweep")
    parser.add_argument("--frame", default="base_link",
                        help="frame of --object: base_link, any URDF link, wrist_camera or top_camera")
    parser.add_argument("--object-top-pixel", type=float, nargs=2, metavar=("U", "V"),
                        help="object pixel in the top camera (at --object-height-cm above the table)")
    parser.add_argument("--object-wrist-pixel", type=float, nargs=2, metavar=("U", "V"),
                        help="object pixel in the current wrist view (at --object-height-cm above the table)")
    parser.add_argument("--object-height-cm", type=float, default=0.0, help="height of the pixel's point above the table")
    parser.add_argument("--detector", choices=("none", "color", "round"), default="none",
                        help="find the object in the wrist image to close the centring loop (none: kinematics only)")
    parser.add_argument("--search-px", type=int, default=160, help="detector search radius around the predicted pixel")
    parser.add_argument("--centre-tol-px", type=float, default=30.0, help="accepted distance from the image centre")
    parser.add_argument("--centre-iters", type=int, default=2, help="re-centring moves after the first one")
    parser.add_argument("--tip-clearance-cm", type=float, default=6.0, help="minimum jaw-tip to object distance")
    parser.add_argument("--no-sweep", action="store_true", help="stop after centring the object")
    parser.add_argument("--min-tilt-deg", type=float, default=0.0,
                        help="only centring views at least this far from straight down (side views for heights)")
    parser.add_argument("--object-size-cm", type=float, nargs=2, metavar=("DIAMETER", "HEIGHT"),
                        help="treat the object as an upright cylinder on the table (e.g. a can) and keep clear of it")
    parser.add_argument("--clearance-cm", type=float, default=4.0,
                        help="minimum distance from jaws, flange and camera to the --object-size-cm cylinder")
    parser.add_argument("--look", action="store_true", help="only save the current wrist frame")
    parser.add_argument("--execute", action="store_true", help="move the arm (otherwise dry run)")
    parser.add_argument("--offline", type=Path, help="re-evaluate a saved attempt directory (needs --roi)")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--parallax-deg", type=float, default=8.0, help="target parallax used to size the slide")
    parser.add_argument("--max-slide-cm", type=float, default=5.0)
    parser.add_argument("--min-slide-cm", type=float, default=1.0)
    parser.add_argument("--max-joint-deg", type=float, default=20.0, help="largest acceptable joint swing")
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--no-table-align", dest="table_align", action="store_false",
                        help="skip levelling/scaling the cloud to the known table plane")
    parser.add_argument("--out", type=Path, default=REPO / "calib_frames" / "stereo_sweep")
    parser.add_argument("--rig", type=Path, default=REPO / "calib" / "rig_calib.json")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_WRIST_INTRINSICS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    args.obstacle = None
    if args.offline:
        if not args.roi:
            parser.error("--offline needs --roi")
        summary, _ = evaluate(args.offline, tuple(args.roi), args)
        print(json.dumps({k: summary[k] for k in summary }, indent=2, default=float))
        return 0 if summary["ok"] else 1
    sources = [args.target, args.object, args.object_top_pixel, args.object_wrist_pixel]
    if not args.look and sum(s is not None for s in sources) != 1:
        parser.error("give exactly one of --target, --object, --object-top-pixel, --object-wrist-pixel "
                     "(or --look / --offline)")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
