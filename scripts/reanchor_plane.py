#!/usr/bin/env python3
"""<60s tabletop planar homography fallback & fast re-anchor utility for DESK-CORE.

Examples:
  # 1. Headless 4-corner re-anchor (writes H_table + table_anchors to calib/rig_calib.json)
  uv run --offline --managed-python python scripts/reanchor_plane.py \\
      --pixels "120,110;520,110;520,390;120,390" \\
      --world-points "0.35,0.18;0.35,-0.18;0.15,-0.18;0.15,0.18" \\
      --z-table 0.00237

  # 2. Repeat Sunday-morning re-anchor using previously saved world_points (only 4 new clicks)
  uv run --offline --managed-python python scripts/reanchor_plane.py --live --interactive

  # 3. Automatic <1s checkerboard re-anchor on a captured or live frame (pass board origin & axes in base_link)
  uv run --offline --managed-python python scripts/reanchor_plane.py \\
      --image calib_frames/scene_1/scene.jpg --checkerboard 10 7 --square-size-m 0.015 \\
      --board-origin 0.15 0.10 --board-x-dir 1 0 --board-y-dir 0 1 --origin-hint 320 240

  # 4. Offline mock simulation re-anchor + move-above-pixel (<1 cm) verification (--mock implies --dry-run unless --calib is passed)
  DESK_MOCK_SYNTHETIC=1 uv run --offline --managed-python python scripts/reanchor_plane.py \\
      --mock --dry-run --pixels "110.8,135.4;529.2,135.4;529.2,344.6;110.8,344.6" \\
      --world-points "0.38,0.20;0.38,-0.20;0.18,-0.20;0.18,0.20" \\
      --verify-pixel 382.8 250.5 --move-test

  # 5. Clear H_table override to revert to T_world_cam_top ray-plane projection
  uv run --offline --managed-python python scripts/reanchor_plane.py --clear
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if os.environ.get("DESK_NO_DOTENV") != "1":
    for _k, _v in dotenv_values(REPO_ROOT / ".env").items():
        if _v is not None and not os.environ.get(_k, "").strip():
            os.environ[_k] = _v

from desk_core.calib_store import CalibrationStore, resolve_default_calib_path  # noqa: E402
from desk_core.homography import (  # noqa: E402
    DEFAULT_MAX_ERROR_M,
    HomographyFit,
    clear_plane_homography,
    draw_reanchor_overlay,
    fit_homography_from_checkerboard,
    fit_plane_homography,
    reanchor_plane,
)
from desk_core.ik_adapter import IKAdapter  # noqa: E402
from desk_core.station_bridge import LiveStationBridge, MockStationBridge  # noqa: E402


def parse_points(raw: str) -> list[list[float]]:
    """Parse 2D points from JSON ``[[x, y], ...]`` or semicolon syntax ``x1,y1;x2,y2;...``."""
    text = raw.strip()
    if text.startswith("["):
        parsed = json.loads(text)
        return [[float(p[0]), float(p[1])] for p in parsed]
    pts: list[list[float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected 'x,y' pair, got {chunk!r}")
        pts.append([float(parts[0]), float(parts[1])])
    return pts


def _make_bridge(live: bool, calib: CalibrationStore, *, follow_motor_rx: bool = False) -> Any:
    if not live:
        return MockStationBridge()
    if os.environ.get("DESK_MOCK") == "1":
        raise RuntimeError("DESK_MOCK=1 is set; refusing to open a live Station connection.")
    hw = calib.hardware_config()
    host = os.environ.get("STATION_HOST", "").strip() or "127.0.0.1"
    port = int(os.environ.get("STATION_PORT", "8888"))
    bus = os.environ.get("STATION_BUS_SERIAL", "").strip() or hw.get("bus_serial") or None
    cam_top = os.environ.get("CAMERA_TOP", "").strip() or hw.get("camera_top") or None
    cam_wrist = os.environ.get("CAMERA_WRIST", "").strip() or hw.get("camera_wrist") or None
    return LiveStationBridge(
        host=host,
        port=port,
        target_bus=bus,
        cam_top=cam_top,
        cam_wrist=cam_wrist,
        follow_motor_rx=follow_motor_rx,
    )


async def _grab_frame(live: bool, calib: CalibrationStore) -> tuple[np.ndarray, Any]:
    bridge = _make_bridge(live=live, calib=calib, follow_motor_rx=False)
    await bridge.connect()
    try:
        jpeg = await bridge.get_frame("top")
    finally:
        if hasattr(bridge, "close"):
            await bridge.close()
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode top camera JPEG frame.")
    return img, bridge


def _click_corners_interactive(image: np.ndarray, expected_count: int = 4) -> list[list[float]]:
    """OpenCV interactive point picker for <60s 4-corner re-anchor."""
    clicked: list[list[float]] = []
    win = "DESK-CORE Re-Anchor (click corners in order; Enter=done, Backspace=undo, Esc=cancel)"

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([float(x), float(y)])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse)
    try:
        while True:
            vis = image.copy()
            prompt = f"Clicked {len(clicked)}/{expected_count} corners (Enter=confirm, Backspace=undo, Esc=cancel)"
            cv2.rectangle(vis, (8, 8), (620, 34), (20, 20, 20), -1)
            cv2.putText(vis, prompt, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 220), 1)
            for idx, (u, v) in enumerate(clicked, start=1):
                pt = (int(round(u)), int(round(v)))
                cv2.circle(vis, pt, 7, (0, 220, 0), 2)
                cv2.putText(vis, f"P{idx}", (pt[0] + 8, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            cv2.imshow(win, vis)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and len(clicked) >= 4:
                return clicked
            if key in (8, 127, ord("r")) and clicked:
                clicked.pop()
            if key in (27, ord("q")):
                raise KeyboardInterrupt("Interactive re-anchor cancelled by user.")
    finally:
        cv2.destroyWindow(win)


async def _run_move_test(
    calib: CalibrationStore,
    pixel_uv: tuple[float, float],
    *,
    live: bool,
    hover_z_m: float,
    max_error_m: float,
) -> dict[str, Any]:
    """Project pixel (u, v) -> tabletop (x, y) via H_table, solve TRAC-IK, move arm above spot, verify FK XY < 1 cm."""
    wx, wy, wz = calib.pixel_to_world(pixel_uv[0], pixel_uv[1], camera="top")
    target_z = wz + float(hover_z_m)
    bridge: Any = _make_bridge(live=live, calib=calib, follow_motor_rx=live)
    await bridge.connect()
    try:
        if live:
            await bridge.get_frame("top")
        st = await bridge.get_state()
        q_seed = calib.ticks_to_rad(st.motors)
        ik = IKAdapter()
        # Try downward-ish grasp orientation first, then neutral flange orientation.
        q_sol = None
        last_err = None
        for qx, qy, qz, qw in (
            (0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
            (0.0, 0.0, 0.0, 1.0),
        ):
            try:
                q_sol = ik.solve_ik(
                    x=wx,
                    y=wy,
                    z=target_z,
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                    seed_joints_rad=q_seed,
                    joint_bounds_rad=calib.bounds_rad(st.motors),
                )
                break
            except Exception as exc:
                last_err = exc
        if q_sol is None:
            raise RuntimeError(f"IK failed for tabletop target ({wx:.4f}, {wy:.4f}, {target_z:.4f}): {last_err}")

        ticks = calib.rad_to_ticks(q_sol, st.motors)
        # Command joints in safe steps until settled (on mock or live)
        for _ in range(25):
            applied = await bridge.send_joint_targets(ticks)
            if all(abs(applied[m_id] - ticks[m_id]) <= 1 for m_id in ticks):
                break
        await bridge.wait_settle(ticks)
        st_after = await bridge.get_state()
        q_after = calib.ticks_to_rad(st_after.motors)
        T_ee = calib.fk_ee_pose(q_after)
        ee_x, ee_y, ee_z = float(T_ee[0, 3]), float(T_ee[1, 3]), float(T_ee[2, 3])
        xy_err_m = math.hypot(ee_x - wx, ee_y - wy)
        return {
            "ok": xy_err_m <= max_error_m,
            "pixel": [pixel_uv[0], pixel_uv[1]],
            "target_world_m": [round(wx, 5), round(wy, 5), round(target_z, 5)],
            "ee_fk_world_m": [round(ee_x, 5), round(ee_y, 5), round(ee_z, 5)],
            "xy_error_mm": round(xy_err_m * 1000.0, 3),
            "threshold_mm": round(max_error_m * 1000.0, 3),
            "mode": "live" if live else "mock",
        }
    finally:
        if hasattr(bridge, "close"):
            await bridge.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fast <60s planar homography fallback & table re-anchor utility.")
    ap.add_argument("--calib", type=Path, default=None, help="Path to rig_calib.json (default: calib/rig_calib.json).")
    ap.add_argument("--live", action="store_true", help="Capture top frame from live Station.")
    ap.add_argument("--mock", action="store_true", help="Capture top frame from MockStationBridge.")
    ap.add_argument("--image", type=Path, default=None, help="Load overhead camera image from disk.")
    ap.add_argument("--pixels", type=str, default=None, help="4+ image pixels: JSON [[u,v],...] or 'u1,v1;u2,v2;...'.")
    ap.add_argument("--world-points", type=str, default=None, help="4+ tabletop (x,y) meters: JSON or 'x1,y1;...'.")
    ap.add_argument("--interactive", action="store_true", help="Click 4+ corners interactively in an OpenCV window.")
    ap.add_argument("--checkerboard", nargs=2, type=int, metavar=("COLS", "ROWS"), default=None,
                    help="Auto-detect checkerboard inner corners and fit H_table in <1s.")
    ap.add_argument("--square-size-m", type=float, default=0.025, help="Checkerboard square size in meters.")
    ap.add_argument("--board-origin", nargs=2, type=float, metavar=("X", "Y"), default=[0.0, 0.0],
                    help="Tabletop (x, y) meters of checkerboard corner (0, 0).")
    ap.add_argument("--board-x-dir", nargs=2, type=float, metavar=("DX", "DY"), default=[1.0, 0.0],
                    help="Tabletop direction of checkerboard +column axis.")
    ap.add_argument("--board-y-dir", nargs=2, type=float, metavar=("DX", "DY"), default=[0.0, 1.0],
                    help="Tabletop direction of checkerboard +row axis.")
    ap.add_argument("--origin-hint", nargs=2, type=float, metavar=("U", "V"), default=None,
                    help="Optional pixel hint near checkerboard corner (0, 0).")
    ap.add_argument("--z-table", type=float, default=None, help="Tabletop height in base_link meters.")
    ap.add_argument("--max-error-cm", type=float, default=DEFAULT_MAX_ERROR_M * 100.0,
                    help="Max allowed reprojection error in cm (default: 1.0).")
    ap.add_argument("--allow-high-error", action="store_true", help="Write H_table even if max error exceeds threshold.")
    ap.add_argument("--dry-run", action="store_true", help="Compute homography without writing rig_calib.json.")
    ap.add_argument("--clear", action="store_true", help="Remove H_table override from rig_calib.json and exit.")
    ap.add_argument("--save-overlay", type=Path, default=None, help="Save annotated verification overlay image.")
    ap.add_argument("--verify-pixel", nargs=2, type=float, metavar=("U", "V"), default=None,
                    help="Project a test pixel (u, v) after re-anchor and print tabletop (x, y, z).")
    ap.add_argument("--move-test", action="store_true",
                    help="With --verify-pixel, move the arm above the pixel and assert FK XY error <= max-error-cm.")
    ap.add_argument("--hover-z", type=float, default=0.15, help="Hover height above z_table for --move-test (meters).")
    ap.add_argument("--json", action="store_true", help="Output result as JSON.")
    args = ap.parse_args(argv)

    calib_path = args.calib or resolve_default_calib_path()
    max_error_m = float(args.max_error_cm) / 100.0

    if args.clear:
        res = clear_plane_homography(calib_path=calib_path, save=not args.dry_run)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            prefix = "[DRY-RUN] Would clear" if args.dry_run else "Cleared"
            print(f"{prefix} H_table from {res['calib_path']} (removed={res['removed']}, saved={res['saved']})")
        return 0

    calib = CalibrationStore(calib_path)
    image: np.ndarray | None = None
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            print(f"Error: could not read image {args.image}", file=sys.stderr)
            return 1
    elif args.live or args.mock or args.interactive or args.checkerboard is not None:
        image, _ = asyncio.run(_grab_frame(live=args.live, calib=calib))

    pixel_points: list[list[float]] | np.ndarray
    world_points: list[list[float]] | np.ndarray | None
    method_label = "4-corner"

    if args.checkerboard is not None:
        if image is None:
            print("Error: --checkerboard requires --image, --live, or --mock.", file=sys.stderr)
            return 1
        if args.origin_hint is None:
            print(
                "Error: --checkerboard requires --origin-hint U V near corner (0, 0) to avoid 180-degree flips.",
                file=sys.stderr,
            )
            return 1
        fit_cb, px_cb, w_cb, det_method = fit_homography_from_checkerboard(
            image,
            (args.checkerboard[0], args.checkerboard[1]),
            args.square_size_m,
            board_origin_world_xy=(args.board_origin[0], args.board_origin[1]),
            board_x_dir=(args.board_x_dir[0], args.board_x_dir[1]),
            board_y_dir=(args.board_y_dir[0], args.board_y_dir[1]),
            origin_hint_px=tuple(args.origin_hint) if args.origin_hint is not None else None,
        )
        pixel_points = px_cb
        world_points = w_cb
        method_label = f"checkerboard_{det_method}"
        fit_obj: HomographyFit | None = fit_cb
    else:
        world_points = parse_points(args.world_points) if args.world_points else None
        if args.pixels:
            pixel_points = parse_points(args.pixels)
        elif args.interactive:
            if image is None:
                print("Error: --interactive requires --image, --live, or --mock.", file=sys.stderr)
                return 1
            saved_w = (calib.table_anchors() or {}).get("world_points")
            expected = len(world_points) if world_points is not None else (len(saved_w) if saved_w else 4)
            pixel_points = _click_corners_interactive(image, expected_count=expected)
        else:
            print("Error: provide --pixels, --interactive, or --checkerboard.", file=sys.stderr)
            return 1
        fit_obj = None

    effective_dry_run = bool(args.dry_run or (args.mock and args.calib is None))
    result = reanchor_plane(
        pixel_points=pixel_points,
        world_points=world_points,
        z_table=args.z_table,
        calib_path=calib_path,
        max_error_m=max_error_m,
        allow_high_error=args.allow_high_error,
        method_label=method_label,
        save=not effective_dry_run,
    )

    if not effective_dry_run:
        calib.reload()
    else:
        calib.data["H_table"] = result["H_table"]
        calib.data["z_table"] = result["z_table"]

    test_points_overlay: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if args.verify_pixel is not None:
        vu, vv = float(args.verify_pixel[0]), float(args.verify_pixel[1])
        wx, wy, wz = calib.pixel_to_world(vu, vv, camera="top")
        result["verify_pixel"] = {"pixel": [vu, vv], "world_m": [wx, wy, wz]}
        test_points_overlay.append(((vu, vv), (wx, wy)))
        if args.move_test:
            move_res = asyncio.run(
                _run_move_test(
                    calib,
                    (vu, vv),
                    live=args.live,
                    hover_z_m=args.hover_z,
                    max_error_m=max_error_m,
                )
            )
            result["move_test"] = move_res

    if args.save_overlay is not None and image is not None:
        if fit_obj is None:
            effective_world = result["table_anchors"]["world_points"]
            fit_obj = fit_plane_homography(pixel_points, effective_world)
        overlay = draw_reanchor_overlay(
            image,
            fit_obj,
            pixel_points,
            result["table_anchors"]["world_points"],
            test_points=test_points_overlay or None,
        )
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_overlay), overlay)
        result["overlay_path"] = str(args.save_overlay)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status_word = "SAVED" if result["saved"] else "DRY-RUN"
        print(
            f"[{status_word}] H_table ({result['method']}, {result['num_points']} pts) -> {result['calib_path']}\n"
            f"  RMSE={result['rmse_mm']:.2f} mm | max_err={result['max_error_mm']:.2f} mm "
            f"(limit {result['max_error_threshold_mm']:.1f} mm) | px_RMSE={result['pixel_rmse_px']:.2f} px "
            f"| z_table={result['z_table']} m | elapsed={result['elapsed_s']:.3f}s"
        )
        if "verify_pixel" in result:
            vp = result["verify_pixel"]
            print(f"  verify pixel ({vp['pixel'][0]:.1f}, {vp['pixel'][1]:.1f}) -> world {vp['world_m']} m")
        if "move_test" in result:
            mt = result["move_test"]
            verdict = "PASS" if mt["ok"] else "FAIL"
            print(
                f"  [{verdict}] move_test ({mt['mode']}): target={mt['target_world_m']} m "
                f"ee_fk={mt['ee_fk_world_m']} m xy_err={mt['xy_error_mm']:.2f} mm"
            )
        if "overlay_path" in result:
            print(f"  overlay saved: {result['overlay_path']}")

    if "move_test" in result and not result["move_test"]["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
