#!/usr/bin/env python3
"""Sanity-check calib/rig_calib.json geometry, SO(3) extrinsics, joint zeros, and wrist FK.

Usage:
  Offline schema + FK + tabletop geometry check:
      uv run --offline --managed-python python scripts/verify_calib.py

  Live rig check + visual base/gripper reprojection overlay:
      uv run --offline --managed-python python scripts/verify_calib.py --live --save-overlay
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from desk_core.calib_store import CalibrationStore, DEFAULT_CALIB_PATH, TICKS_PER_RAD  # noqa: E402
from desk_core.station_bridge import MockStationBridge, create_bridge  # noqa: E402
from desk_core.urdf_fk import load_urdf  # noqa: E402


def _check_so3(name: str, T: np.ndarray, warnings: list[str], errors: list[str]) -> None:
    if T.shape != (4, 4):
        errors.append(f"{name}: expected shape (4,4), got {T.shape}")
        return
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        errors.append(f"{name}: bottom row must be [0, 0, 0, 1], got {T[3].tolist()}")
    R = T[:3, :3]
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det = float(np.linalg.det(R))
    print(f"  {name}: t={np.round(T[:3, 3], 4).tolist()} m | ortho_err={ortho_err:.2e} | det(R)={det:.6f}")
    if ortho_err > 1e-2 or abs(det - 1.0) > 1e-2:
        errors.append(f"{name}: R is not a valid right-handed rotation (ortho_err={ortho_err:.4f}, det={det:.4f})")
    elif ortho_err > 1e-4 or abs(det - 1.0) > 1e-4:
        warnings.append(f"{name}: slight R drift (ortho_err={ortho_err:.2e}, det={det:.5f})")


async def _fetch_live_or_mock(live: bool):
    if live:
        bridge = create_bridge()
        if isinstance(bridge, MockStationBridge):
            raise RuntimeError("STATION_HOST is not set (or DESK_MOCK=1); cannot run --live.")
    else:
        bridge = MockStationBridge()
    await bridge.connect()
    st = await bridge.get_state()
    frame = await bridge.get_frame("top", denoise=False)
    return st, frame


def _save_overlay(
    calib: CalibrationStore,
    frame_jpg: bytes,
    q_rad: list[float],
    out_path: Path,
) -> None:
    arr = cv2.imdecode(np.frombuffer(frame_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return
    # Draw base_link triad (X=red, Y=green, Z=blue, 10 cm arrows)
    axis_len = 0.10
    p0 = calib.project_world_to_pixel([0.0, 0.0, 0.0], camera="top")
    axes = [
        ("X", [axis_len, 0.0, 0.0], (0, 0, 255)),
        ("Y", [0.0, axis_len, 0.0], (0, 255, 0)),
        ("Z", [0.0, 0.0, axis_len], (255, 100, 0)),
    ]
    if p0 is not None:
        c0 = (int(round(p0[0])), int(round(p0[1])))
        cv2.circle(arr, c0, 6, (255, 255, 255), -1)
        cv2.putText(arr, "base_link", (c0[0] + 8, c0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        for label, pt, color in axes:
            p1 = calib.project_world_to_pixel(pt, camera="top")
            if p1 is not None:
                c1 = (int(round(p1[0])), int(round(p1[1])))
                cv2.arrowedLine(arr, c0, c1, color, 2, tipLength=0.18)
                cv2.putText(arr, label, c1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Project current EE gripper pose via FK
    T_ee = calib.fk_ee_pose(q_rad)
    ee_xyz = T_ee[:3, 3]
    p_ee = calib.project_world_to_pixel(ee_xyz, camera="top")
    if p_ee is not None:
        ce = (int(round(p_ee[0])), int(round(p_ee[1])))
        cv2.circle(arr, ce, 8, (0, 255, 255), 2)
        cv2.circle(arr, ce, 3, (0, 255, 255), -1)
        cv2.putText(arr, f"EE ({ee_xyz[0]:.2f},{ee_xyz[1]:.2f},{ee_xyz[2]:.2f})", (ce[0] + 10, ce[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), arr)
    print(f"\nSaved overlay: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH, help="Path to rig_calib.json")
    ap.add_argument("--live", action="store_true", help="Read live robot ticks & top camera from Station")
    ap.add_argument("--save-overlay", action="store_true", help="Save base/EE reprojection overlay JPEG")
    ap.add_argument("--overlay-path", type=Path, default=REPO_ROOT / "calib_frames" / "verify_top_overlay.jpg")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    print(f"=== DESK-CORE Calibration Check: {args.calib} ===")
    if not args.calib.exists():
        print(f"ERROR: file not found: {args.calib}", file=sys.stderr)
        return 1

    calib = CalibrationStore(args.calib)
    st_info = calib.status()
    print(f"  arm_type={st_info['arm_type']} | calibrated={st_info['calibrated']} | z_table={calib.z_table()} m")
    print(f"  hardware (effective): {st_info['hardware']}")
    if not st_info["calibrated"]:
        warnings.append("calibrated=false (still running placeholder values from rig_calib.json)")

    # 1. Camera intrinsics & extrinsics
    print("\n[1] Cameras & Extrinsics:")
    for cam in ("top", "wrist"):
        K = calib._K(cam)
        dist = calib._dist(cam)
        print(f"  {cam}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} | dist={np.round(dist, 5).tolist()}")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            errors.append(f"K_{cam} has non-positive focal length")

    T_top = calib.T_world_cam("top")
    if T_top is None:
        errors.append("T_world_cam_top is missing")
    else:
        _check_so3("T_world_cam_top", T_top, warnings, errors)
        if T_top[2, 3] <= calib.z_table():
            warnings.append(f"T_world_cam_top z={T_top[2,3]:.3f}m is not above z_table={calib.z_table():.3f}m")
        if T_top[2, 2] >= 0:
            warnings.append(f"T_world_cam_top R[2,2]={T_top[2,2]:.3f} >= 0 (optical +Z not pointing downward)")

    if "T_ee_cam_wrist" in calib.data:
        _check_so3("T_ee_cam_wrist", np.asarray(calib.data["T_ee_cam_wrist"], dtype=float), warnings, errors)
    else:
        warnings.append("T_ee_cam_wrist missing (wrist ray projection will fall back to nominal height)")

    # 2. Ray-plane vs H_table cross-check on top camera
    print("\n[2] Top-Camera Tabletop Geometry (ray-plane vs H_table):")
    K_top = calib._K("top")
    cx, cy = float(K_top[0, 2]), float(K_top[1, 2])
    test_px = [(cx, cy), (cx - 120, cy - 80), (cx + 120, cy + 80)]
    has_H = "H_table" in calib.data and calib.data["H_table"] is not None
    H = np.asarray(calib.data["H_table"], dtype=float).reshape(3, 3) if has_H else None
    for u, v in test_px:
        # Force ray-plane projection directly (bypass H_table shortcut)
        ray = calib._ray_cam(u, v, "top")
        if T_top is not None:
            rw = T_top[:3, :3] @ ray
            s = (calib.z_table() - T_top[2, 3]) / rw[2]
            pw_ray = T_top[:3, 3] + s * rw
            depth_z = (s * ray)[2]
        else:
            pw_ray = np.zeros(3)
            depth_z = calib.get_metric_depth(u, v, "top")
        line = f"  px=({u:.0f},{v:.0f}) -> ray_world=({pw_ray[0]:+.4f}, {pw_ray[1]:+.4f}, {pw_ray[2]:+.4f})m depth={depth_z:.4f}m"
        if H is not None:
            hp = H @ np.array([u, v, 1.0])
            hx, hy = hp[0] / hp[2], hp[1] / hp[2]
            diff_mm = math.hypot(hx - pw_ray[0], hy - pw_ray[1]) * 1000.0
            line += f" | H_table_diff={diff_mm:.1f}mm"
            if diff_mm > 25.0:
                warnings.append(f"top px ({u:.0f},{v:.0f}) T_world_cam_top vs H_table differ by {diff_mm:.1f}mm")
        print(line)

    # 3. Joint zeros, signs, URDF limits & live/mock FK
    print("\n[3] ElRobot Joint Zeros & URDF FK:")
    urdf = load_urdf(calib.urdf_path())
    jnames = calib.joint_names()
    lims = urdf.joint_limits(jnames)
    signs = calib.data.get("joint_signs", {})
    zeros = calib.data.get("joint_zero_ticks", {})
    for idx, (jname, (lo, hi)) in enumerate(zip(jnames, lims), start=1):
        z_t = zeros.get(str(idx), 2048) if isinstance(zeros, dict) else zeros[idx - 1]
        s_t = signs.get(str(idx), 1) if isinstance(signs, dict) else signs[idx - 1]
        if s_t not in (-1, 1, -1.0, 1.0):
            errors.append(f"joint_signs[{idx}]={s_t} must be +1 or -1")
        print(f"  j{idx} ({jname}): zero_tick={z_t} sign={s_t:+d} urdf_limits=[{lo:.3f}, {hi:.3f}] rad")

    T_ee_zero = calib.fk_ee_pose([0.0] * len(jnames))
    print(f"  FK at zero pose q=[0]*7 -> EE pos={np.round(T_ee_zero[:3, 3], 4).tolist()} m")

    frame_jpg: bytes | None = None
    q_live = [0.0] * len(jnames)
    try:
        st, frame_jpg = asyncio.run(_fetch_live_or_mock(args.live))
        q_live = calib.ticks_to_rad(st.motors)
        T_ee_live = calib.fk_ee_pose(q_live)
        T_wrist_live = calib.T_world_cam("wrist", q_rad=q_live)
        mode = "LIVE" if args.live else "MOCK"
        print(f"\n[4] {mode} State ({st.bus_serial}):")
        print("  q_deg:", [round(math.degrees(a), 1) for a in q_live])
        print("  EE world xyz (m):", np.round(T_ee_live[:3, 3], 4).tolist())
        if T_wrist_live is not None:
            print("  Wrist cam world xyz (m):", np.round(T_wrist_live[:3, 3], 4).tolist(),
                  "| optical +Z:", np.round(T_wrist_live[:3, 2], 3).tolist())
            px_w = calib.pixel_to_world(320.0, 240.0, camera="wrist", q_rad=q_live)
            print("  Wrist center pixel -> world (m):", px_w)
    except Exception as exc:
        warnings.append(f"telemetry/camera check skipped ({exc})")

    if args.save_overlay and frame_jpg is not None:
        _save_overlay(calib, frame_jpg, q_live, args.overlay_path)

    print("\n=== Summary ===")
    for w in warnings:
        print(f"  [WARN] {w}")
    for e in errors:
        print(f"  [FAIL] {e}")
    print(f"Result: {'FAIL' if errors else ('PASS WITH WARNINGS' if warnings else 'PASS')} ({len(errors)} errors, {len(warnings)} warnings)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
