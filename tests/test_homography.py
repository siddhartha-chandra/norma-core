"""Unit, calibration, CLI, and Norma-core simulation tests for issue #4:
4-point planar homography fallback (`fit_plane_homography`) and `<60s` table re-anchor (`reanchor_plane`).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from desk_core.calib_store import CalibrationStore
from desk_core.homography import (
    apply_homography,
    clear_plane_homography,
    fit_homography_from_checkerboard,
    fit_plane_homography,
    invert_homography_point,
    reanchor_plane,
)
from desk_core.ik_adapter import IKAdapter
from desk_core.station_bridge import MockStationBridge


def _make_tilted_camera_homography() -> np.ndarray:
    """Build a realistic non-affine (projective) world (x, y) -> pixel (u, v) homography."""
    K = np.array([[580.0, 0.0, 320.0], [0.0, 580.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # Camera mounted at (0.04, -0.36, 0.58) looking down and forward at tabletop (x in [-0.15, 0.15], y in [0.10, 0.35])
    angle = math.radians(14.0)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ])
    # World-to-camera rotation: camera +X = world +X, camera +Y = world -Y (tilted), camera +Z = down
    R_base = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    R_cw = Rx @ R_base
    t_wc = np.array([0.02, 0.22, 0.62])
    t_cw = -R_cw @ t_wc
    # Homography from tabletop z=0 to image pixels: H_w2i = K @ [r1, r2, t_cw]
    H_w2i = K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], t_cw])
    return H_w2i / H_w2i[2, 2]


def _proj_pts(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = np.empty_like(pts, dtype=np.float64)
    for i, (x, y) in enumerate(pts):
        out[i] = apply_homography(H, float(x), float(y))
    return out


# ---------------------------------------------------------------------------
# 1. Core math: fit_plane_homography
# ---------------------------------------------------------------------------


def test_fit_plane_homography_4pt_projective_exact():
    H_w2i = _make_tilted_camera_homography()
    world_corners = np.array([
        [-0.18, 0.10],
        [0.18, 0.10],
        [0.18, 0.36],
        [-0.18, 0.36],
    ], dtype=np.float64)
    pixel_corners = _proj_pts(H_w2i, world_corners)

    fit = fit_plane_homography(pixel_corners, world_corners)
    assert fit.method == "dlt_4pt"
    assert fit.num_points == 4
    assert fit.max_error_m < 1e-6
    assert fit.rmse_m < 1e-6
    assert fit.pixel_rmse_px < 1e-4

    # Verify interior and near-edge test points map accurately and round-trip
    test_world = np.array([
        [0.0, 0.22],
        [-0.10, 0.15],
        [0.12, 0.31],
        [-0.05, 0.28],
    ], dtype=np.float64)
    test_px = _proj_pts(H_w2i, test_world)
    for (u, v), (wx, wy) in zip(test_px, test_world):
        px_x, px_y = fit.project(u, v)
        assert math.hypot(px_x - wx, px_y - wy) < 1e-6
        ru, rv = fit.inverse_project(wx, wy)
        assert math.hypot(ru - u, rv - v) < 1e-4
        ru2, rv2 = invert_homography_point(fit.H, wx, wy)
        assert math.hypot(ru2 - u, rv2 - v) < 1e-4


def test_fit_plane_homography_overdetermined_and_ransac():
    H_w2i = _make_tilted_camera_homography()
    rng = np.random.default_rng(42)
    world_pts = np.array([
        [-0.18, 0.10], [0.0, 0.10], [0.18, 0.10],
        [-0.18, 0.23], [0.0, 0.23], [0.18, 0.23],
        [-0.18, 0.36], [0.0, 0.36], [0.18, 0.36],
    ], dtype=np.float64)
    pixel_pts = _proj_pts(H_w2i, world_pts) + rng.normal(0.0, 0.4, size=world_pts.shape)

    fit_lsq = fit_plane_homography(pixel_pts, world_pts)
    assert fit_lsq.method == "dlt_lsq"
    assert fit_lsq.max_error_m < 0.003

    # Inject a gross outlier on point 4 and verify RANSAC rejects it
    noisy_px = pixel_pts.copy()
    noisy_px[4] += np.array([95.0, -75.0])
    fit_ransac = fit_plane_homography(noisy_px, world_pts, ransac_reproj_threshold_m=0.015)
    assert fit_ransac.method == "ransac"
    assert fit_ransac.inliers[4] is False
    assert sum(fit_ransac.inliers) == 8
    assert fit_ransac.max_error_m < 0.003


@pytest.mark.parametrize(
    "bad_px,bad_w,err_sub",
    [
        ([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]], "at least 4"),
        ([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]], "same length"),
        ([[0, 0], [100, 0], [200, 0], [300, 0]], [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], "collinear"),
        ([[0, 0], [100, 0], [200, 0], [100, 100]], [[0, 0], [0.1, 0], [0.2, 0.1], [0, 0.1]], "collinear"),
        ([[0, 0], [100, 0], [100, 100], [float("nan"), 100]], [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], "non-finite"),
    ],
)
def test_fit_plane_homography_rejects_invalid_inputs(bad_px, bad_w, err_sub):
    with pytest.raises(ValueError, match=err_sub):
        fit_plane_homography(bad_px, bad_w)


# ---------------------------------------------------------------------------
# 2. Automatic <1s checkerboard homography
# ---------------------------------------------------------------------------


def test_fit_homography_from_checkerboard_synthetic():
    cols, rows = 6, 4
    sq_m = 0.025
    # Build a crisp flat checkerboard image (cols+1 x rows+1 squares) with white border
    cell_px = 40
    margin = 40
    bw = (cols + 1) * cell_px + 2 * margin
    bh = (rows + 1) * cell_px + 2 * margin
    board_img = np.full((bh, bw, 3), 245, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                x0 = margin + c * cell_px
                y0 = margin + r * cell_px
                board_img[y0 : y0 + cell_px, x0 : x0 + cell_px] = 20

    # Warp the flat board into a 640x480 tilted overhead camera view
    src_quad = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
    dst_quad = np.array([[150, 110], [490, 95], [520, 360], [125, 345]], dtype=np.float32)
    M_warp = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cam_img = cv2.warpPerspective(board_img, M_warp, (640, 480), borderValue=(180, 180, 180))

    # Hint the top-left inner corner in warped camera coordinates
    tl_board = np.array([[[margin + cell_px, margin + cell_px]]], dtype=np.float32)
    tl_cam = cv2.perspectiveTransform(tl_board, M_warp)[0, 0]

    origin_xy = (0.05, 0.15)
    fit, det_px, world_pts, det_method = fit_homography_from_checkerboard(
        cam_img,
        (cols, rows),
        sq_m,
        board_origin_world_xy=origin_xy,
        origin_hint_px=(float(tl_cam[0]), float(tl_cam[1])),
    )
    assert det_method in ("sb", "classic")
    assert len(det_px) == cols * rows
    assert fit.max_error_m < 0.001  # < 1 mm!
    # Check an arbitrary board square center maps accurately
    sample_board = np.array([[[margin + 3.5 * cell_px, margin + 2.5 * cell_px]]], dtype=np.float32)
    sample_cam = cv2.perspectiveTransform(sample_board, M_warp)[0, 0]
    wx, wy = fit.project(float(sample_cam[0]), float(sample_cam[1]))
    expected_x = origin_xy[0] + 2.5 * sq_m
    expected_y = origin_xy[1] + 1.5 * sq_m
    assert math.hypot(wx - expected_x, wy - expected_y) < 0.001


# ---------------------------------------------------------------------------
# 3. reanchor_plane persistence, anchor reuse, tolerance & CalibrationStore
# ---------------------------------------------------------------------------


def test_reanchor_plane_persistence_and_repeat_reanchor(tmp_path: Path):
    calib_file = tmp_path / "rig_calib.json"
    calib_file.write_text(json.dumps({
        "calibrated": True,
        "arm_type": "elrobot",
        "num_arm_joints": 7,
        "z_table": 0.00237,
        "joint_zero_ticks": {"1": 2048, "2": 2048, "3": 2048, "4": 2048, "5": 1845, "6": 1927, "7": 2200},
        "joint_signs": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1},
    }), encoding="utf-8")

    H_w2i = _make_tilted_camera_homography()
    world_corners = [[-0.15, 0.12], [0.15, 0.12], [0.15, 0.34], [-0.15, 0.34]]
    px_corners = _proj_pts(H_w2i, np.array(world_corners)).tolist()

    res1 = reanchor_plane(px_corners, world_corners, z_table=0.003, calib_path=calib_file)
    assert res1["ok"] is True
    assert res1["saved"] is True
    assert res1["elapsed_s"] < 1.0
    assert res1["max_error_mm"] < 0.5

    # Existing keys preserved + H_table and table_anchors written
    cs = CalibrationStore(calib_file)
    assert cs.data["joint_zero_ticks"]["5"] == 1845
    assert cs.z_table() == pytest.approx(0.003)
    st = cs.status()
    assert st["has_H_table"] is True
    assert st["has_table_anchors"] is True
    assert "T_world_cam_top (or H_table)" not in st["missing_required"]

    # Pure homography fallback works both for z_m=None AND explicit z_m (when T_world_cam_top is absent)
    mid_px = apply_homography(H_w2i, 0.02, 0.24)
    wx0, wy0, wz0 = cs.pixel_to_world(mid_px[0], mid_px[1], camera="top")
    assert math.hypot(wx0 - 0.02, wy0 - 0.24) < 1e-4
    assert wz0 == pytest.approx(0.003)
    wx1, wy1, wz1 = cs.pixel_to_world(mid_px[0], mid_px[1], z_m=0.05, camera="top")
    assert math.hypot(wx1 - 0.02, wy1 - 0.24) < 1e-4
    assert wz1 == pytest.approx(0.05)

    # Simulate Sunday morning tripod bump (+35 px shift): repeat reanchor omitting world_points!
    shifted_px = [[u + 35.0, v - 22.0] for u, v in px_corners]
    res2 = reanchor_plane(shifted_px, calib_path=calib_file)
    assert res2["ok"] is True
    cs.reload()
    wx_s, wy_s, _ = cs.pixel_to_world(mid_px[0] + 35.0, mid_px[1] - 22.0, camera="top")
    assert math.hypot(wx_s - 0.02, wy_s - 0.24) < 1e-4

    # Clear H_table override
    clr = clear_plane_homography(calib_path=calib_file)
    assert clr["removed"] is True
    cs.reload()
    assert cs.status()["has_H_table"] is False
    assert cs.status()["has_table_anchors"] is True


def test_reanchor_plane_rejects_high_error(tmp_path: Path):
    calib_file = tmp_path / "rig_calib.json"
    calib_file.write_text("{}", encoding="utf-8")
    H_w2i = _make_tilted_camera_homography()
    world_pts = np.array([
        [-0.15, 0.12], [0.15, 0.12], [0.15, 0.34], [-0.15, 0.34], [0.0, 0.23],
    ])
    px_pts = _proj_pts(H_w2i, world_pts)
    px_pts[4] += np.array([60.0, -50.0])  # ~5 cm inconsistent 5th click

    with pytest.raises(ValueError, match="exceeds threshold"):
        reanchor_plane(px_pts, world_pts, calib_path=calib_file, max_error_m=0.01)
    assert json.loads(calib_file.read_text(encoding="utf-8")) == {}


# ---------------------------------------------------------------------------
# 4. Real rig_calib.json ray-plane vs 4-corner homography consistency
# ---------------------------------------------------------------------------


def test_real_rig_calib_homography_within_1cm():
    real_calib = CalibrationStore(REPO_ROOT / "calib" / "rig_calib.json")
    z_table = real_calib.z_table()
    # 4 corners spanning the reachable tabletop in front of ElRobot
    corners_w = [(-0.18, -0.15), (0.12, -0.15), (0.12, 0.10), (-0.18, 0.10)]
    corners_px = [real_calib.project_world_to_pixel([x, y, z_table], camera="top") for x, y in corners_w]
    assert all(p is not None for p in corners_px)

    fit = fit_plane_homography(corners_px, corners_w)
    assert fit.max_error_m < 1e-4

    # Check interior tabletop points against true distorted ray-plane projection
    for tx in (-0.12, -0.04, 0.04, 0.08):
        for ty in (-0.10, -0.02, 0.06):
            uv = real_calib.project_world_to_pixel([tx, ty, z_table], camera="top")
            assert uv is not None
            hx, hy = fit.project(uv[0], uv[1])
            err_m = math.hypot(hx - tx, hy - ty)
            assert err_m < 0.01, f"real rig homography error {err_m*1000:.2f} mm >= 10 mm at ({tx}, {ty})"


# ---------------------------------------------------------------------------
# 5. End-to-end Norma-core simulation verification (<1 cm arm XY accuracy)
# ---------------------------------------------------------------------------


def test_simulation_tripod_shift_reanchor_and_arm_xy_within_1cm(tmp_path: Path):
    """End-to-end simulation check for Issue #4:
    1. Start with calibrated rig_calib.json whose T_world_cam_top is now stale (tripod shifted).
    2. Re-anchor in <1s with 4 corner pixels via reanchor_plane().
    3. For multiple clicked tabletop pixels (u, v), map pixel -> tabletop (x, y) via
       CalibrationStore.pixel_to_world, solve TRAC-IK, command MockStationBridge joints,
       and verify URDF FK end-effector lands directly above (true_x, true_y) within < 1 cm.
    """
    real_raw = json.loads((REPO_ROOT / "calib" / "rig_calib.json").read_text(encoding="utf-8"))
    calib_file = tmp_path / "rig_calib.json"
    calib_file.write_text(json.dumps(real_raw), encoding="utf-8")

    # Simulate a bumped/tilted overhead tripod camera
    H_shifted_w2i = _make_tilted_camera_homography()
    z_table = float(real_raw["z_table"])

    # Before re-anchor, stale T_world_cam_top maps a tabletop pixel far off (>5 cm)
    cs = CalibrationStore(calib_file)
    probe_world = (0.02, 0.22)
    probe_px = apply_homography(H_shifted_w2i, *probe_world)
    stale_x, stale_y, _ = cs.pixel_to_world(probe_px[0], probe_px[1], camera="top")
    assert math.hypot(stale_x - probe_world[0], stale_y - probe_world[1]) > 0.05

    # Fast <60s re-anchor from 4 taped table corners
    anchor_world = [[-0.18, 0.12], [0.18, 0.12], [0.18, 0.34], [-0.18, 0.34]]
    anchor_px = _proj_pts(H_shifted_w2i, np.array(anchor_world)).tolist()
    res = reanchor_plane(anchor_px, anchor_world, z_table=z_table, calib_path=calib_file)
    assert res["ok"] is True
    assert res["elapsed_s"] < 1.0
    cs.reload()

    # Drive MockStationBridge + TRAC-IK + URDF FK to multiple clicked tabletop pixels
    test_targets = [
        (0.00, 0.24),
        (0.07, 0.22),
        (-0.07, 0.23),
        (0.03, 0.19),
    ]
    ik = IKAdapter()

    async def _verify_all() -> None:
        bridge = MockStationBridge()
        await bridge.connect()
        hover_z = z_table + 0.18
        for true_x, true_y in test_targets:
            u, v = apply_homography(H_shifted_w2i, true_x, true_y)
            wx, wy, wz = cs.pixel_to_world(u, v, camera="top")
            assert math.hypot(wx - true_x, wy - true_y) < 0.001

            st = await bridge.get_state()
            q_seed = cs.ticks_to_rad(st.motors)
            q_sol = ik.solve_ik(
                x=wx,
                y=wy,
                z=hover_z,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                qw=1.0,
                seed_joints_rad=q_seed,
                joint_bounds_rad=cs.bounds_rad(st.motors),
            )
            ticks = cs.rad_to_ticks(q_sol, st.motors)
            for _ in range(30):
                applied = await bridge.send_joint_targets(ticks)
                if all(abs(applied[mid] - ticks[mid]) <= 1 for mid in ticks):
                    break
            st_after = await bridge.get_state()
            q_after = cs.ticks_to_rad(st_after.motors)
            T_ee = cs.fk_ee_pose(q_after)
            xy_err_m = math.hypot(float(T_ee[0, 3]) - true_x, float(T_ee[1, 3]) - true_y)
            assert xy_err_m < 0.01, (
                f"Arm FK XY error {xy_err_m * 1000.0:.2f} mm >= 10 mm for target ({true_x}, {true_y})"
            )

    asyncio.run(_verify_all())


def test_mcp_reanchor_plane_and_move_in_mock_simulation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify the MCP tool `reanchor_plane` hot-reloads calibration and drives MockStationBridge accurately."""
    import desk_core.mcp_server as srv

    real_raw = json.loads((REPO_ROOT / "calib" / "rig_calib.json").read_text(encoding="utf-8"))
    calib_file = tmp_path / "rig_calib.json"
    calib_file.write_text(json.dumps(real_raw), encoding="utf-8")

    # Point MCP server at isolated temp rig_calib.json and a fresh MockStationBridge
    monkeypatch.setattr(srv, "_calib", CalibrationStore(calib_file))
    monkeypatch.setattr(srv, "_bridge", MockStationBridge())
    monkeypatch.setattr(srv, "_main_control", None)

    # MockStationBridge synthetic top camera projection:
    # u = 320 - (wy / 0.65) * 680, v = 240 - ((wx - 0.28) / 0.65) * 680
    def mock_top_proj(wx: float, wy: float) -> list[float]:
        return [320.0 - (wy / 0.65) * 680.0, 240.0 - ((wx - 0.28) / 0.65) * 680.0]

    anchor_world = [[0.38, 0.20], [0.38, -0.20], [0.18, -0.20], [0.18, 0.20]]
    anchor_px = [mock_top_proj(x, y) for x, y in anchor_world]

    async def _run() -> None:
        res = await srv.reanchor_plane(pixel_points=anchor_px, world_points=anchor_world, z_table=0.0)
        assert res["ok"] is True
        assert res["calibration"]["has_H_table"] is True
        assert res["calibration"]["has_table_anchors"] is True

        # Project simulated 'keys' object at (0.27, -0.06) through pixel_to_world
        ku, kv = mock_top_proj(0.27, -0.06)
        world = await srv.pixel_to_world(x=ku, y=kv, camera="top")
        assert math.hypot(world["x"] - 0.27, world["y"] - (-0.06)) < 0.001

        move = await srv.move_to_pose(x=world["x"], y=world["y"], z=world["z"] + 0.14)
        assert move["ok"] is True
        scene = srv._bridge.get_mock_scene_state()
        assert math.hypot(scene["ee_pos"]["x"] - 0.27, scene["ee_pos"]["y"] - (-0.06)) < 0.01

        # Clear H_table via MCP tool
        clr = await srv.reanchor_plane(clear=True)
        assert clr["removed"] is True
        assert clr["calibration"]["has_H_table"] is False

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. CLI utility scripts/reanchor_plane.py
# ---------------------------------------------------------------------------


def test_cli_reanchor_plane_mock_and_move_test(tmp_path: Path):
    real_raw = json.loads((REPO_ROOT / "calib" / "rig_calib.json").read_text(encoding="utf-8"))
    calib_file = tmp_path / "rig_calib.json"
    calib_file.write_text(json.dumps(real_raw), encoding="utf-8")
    overlay_file = tmp_path / "reanchor_overlay.jpg"

    H_w2i = _make_tilted_camera_homography()
    anchor_world = [[-0.15, 0.14], [0.15, 0.14], [0.15, 0.34], [-0.15, 0.34]]
    anchor_px = _proj_pts(H_w2i, np.array(anchor_world)).tolist()
    test_u, test_v = apply_homography(H_w2i, 0.02, 0.23)

    env = dict(os.environ)
    env["DESK_MOCK_SYNTHETIC"] = "1"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "reanchor_plane.py"),
        "--calib", str(calib_file),
        "--mock",
        "--pixels", json.dumps(anchor_px),
        "--world-points", json.dumps(anchor_world),
        "--z-table", str(real_raw["z_table"]),
        "--verify-pixel", f"{test_u:.3f}", f"{test_v:.3f}",
        "--move-test",
        "--hover-z", "0.18",
        "--save-overlay", str(overlay_file),
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    data = json.loads(proc.stdout)
    assert data["ok"] is True
    assert data["saved"] is True
    assert data["move_test"]["ok"] is True
    assert data["move_test"]["xy_error_mm"] < 10.0
    assert overlay_file.exists() and overlay_file.stat().st_size > 1000


def test_pixel_to_world_uses_homography_for_explicit_z_table(tmp_path: Path):
    """Explicit z_m == z_table() must still use re-anchored H_table over stale T_world_cam_top."""
    real_raw = json.loads((REPO_ROOT / "calib" / "rig_calib.json").read_text(encoding="utf-8"))
    calib_file = tmp_path / "rig_calib.json"
    calib_file.write_text(json.dumps(real_raw), encoding="utf-8")

    # Fit a shifted H_table while keeping stale T_world_cam_top in rig_calib.json
    H_w2i = _make_tilted_camera_homography()
    world_corners = [[-0.15, 0.14], [0.15, 0.14], [0.15, 0.34], [-0.15, 0.34]]
    px_corners = _proj_pts(H_w2i, np.array(world_corners)).tolist()
    reanchor_plane(px_corners, world_corners, z_table=real_raw["z_table"], calib_path=calib_file)

    cs = CalibrationStore(calib_file)
    assert cs.T_world_cam("top") is not None
    u, v = apply_homography(H_w2i, 0.03, 0.22)
    wx_none, wy_none, wz_none = cs.pixel_to_world(u, v, camera="top")
    wx_exp, wy_exp, wz_exp = cs.pixel_to_world(u, v, z_m=cs.z_table(), camera="top")
    assert math.hypot(wx_none - 0.03, wy_none - 0.22) < 0.001
    assert (wx_exp, wy_exp, wz_exp) == (wx_none, wy_none, wz_none)


def test_mcp_reanchor_plane_blocked_during_active_motion():
    """MCP reanchor_plane must serialize against active checked moves via @_motion_tool."""
    import desk_core.mcp_server as srv

    async def _run() -> None:
        async with srv._flange_lock:
            with pytest.raises(RuntimeError, match="motion operation is running"):
                await srv.reanchor_plane(clear=True)

    asyncio.run(_run())


def test_fit_homography_from_checkerboard_requires_origin_hint():
    """fit_homography_from_checkerboard must reject missing origin_hint_px to prevent 180-deg flips."""
    dummy = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="origin_hint_px is required"):
        fit_homography_from_checkerboard(dummy, (6, 4), 0.025)


def test_clear_plane_homography_removes_nested_h_table(tmp_path: Path):
    """clear_plane_homography must also remove nested top/scene H_table so reload() does not restore it."""
    calib_file = tmp_path / "nested_rig.json"
    calib_file.write_text(
        json.dumps(
            {
                "H_table": np.eye(3).tolist(),
                "scene": {"H_table": (2.0 * np.eye(3)).tolist()},
                "top": {"H_table": (3.0 * np.eye(3)).tolist()},
            }
        ),
        encoding="utf-8",
    )
    res = clear_plane_homography(calib_path=calib_file)
    assert res["removed"] is True
    cs = CalibrationStore(calib_file)
    assert cs.data.get("H_table") is None


def test_cli_dry_run_clear_preserves_calibration(tmp_path: Path):
    """--dry-run --clear must leave rig_calib.json unchanged on disk."""
    calib_file = tmp_path / "rig_calib.json"
    calib_file.write_text(json.dumps({"H_table": np.eye(3).tolist()}), encoding="utf-8")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "reanchor_plane.py"),
        "--calib", str(calib_file),
        "--dry-run",
        "--clear",
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["removed"] is True
    assert data["saved"] is False
    assert "H_table" in json.loads(calib_file.read_text(encoding="utf-8"))


def test_cli_mock_without_explicit_calib_does_not_overwrite_default_rig(tmp_path: Path):
    """--mock without --calib must default to dry-run so it never overwrites live rig_calib.json."""
    real_raw = json.loads((REPO_ROOT / "calib" / "rig_calib.json").read_text(encoding="utf-8"))
    calib_file = tmp_path / "rig_calib.json"
    calib_file.write_text(json.dumps(real_raw), encoding="utf-8")

    H_w2i = _make_tilted_camera_homography()
    anchor_world = [[-0.15, 0.14], [0.15, 0.14], [0.15, 0.34], [-0.15, 0.34]]
    anchor_px = _proj_pts(H_w2i, np.array(anchor_world)).tolist()

    env = dict(os.environ)
    env["DESK_MOCK_SYNTHETIC"] = "1"
    env["DESK_CALIB_PATH"] = str(calib_file)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "reanchor_plane.py"),
        "--mock",
        "--pixels", json.dumps(anchor_px),
        "--world-points", json.dumps(anchor_world),
        "--json",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["ok"] is True
    assert data["saved"] is False
    assert "H_table" not in json.loads(calib_file.read_text(encoding="utf-8"))
