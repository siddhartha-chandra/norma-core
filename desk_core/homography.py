"""Planar 4-point (and N-point) tabletop homography fallback and <60s table re-anchor.

Maps raw overhead camera pixels ``(u, v)`` directly to tabletop ``(x, y)`` coordinates
in meters in the robot ``base_link`` frame (at height ``z_table``).

Used as:
  1. A zero-dependency fallback when full 3D hand-eye extrinsics (``T_world_cam_top``)
     are unavailable.
  2. A ``<60s`` recalibration utility (``reanchor_plane``) when the overhead tripod or
     table shifts between sessions: writing ``H_table`` to ``calib/rig_calib.json``
     immediately overrides top-camera tabletop projection without re-running full
     hand-eye calibration.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .calib_store import resolve_default_calib_path

DEFAULT_MAX_ERROR_M = 0.01  # 1 cm acceptance threshold from issue #4


@dataclass(frozen=True)
class HomographyFit:
    """Result of fitting a 3x3 planar homography mapping pixels ``(u, v)`` -> world ``(x, y)``."""

    H: np.ndarray
    H_inv: np.ndarray
    rmse_m: float
    max_error_m: float
    mean_error_m: float
    pixel_rmse_px: float
    per_point_errors_m: list[float]
    inliers: list[bool]
    num_points: int
    method: str

    def project(self, u: float, v: float) -> tuple[float, float]:
        """Map overhead camera pixel ``(u, v)`` to tabletop ``(x, y)`` in meters."""
        return apply_homography(self.H, u, v)

    def inverse_project(self, x: float, y: float) -> tuple[float, float]:
        """Map tabletop ``(x, y)`` in meters back to overhead camera pixel ``(u, v)``."""
        return apply_homography(self.H_inv, x, y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "H_table": self.H.tolist(),
            "H_inv": self.H_inv.tolist(),
            "rmse_m": round(self.rmse_m, 6),
            "rmse_mm": round(self.rmse_m * 1000.0, 3),
            "max_error_m": round(self.max_error_m, 6),
            "max_error_mm": round(self.max_error_m * 1000.0, 3),
            "mean_error_m": round(self.mean_error_m, 6),
            "mean_error_mm": round(self.mean_error_m * 1000.0, 3),
            "pixel_rmse_px": round(self.pixel_rmse_px, 3),
            "per_point_errors_mm": [round(e * 1000.0, 3) for e in self.per_point_errors_m],
            "inliers": list(self.inliers),
            "num_inliers": int(sum(self.inliers)),
            "num_points": self.num_points,
            "method": self.method,
        }


def apply_homography(H: np.ndarray | Sequence[Sequence[float]], x: float, y: float) -> tuple[float, float]:
    """Apply a 3x3 projective homography to 2D point ``(x, y)``."""
    H_mat = np.asarray(H, dtype=np.float64).reshape(3, 3)
    pt = H_mat @ np.array([float(x), float(y), 1.0], dtype=np.float64)
    if not np.isfinite(pt).all() or abs(pt[2]) <= 1e-12:
        raise ValueError(f"Point ({x}, {y}) maps to infinity under homography (w={pt[2]}).")
    return (float(pt[0] / pt[2]), float(pt[1] / pt[2]))


def invert_homography_point(H: np.ndarray | Sequence[Sequence[float]], x: float, y: float) -> tuple[float, float]:
    """Apply the inverse of a 3x3 homography ``H`` to 2D point ``(x, y)``."""
    H_mat = np.asarray(H, dtype=np.float64).reshape(3, 3)
    try:
        H_inv = np.linalg.inv(H_mat)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Homography matrix is singular and cannot be inverted.") from exc
    return apply_homography(H_inv, x, y)


def _as_points_2d(points: Any, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 sequence of 2D coordinates, got shape {arr.shape}.")
    if arr.shape[0] < 4:
        raise ValueError(f"{name} requires at least 4 correspondences, got {arr.shape[0]}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _has_collinear_triple(pts: np.ndarray, rel_tol: float = 1e-4) -> bool:
    """Return True if any 3 of 4 points are nearly collinear."""
    span = float(np.max(np.ptp(pts, axis=0)))
    if span <= 1e-9:
        return True
    area_tol = rel_tol * span * span
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area2 = abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
                if area2 <= area_tol:
                    return True
    return False


def _is_degenerate_point_set(pts: np.ndarray) -> bool:
    """Reject point sets that collapse onto a line or single point."""
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s[0] <= 1e-9 or (s[1] / s[0]) < 1e-3:
        return True
    if len(pts) == 4 and _has_collinear_triple(pts):
        return True
    return False


def fit_plane_homography(
    pixel_points: Sequence[Sequence[float]] | np.ndarray,
    world_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    ransac_reproj_threshold_m: float | None = None,
) -> HomographyFit:
    """Fit a 3x3 planar homography mapping overhead pixels ``(u, v)`` to tabletop ``(x, y)`` meters.

    Uses exact DLT + LM refinement for 4+ points by default, or RANSAC followed by
    inlier least-squares refinement when ``ransac_reproj_threshold_m`` is given and
    ``len(pixel_points) >= 5``.
    """
    src = _as_points_2d(pixel_points, "pixel_points")
    dst = _as_points_2d(world_points, "world_points")
    if len(src) != len(dst):
        raise ValueError(
            f"pixel_points ({len(src)}) and world_points ({len(dst)}) must have the same length."
        )
    if _is_degenerate_point_set(src):
        raise ValueError("pixel_points are collinear or degenerate; choose 4 non-collinear tabletop points.")
    if _is_degenerate_point_set(dst):
        raise ValueError("world_points are collinear or degenerate; choose 4 non-collinear tabletop coordinates.")

    use_ransac = ransac_reproj_threshold_m is not None and len(src) >= 5
    if use_ransac:
        threshold = float(ransac_reproj_threshold_m)
        if threshold <= 0.0:
            raise ValueError("ransac_reproj_threshold_m must be positive.")
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=threshold)
        if H is None or mask is None:
            raise ValueError("RANSAC homography estimation failed; check correspondences.")
        inlier_mask = mask.ravel().astype(bool)
        if int(inlier_mask.sum()) < 4:
            raise ValueError(
                f"RANSAC found only {int(inlier_mask.sum())} inliers (need at least 4)."
            )
        # Refine over inliers with least-squares + LM for sub-mm precision.
        H_refined, _ = cv2.findHomography(src[inlier_mask], dst[inlier_mask], 0)
        if H_refined is not None:
            H = H_refined
        method = "ransac"
    else:
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            raise ValueError("Homography estimation failed; points may be degenerate.")
        inlier_mask = np.ones(len(src), dtype=bool)
        method = "dlt_4pt" if len(src) == 4 else "dlt_lsq"

    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(H).all():
        raise ValueError("Estimated homography contains non-finite entries.")
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    det = float(np.linalg.det(H))
    if not math.isfinite(det) or abs(det) <= 1e-15:
        raise ValueError(f"Estimated homography is singular (det={det:.3e}).")
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Estimated homography is not invertible.") from exc
    if not np.isfinite(H_inv).all():
        raise ValueError("Inverse homography contains non-finite entries.")
    if abs(H_inv[2, 2]) > 1e-12:
        H_inv = H_inv / H_inv[2, 2]

    # Verify all inlier projections stay finite and on the same side of the line at infinity.
    src_h = np.hstack([src, np.ones((len(src), 1), dtype=np.float64)])
    proj_h = (H @ src_h.T).T
    w = proj_h[:, 2]
    if np.any(np.abs(w[inlier_mask]) <= 1e-9):
        raise ValueError("Estimated homography projects an inlier to infinity.")
    if not (np.all(w[inlier_mask] > 0) or np.all(w[inlier_mask] < 0)):
        raise ValueError(
            "Estimated homography flips projective orientation across inliers; check corner ordering."
        )

    proj_xy = proj_h[:, :2] / w[:, None]
    errors_m = np.linalg.norm(proj_xy - dst, axis=1)
    inlier_errors_m = errors_m[inlier_mask]

    dst_h = np.hstack([dst, np.ones((len(dst), 1), dtype=np.float64)])
    back_h = (H_inv @ dst_h.T).T
    back_uv = back_h[:, :2] / back_h[:, 2:3]
    pixel_errors = np.linalg.norm(back_uv - src, axis=1)
    inlier_pixel_errors = pixel_errors[inlier_mask]

    return HomographyFit(
        H=H,
        H_inv=H_inv,
        rmse_m=float(np.sqrt(np.mean(inlier_errors_m ** 2))),
        max_error_m=float(np.max(inlier_errors_m)),
        mean_error_m=float(np.mean(inlier_errors_m)),
        pixel_rmse_px=float(np.sqrt(np.mean(inlier_pixel_errors ** 2))),
        per_point_errors_m=[float(e) for e in errors_m],
        inliers=[bool(v) for v in inlier_mask],
        num_points=int(len(src)),
        method=method,
    )


def detect_checkerboard_corners(
    image: np.ndarray,
    inner_corners: tuple[int, int],
    *,
    roi: tuple[int, int, int, int] | None = None,
    origin_hint_px: tuple[float, float] | None = None,
) -> tuple[np.ndarray, str]:
    """Detect checkerboard inner corners ``(cols, rows)`` in an image.

    Returns ``(corners_Nx2, detector_method)`` in full-image pixel coordinates.
    """
    cols, rows = int(inner_corners[0]), int(inner_corners[1])
    if cols < 2 or rows < 2:
        raise ValueError(f"inner_corners must be >= (2, 2), got {inner_corners}.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    ox, oy = 0, 0
    work = gray
    if roi is not None:
        rx, ry, rw, rh = (int(v) for v in roi)
        ox, oy = rx, ry
        work = gray[ry : ry + rh, rx : rx + rw]

    pattern = (cols, rows)
    method = "sb"
    found, corners = cv2.findChessboardCornersSB(work, pattern, cv2.CALIB_CB_EXHAUSTIVE)
    if not found or corners is None:
        method = "classic"
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(work, pattern, flags)
        if found and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
            corners = cv2.cornerSubPix(work, corners, (7, 7), (-1, -1), criteria)

    if not found or corners is None:
        raise ValueError(f"Checkerboard ({cols}x{rows} inner corners) not detected in image.")

    pts = corners.reshape(-1, 2).astype(np.float64)
    pts[:, 0] += float(ox)
    pts[:, 1] += float(oy)

    if origin_hint_px is not None:
        hint = np.asarray(origin_hint_px, dtype=np.float64).reshape(2)
        if np.linalg.norm(pts[-1] - hint) < np.linalg.norm(pts[0] - hint):
            pts = pts[::-1].copy()

    return pts, method


def fit_homography_from_checkerboard(
    image: np.ndarray,
    inner_corners: tuple[int, int],
    square_size_m: float,
    *,
    board_origin_world_xy: tuple[float, float] = (0.0, 0.0),
    board_x_dir: tuple[float, float] = (1.0, 0.0),
    board_y_dir: tuple[float, float] = (0.0, 1.0),
    roi: tuple[int, int, int, int] | None = None,
    origin_hint_px: tuple[float, float] | None = None,
    ransac_reproj_threshold_m: float | None = None,
) -> tuple[HomographyFit, np.ndarray, np.ndarray, str]:
    """Detect a tabletop checkerboard and fit ``H_table`` in one shot (<1s)."""
    cols, rows = int(inner_corners[0]), int(inner_corners[1])
    sq = float(square_size_m)
    if sq <= 0.0:
        raise ValueError("square_size_m must be positive.")
    if origin_hint_px is None:
        raise ValueError(
            "origin_hint_px is required for fit_homography_from_checkerboard to "
            "disambiguate corner (0, 0) and prevent 180-degree-rotated tabletop mappings."
        )

    x_vec = np.asarray(board_x_dir, dtype=np.float64).reshape(2)
    y_vec = np.asarray(board_y_dir, dtype=np.float64).reshape(2)
    x_norm = float(np.linalg.norm(x_vec))
    y_norm = float(np.linalg.norm(y_vec))
    if x_norm <= 1e-9 or y_norm <= 1e-9:
        raise ValueError("board_x_dir and board_y_dir must be non-zero vectors.")
    x_hat = x_vec / x_norm
    y_hat = y_vec / y_norm

    pixels, det_method = detect_checkerboard_corners(
        image, (cols, rows), roi=roi, origin_hint_px=origin_hint_px
    )
    origin = np.asarray(board_origin_world_xy, dtype=np.float64).reshape(2)
    world_pts = np.empty((rows * cols, 2), dtype=np.float64)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            world_pts[idx] = origin + (c * sq) * x_hat + (r * sq) * y_hat
            idx += 1

    fit = fit_plane_homography(
        pixels, world_pts, ransac_reproj_threshold_m=ransac_reproj_threshold_m
    )
    return fit, pixels, world_pts, det_method


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def reanchor_plane(
    pixel_points: Sequence[Sequence[float]] | np.ndarray,
    world_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    z_table: float | None = None,
    calib_path: Path | str | None = None,
    max_error_m: float = DEFAULT_MAX_ERROR_M,
    allow_high_error: bool = False,
    ransac_reproj_threshold_m: float | None = None,
    method_label: str = "4-corner",
    save: bool = True,
) -> dict[str, Any]:
    """Fast ``<60s`` tabletop plane re-anchor utility.

    Fits ``H_table`` from 4+ overhead camera pixels ``(u, v)`` and known tabletop
    ``(x, y)`` coordinates (in meters), validates that reprojection error is within
    ``max_error_m`` (default 1 cm), and atomically writes ``H_table`` + anchor metadata
    into ``calib/rig_calib.json`` (preserving all other calibration fields).

    If ``world_points`` is omitted, reuses the previously stored ``table_anchors.world_points``
    from ``calib/rig_calib.json`` so repeat tripod/table re-anchors only need 4 new clicks.
    """
    t0 = time.monotonic()
    target_path = Path(calib_path) if calib_path is not None else resolve_default_calib_path()
    raw: dict[str, Any] = {}
    if target_path.exists():
        raw = json.loads(target_path.read_text(encoding="utf-8"))

    if world_points is None:
        saved_anchors = raw.get("table_anchors")
        if isinstance(saved_anchors, dict) and saved_anchors.get("world_points"):
            world_points = saved_anchors["world_points"]
        else:
            raise ValueError(
                "world_points not provided and no saved table_anchors.world_points found in "
                f"{target_path}. Pass 4+ tabletop (x, y) meter coordinates."
            )

    fit = fit_plane_homography(
        pixel_points,
        world_points,
        ransac_reproj_threshold_m=ransac_reproj_threshold_m,
    )

    within_tolerance = fit.max_error_m <= float(max_error_m)
    if not within_tolerance and not allow_high_error:
        raise ValueError(
            f"Homography max reprojection error {fit.max_error_m * 1000.0:.2f} mm exceeds "
            f"threshold {float(max_error_m) * 1000.0:.2f} mm (RMSE={fit.rmse_m * 1000.0:.2f} mm). "
            "Check corner ordering/clicks or pass allow_high_error=True."
        )

    effective_z = float(z_table) if z_table is not None else float(raw.get("z_table", 0.0))
    px_list = [[round(float(u), 3), round(float(v), 3)] for u, v in np.asarray(pixel_points, dtype=float)]
    w_list = [[round(float(x), 6), round(float(y), 6)] for x, y in np.asarray(world_points, dtype=float)]

    anchors_meta = {
        "method": method_label,
        "world_points": w_list,
        "pixel_points": px_list,
        "z_table": effective_z,
        "rmse_mm": round(fit.rmse_m * 1000.0, 3),
        "max_error_mm": round(fit.max_error_m * 1000.0, 3),
        "pixel_rmse_px": round(fit.pixel_rmse_px, 3),
        "num_points": fit.num_points,
        "num_inliers": int(sum(fit.inliers)),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if save:
        raw["H_table"] = fit.H.tolist()
        if z_table is not None or "z_table" not in raw:
            raw["z_table"] = effective_z
        raw["table_anchors"] = anchors_meta
        _atomic_write_json(target_path, raw)

    elapsed_s = time.monotonic() - t0
    return {
        "ok": True,
        "saved": bool(save),
        "calib_path": str(target_path),
        "z_table": effective_z,
        "within_tolerance": within_tolerance,
        "max_error_threshold_mm": round(float(max_error_m) * 1000.0, 3),
        "elapsed_s": round(elapsed_s, 4),
        "table_anchors": anchors_meta,
        **fit.to_dict(),
    }


def clear_plane_homography(
    calib_path: Path | str | None = None,
    *,
    keep_anchors: bool = True,
) -> dict[str, Any]:
    """Remove ``H_table`` override from ``calib/rig_calib.json`` to revert to ``T_world_cam_top``."""
    target_path = Path(calib_path) if calib_path is not None else resolve_default_calib_path()
    if not target_path.exists():
        return {"ok": True, "removed": False, "calib_path": str(target_path)}
    raw = json.loads(target_path.read_text(encoding="utf-8"))
    removed = "H_table" in raw
    raw.pop("H_table", None)
    if not keep_anchors:
        raw.pop("table_anchors", None)
    if removed or not keep_anchors:
        _atomic_write_json(target_path, raw)
    return {"ok": True, "removed": removed, "calib_path": str(target_path)}


def draw_reanchor_overlay(
    image: np.ndarray,
    fit: HomographyFit,
    pixel_points: Sequence[Sequence[float]] | np.ndarray,
    world_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    test_points: Sequence[tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> np.ndarray:
    """Render an annotated verification image with anchor corners, projected table grid, and residuals."""
    vis = image.copy()
    px = np.asarray(pixel_points, dtype=np.float64)
    wp = np.asarray(world_points, dtype=np.float64)

    # Draw subtle metric grid inside the anchor bounding box
    xmin, ymin = wp.min(axis=0)
    xmax, ymax = wp.max(axis=0)
    for gx in np.linspace(xmin, xmax, 5):
        p0 = fit.inverse_project(float(gx), float(ymin))
        p1 = fit.inverse_project(float(gx), float(ymax))
        cv2.line(vis, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), (180, 220, 120), 1)
    for gy in np.linspace(ymin, ymax, 5):
        p0 = fit.inverse_project(float(xmin), float(gy))
        p1 = fit.inverse_project(float(xmax), float(gy))
        cv2.line(vis, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), (180, 220, 120), 1)

    for idx, ((u, v), (wx, wy), err_m) in enumerate(zip(px, wp, fit.per_point_errors_m), start=1):
        pt = (int(round(u)), int(round(v)))
        color = (0, 220, 0) if err_m <= DEFAULT_MAX_ERROR_M else (0, 60, 255)
        cv2.circle(vis, pt, 7, color, 2)
        cv2.drawMarker(vis, pt, color, cv2.MARKER_CROSS, 14, 1)
        label = f"P{idx} ({wx:+.2f},{wy:+.2f})m {err_m * 1000.0:.1f}mm"
        cv2.putText(vis, label, (pt[0] + 8, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 2)
        cv2.putText(vis, label, (pt[0] + 8, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    if test_points:
        for idx, ((tu, tv), (tx, ty)) in enumerate(test_points, start=1):
            pt = (int(round(tu)), int(round(tv)))
            cv2.circle(vis, pt, 6, (255, 180, 0), 2)
            cv2.putText(
                vis,
                f"T{idx}->({tx:+.3f},{ty:+.3f})m",
                (pt[0] + 8, pt[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 180, 0),
                1,
            )

    summary = f"H_table ({fit.method}) | RMSE={fit.rmse_m * 1000.0:.2f}mm max={fit.max_error_m * 1000.0:.2f}mm"
    cv2.rectangle(vis, (8, 8), (520, 34), (20, 20, 20), -1)
    cv2.putText(vis, summary, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180), 1)
    return vis
