"""Robo-Jev <150 ms System-1 Reflex Gatekeeper (DESK-CORE issue #12).

Provides deterministic, sub-150 ms CPU safety checks on workspace frames,
registered obstacles, and ST3215 jaw telemetry:
  - `target_shifted`: detects sideways target displacement (> threshold) via
    template cross-correlation + ROI delta, immediately calling `bridge.abort()`.
  - `grasp_secure`: scores jaw closure slack + motor current draw.
  - `will_tip`: checks proximity to fragile obstacles (e.g. mug) / high-aspect targets.
  - `path_clear`: checks 2D EE-to-target segment against registered SceneObstacles.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .calib_store import CalibrationStore
from .interfaces import SceneObstacle


def _decode_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid JPEG frame")
    return bgr


@dataclass
class WorkspaceBounds:
    """Safe 3D tabletop bounding box (meters in robot base_link) for target recalibration."""
    x_min: float = 0.08
    x_max: float = 0.44
    y_min: float = -0.26
    y_max: float = 0.26
    z_min: float = -0.02
    z_max: float = 0.35

    def contains(self, x: float, y: float, z: float = 0.0) -> bool:
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass
class ReflexWatchTarget:
    """Reference snapshot around a localized target for mid-flight shift detection."""
    query: str
    ref_pixel: tuple[float, float]
    ref_crop_gray: np.ndarray
    box_px: tuple[int, int, int, int]  # x1, y1, x2, y2
    depth_m: float = 0.62
    created_at: float = field(default_factory=time.monotonic)


class RoboJevGatekeeper:
    """Sub-150 ms visual + geometric reflex gatekeeper."""

    def __init__(
        self,
        shift_px_threshold: float = 18.0,
        shift_mm_threshold: float = 22.0,
        safe_bounds: WorkspaceBounds | None = None,
    ) -> None:
        self.shift_px_threshold = float(shift_px_threshold)
        self.shift_mm_threshold = float(shift_mm_threshold)
        self.safe_bounds = safe_bounds or WorkspaceBounds()
        self.watch: ReflexWatchTarget | None = None
        self.last_report: dict[str, Any] = {
            "ok": True,
            "target_shifted": False,
            "path_clear": 1.0,
            "grasp_secure": 1.0,
            "will_tip": 0.0,
            "shift_px": 0.0,
            "shift_mm": 0.0,
            "latency_ms": 0.0,
            "reason": "idle",
        }
        self.events: deque[dict[str, Any]] = deque(maxlen=50)

    def arm_watch(
        self,
        jpeg_bytes: bytes,
        pixel: tuple[float, float],
        query: str = "target",
        depth_m: float = 0.62,
        half_crop: int = 36,
    ) -> dict[str, Any]:
        """Capture a reference grayscale crop around `pixel` for `<150 ms` shift tracking."""
        bgr = _decode_bgr(jpeg_bytes)
        h, w = bgr.shape[:2]
        cx = max(half_crop, min(w - half_crop - 1, int(round(pixel[0]))))
        cy = max(half_crop, min(h - half_crop - 1, int(round(pixel[1]))))
        x1, y1 = max(0, cx - half_crop), max(0, cy - half_crop)
        x2, y2 = min(w, cx + half_crop), min(h, cy + half_crop)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        crop = gray[y1:y2, x1:x2].copy()
        self.watch = ReflexWatchTarget(
            query=query,
            ref_pixel=(float(pixel[0]), float(pixel[1])),
            ref_crop_gray=crop,
            box_px=(x1, y1, x2, y2),
            depth_m=float(depth_m),
        )
        return {
            "armed": True,
            "query": query,
            "ref_pixel": [round(float(pixel[0]), 1), round(float(pixel[1]), 1)],
            "crop_shape": list(crop.shape),
        }

    def clear_watch(self) -> None:
        self.watch = None

    def check_target_shifted(
        self,
        curr_jpeg: bytes,
        calib: CalibrationStore | None = None,
        camera: str = "top",
        search_pad: int = 110,
    ) -> dict[str, Any]:
        """Measure target displacement against armed reference crop in <150 ms CPU time."""
        t0 = time.perf_counter()
        if self.watch is None:
            return {"shifted": False, "shift_px": 0.0, "shift_mm": 0.0, "latency_ms": 0.0, "armed": False}

        bgr = _decode_bgr(curr_jpeg)
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = self.watch.box_px
        crop = self.watch.ref_crop_gray
        ch, cw = crop.shape[:2]

        # Original ROI absolute delta
        roi_curr = gray[y1:y2, x1:x2]
        mad_roi = float(np.mean(np.abs(roi_curr.astype(np.float32) - crop.astype(np.float32)))) if roi_curr.shape == crop.shape else 0.0

        # Search window around original ROI using normalized cross-correlation
        sx1 = max(0, x1 - search_pad)
        sy1 = max(0, y1 - search_pad)
        sx2 = min(w, x2 + search_pad)
        sy2 = min(h, y2 + search_pad)
        search = gray[sy1:sy2, sx1:sx2]

        shift_px = 0.0
        match_score = 1.0
        new_px = [self.watch.ref_pixel[0], self.watch.ref_pixel[1]]

        if search.shape[0] > ch + 2 and search.shape[1] > cw + 2 and float(np.std(crop)) > 2.0:
            res = cv2.matchTemplate(search, crop, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            match_score = float(max_val)
            matched_cx = sx1 + max_loc[0] + 0.5 * cw
            matched_cy = sy1 + max_loc[1] + 0.5 * ch
            dx = matched_cx - (x1 + 0.5 * cw)
            dy = matched_cy - (y1 + 0.5 * ch)
            shift_px = float(math.hypot(dx, dy))
            new_px = [round(self.watch.ref_pixel[0] + dx, 1), round(self.watch.ref_pixel[1] + dy, 1)]

        fx = float(calib._K(camera)[0, 0]) if calib is not None else 560.0
        shift_mm = (shift_px / max(1.0, fx)) * self.watch.depth_m * 1000.0

        # Trigger if object slid noticeably OR vanished from the ROI (low correlation + high pixel change)
        vanished = mad_roi > 24.0 and match_score < 0.45
        shifted = (shift_px >= self.shift_px_threshold) or (shift_mm >= self.shift_mm_threshold) or vanished
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "armed": True,
            "shifted": bool(shifted),
            "vanished": bool(vanished),
            "shift_px": round(shift_px, 2),
            "shift_mm": round(shift_mm, 2),
            "match_score": round(match_score, 3),
            "mad_roi": round(mad_roi, 2),
            "new_pixel": new_px,
            "latency_ms": latency_ms,
        }

    def check_path_clear(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        obstacles: list[SceneObstacle],
        ee_radius_m: float = 0.03,
    ) -> tuple[float, str | None]:
        """Return (clearance_score 0..1, blocking_obstacle_name) for 2D path segment."""
        if not obstacles:
            return 1.0, None
        ax, ay = start_xy
        bx, by = goal_xy
        abx, aby = bx - ax, by - ay
        ab_len2 = abx * abx + aby * aby
        min_margin = 1.0
        blocker: str | None = None
        for obs in obstacles:
            if ab_len2 <= 1e-9:
                dist = math.hypot(obs.x - ax, obs.y - ay)
            else:
                t = max(0.0, min(1.0, ((obs.x - ax) * abx + (obs.y - ay) * aby) / ab_len2))
                px = ax + t * abx
                py = ay + t * aby
                dist = math.hypot(obs.x - px, obs.y - py)
            margin = dist - (obs.radius_m + ee_radius_m)
            if margin < min_margin:
                min_margin = margin
                if margin < 0.0:
                    blocker = obs.name
        score = max(0.0, min(1.0, (min_margin + 0.02) / 0.08))
        return round(score, 3), blocker

    def evaluate(
        self,
        curr_jpeg: bytes | None = None,
        calib: CalibrationStore | None = None,
        obstacles: list[SceneObstacle] | None = None,
        target_xy: tuple[float, float] | None = None,
        start_xy: tuple[float, float] = (0.15, 0.0),
        gripper_current_ma: int = 0,
        gripper_holding: bool | None = None,
        camera: str = "top",
    ) -> dict[str, Any]:
        """Run all 4 Robo-Jev checks (`target_shifted`, `path_clear`, `will_tip`, `grasp_secure`)."""
        t0 = time.perf_counter()
        shift_info = (
            self.check_target_shifted(curr_jpeg, calib=calib, camera=camera)
            if (curr_jpeg is not None and self.watch is not None)
            else {"shifted": False, "shift_px": 0.0, "shift_mm": 0.0, "latency_ms": 0.0, "armed": False}
        )

        obs_list = obstacles or []
        if target_xy is not None:
            path_score, blocker = self.check_path_clear(start_xy, target_xy, obs_list)
            tip_risk = 0.0
            for o in obs_list:
                d = math.hypot(o.x - target_xy[0], o.y - target_xy[1])
                if d < o.radius_m + 0.045:
                    tip_risk = max(tip_risk, round(min(1.0, (o.radius_m + 0.045 - d) / 0.045), 3))
        else:
            path_score, blocker, tip_risk = 1.0, None, 0.0

        if gripper_holding is True:
            grasp_score = 0.95
        elif gripper_holding is False:
            grasp_score = 0.25
        else:
            grasp_score = round(min(1.0, 0.5 + abs(gripper_current_ma) / 600.0), 3)

        ok = (not shift_info["shifted"]) and (blocker is None) and (tip_risk < 0.75)
        reason = "nominal"
        if shift_info["shifted"]:
            reason = "target_shifted"
        elif blocker is not None:
            reason = f"path_blocked:{blocker}"
        elif tip_risk >= 0.75:
            reason = "will_tip"

        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        report = {
            "ok": ok,
            "reason": reason,
            "target_shifted": bool(shift_info["shifted"]),
            "shift_px": shift_info["shift_px"],
            "shift_mm": shift_info["shift_mm"],
            "new_pixel": shift_info.get("new_pixel"),
            "path_clear": path_score,
            "blocker": blocker,
            "will_tip": tip_risk,
            "grasp_secure": grasp_score,
            "latency_ms": total_ms,
            "armed": bool(shift_info.get("armed", False)),
        }
        self.last_report = report
        if not ok:
            self.events.append({"ts": time.time(), **report})
        return report


async def check_reflex_and_abort(
    gatekeeper: RoboJevGatekeeper,
    bridge: Any,
    calib: CalibrationStore | None = None,
    obstacles: list[SceneObstacle] | None = None,
    target_xy: tuple[float, float] | None = None,
    camera: str = "top",
) -> dict[str, Any]:
    """Fetch a single fresh frame, run Robo-Jev in <150 ms, and call `bridge.abort()` on hazard."""
    t0 = time.perf_counter()
    jpeg = await bridge.get_frame(camera, denoise=False)
    report = gatekeeper.evaluate(
        curr_jpeg=jpeg,
        calib=calib,
        obstacles=obstacles,
        target_xy=target_xy,
        camera=camera,
    )
    aborted = False
    abort_res: dict[str, Any] | None = None
    if not report["ok"]:
        abort_res = await bridge.abort()
        aborted = True
    report["aborted"] = aborted
    report["abort_result"] = abort_res
    report["total_roundtrip_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return report


async def guarded_recover(
    gatekeeper: RoboJevGatekeeper,
    bridge: Any,
    calib: CalibrationStore,
    query: str = "target",
    obstacles_fn: Any = None,
    target_xy: tuple[float, float] | None = None,
    camera: str = "top",
    obstacle_wait_timeout_s: float = 8.0,
    poll_interval_s: float = 0.1,
    use_gemini: bool = False,
) -> dict[str, Any]:
    """Pause on obstacle until clear & resume; recalibrate on shift if inside safe bounds.

    Returns:
      - `ok=True, action="proceed"|"resumed_after_obstacle"|"recalibrated"` when safe to continue
      - `ok=False, action="halted", reason="obstacle_timeout"|"target_out_of_bounds"` otherwise
    """
    import asyncio
    from .vision_3d import localize_object_3d

    def _get_obs() -> list[SceneObstacle]:
        return list(obstacles_fn()) if callable(obstacles_fn) else list(obstacles_fn or [])

    jpeg = await bridge.get_frame(camera, denoise=False)
    report = gatekeeper.evaluate(
        curr_jpeg=jpeg,
        calib=calib,
        obstacles=_get_obs(),
        target_xy=target_xy,
        camera=camera,
    )

    if report["ok"]:
        return {"ok": True, "action": "proceed", "target_xy": list(target_xy) if target_xy else None, "report": report}

    # Immediate safe pause
    await bridge.abort()

    # Case 1: Transient obstacle in path -> pause and poll until obstacle moves out
    if report.get("blocker") is not None or report.get("reason") == "will_tip":
        deadline = time.monotonic() + float(obstacle_wait_timeout_s)
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval_s)
            jpeg = await bridge.get_frame(camera, denoise=False)
            rep_retry = gatekeeper.evaluate(
                curr_jpeg=jpeg,
                calib=calib,
                obstacles=_get_obs(),
                target_xy=target_xy,
                camera=camera,
            )
            if rep_retry["ok"]:
                if hasattr(bridge, "rearm"):
                    bridge.rearm()
                return {
                    "ok": True,
                    "action": "resumed_after_obstacle",
                    "target_xy": list(target_xy) if target_xy else None,
                    "report": rep_retry,
                }
        return {
            "ok": False,
            "action": "halted",
            "reason": "obstacle_timeout",
            "report": report,
        }

    # Case 2: Target shifted -> recalibrate & verify inside safe boundary bounding box
    if report.get("target_shifted"):
        det = await localize_object_3d(
            jpeg_bytes=jpeg,
            query=query,
            calib=calib,
            camera=camera,
            use_gemini=use_gemini,
        )
        if det.get("found"):
            new_px = (float(det["pixel"][0]), float(det["pixel"][1]))
            wx, wy, wz = float(det["world"]["x"]), float(det["world"]["y"]), float(det["world"]["z"])
            depth_m = float(det["depth_m"])
        elif report.get("new_pixel"):
            new_px = (float(report["new_pixel"][0]), float(report["new_pixel"][1]))
            wx, wy, wz = calib.pixel_to_world(new_px[0], new_px[1], camera=camera)
            depth_m = calib.get_metric_depth(new_px[0], new_px[1], camera=camera)
        else:
            return {"ok": False, "action": "halted", "reason": "target_lost", "report": report}

        if not gatekeeper.safe_bounds.contains(wx, wy, wz):
            return {
                "ok": False,
                "action": "halted",
                "reason": "target_out_of_bounds",
                "world": {"x": wx, "y": wy, "z": wz},
                "safe_bounds": vars(gatekeeper.safe_bounds),
                "report": report,
            }

        gatekeeper.arm_watch(jpeg_bytes=jpeg, pixel=new_px, query=query, depth_m=depth_m)
        if hasattr(bridge, "rearm"):
            bridge.rearm()
        return {
            "ok": True,
            "action": "recalibrated",
            "pixel": [round(new_px[0], 1), round(new_px[1], 1)],
            "world": {"x": wx, "y": wy, "z": wz},
            "target_xy": [wx, wy],
            "report": report,
        }

    return {"ok": False, "action": "halted", "reason": report.get("reason", "hazard"), "report": report}
