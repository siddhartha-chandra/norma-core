"""Zero-shot 3D object localization via Gemini 4 Hydrogen + fast OpenCV HSV/contour fallback.

Implements:
  - Primary: Gemini 4 Hydrogen (`client.interactions.create`) returning normalized
    `[y, x]` point + `box_2d` `[ymin, xmin, ymax, xmax]` (0-1000) + orientation `yaw_deg`.
  - Fallback: Automatic fast OpenCV HSV/contour detector that explicitly masks out
    ElRobot orange/coral servo horns and brackets when `GEMINI_API_KEY` is missing,
    `use_gemini=False`, or the network/API call fails or times out.
  - Metric projection: Converts normalized coordinates -> pixel `(u, v)` -> 3D base-frame
    `(x, y, z)` + `depth_m` + 6D pose via `CalibrationStore.pixel_to_world(...)`.
"""
from __future__ import annotations

import asyncio
import base64
import math
import os
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .calib_store import CalibrationStore

GEMINI_VISION_MODEL = os.environ.get("DESK_VISION_MODEL", "gemini-4-hydrogen").strip() or "gemini-4-hydrogen"

# Explicit HSV bounds for ElRobot 3D-printed / anodized orange & coral servo horns
ORANGE_HORN_LOWER_1 = np.array([2, 85, 65], dtype=np.uint8)
ORANGE_HORN_UPPER_1 = np.array([24, 255, 255], dtype=np.uint8)
ORANGE_HORN_LOWER_2 = np.array([168, 85, 65], dtype=np.uint8)
ORANGE_HORN_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

COLOR_HINTS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "red": (np.array([0, 100, 60], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
    "blue": (np.array([95, 70, 50], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8)),
    "green": (np.array([35, 60, 50], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
    "yellow": (np.array([22, 80, 80], dtype=np.uint8), np.array([36, 255, 255], dtype=np.uint8)),
    "purple": (np.array([130, 50, 50], dtype=np.uint8), np.array([165, 255, 255], dtype=np.uint8)),
    "pink": (np.array([145, 50, 80], dtype=np.uint8), np.array([175, 255, 255], dtype=np.uint8)),
}


class GeminiPoint2DResult(BaseModel):
    found: bool = Field(description="True if the requested object is visible in the image.")
    label: str = Field(description="Short identifying label for the matched object.")
    point: list[int] = Field(
        default_factory=lambda: [500, 500],
        description="Grasp/center point [y, x] normalized to 0-1000.",
    )
    box_2d: list[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="2D bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000.",
    )
    yaw_deg: float = Field(
        default=0.0,
        description="Principal axis orientation in image plane (degrees, -90 to 90).",
    )
    confidence: float = Field(
        default=0.0,
        description="Detection confidence between 0.0 and 1.0.",
    )


def _clamp_norm(val: float, lo: int = 0, hi: int = 1000) -> int:
    return max(lo, min(hi, int(round(val))))


def _decode_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode camera JPEG frame")
    return bgr


def build_orange_horn_mask(hsv: np.ndarray) -> np.ndarray:
    """Return binary mask (255 = orange robot servo horn/bracket) to exclude from targets."""
    m1 = cv2.inRange(hsv, ORANGE_HORN_LOWER_1, ORANGE_HORN_UPPER_1)
    m2 = cv2.inRange(hsv, ORANGE_HORN_LOWER_2, ORANGE_HORN_UPPER_2)
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, kernel, iterations=1)


def localize_opencv(bgr: np.ndarray, query: str) -> dict[str, Any]:
    """Fast local HSV + contour fallback, explicitly masking orange ElRobot servo horns."""
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    orange_mask = build_orange_horn_mask(hsv)
    q_low = query.lower()

    # Optional explicit color hint if query specifies a non-orange color
    hint_mask: np.ndarray | None = None
    for cname, (lo, hi) in COLOR_HINTS.items():
        if cname in q_low:
            hint_mask = cv2.inRange(hsv, lo, hi)
            break

    if hint_mask is not None:
        fg_mask = cv2.bitwise_and(hint_mask, cv2.bitwise_not(orange_mask))
    else:
        # Estimate desk background from median Lab color & find salient foreground objects
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = np.median(lab.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(lab - bg_lab, axis=2)
        fg_mask = (dist > 22.0).astype(np.uint8) * 255
        # Also include high-saturation non-orange regions
        sat_mask = (hsv[:, :, 1] > 70).astype(np.uint8) * 255
        fg_mask = cv2.bitwise_or(fg_mask, sat_mask)
        fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(orange_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(64.0, 0.0012 * img_area)
    max_area = 0.45 * img_area
    candidates: list[tuple[float, np.ndarray]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if min_area <= area <= max_area:
            candidates.append((area, cnt))

    if not candidates:
        return {
            "found": False,
            "label": query,
            "point": [500, 500],
            "box_2d": [0, 0, 0, 0],
            "yaw_deg": 0.0,
            "confidence": 0.0,
            "orange_masked_px": int(np.count_nonzero(orange_mask)),
        }

    # Prefer largest clean foreground contour
    best_area, best_cnt = max(candidates, key=lambda item: item[0])
    M = cv2.moments(best_cnt)
    if M["m00"] > 1e-6:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        cx, cy = bx + 0.5 * bw, by + 0.5 * bh

    bx, by, bw, bh = cv2.boundingRect(best_cnt)
    rect = cv2.minAreaRect(best_cnt)
    (_, _), (rw, rh), angle = rect
    yaw_deg = float(angle + 90.0) if rw < rh else float(angle)
    while yaw_deg > 90.0:
        yaw_deg -= 180.0
    while yaw_deg < -90.0:
        yaw_deg += 180.0

    y_norm = _clamp_norm((cy / h) * 1000.0)
    x_norm = _clamp_norm((cx / w) * 1000.0)
    ymin = _clamp_norm((by / h) * 1000.0)
    xmin = _clamp_norm((bx / w) * 1000.0)
    ymax = _clamp_norm(((by + bh) / h) * 1000.0)
    xmax = _clamp_norm(((bx + bw) / w) * 1000.0)

    conf = min(0.85, max(0.35, 0.35 + 2.5 * (best_area / img_area)))
    return {
        "found": True,
        "label": f"{query} (opencv)",
        "point": [y_norm, x_norm],
        "box_2d": [ymin, xmin, ymax, xmax],
        "yaw_deg": round(yaw_deg, 2),
        "confidence": round(conf, 3),
        "orange_masked_px": int(np.count_nonzero(orange_mask)),
    }


def _call_gemini_interactions_sync(jpeg_bytes: bytes, query: str, model: str) -> dict[str, Any]:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    b64_jpeg = base64.b64encode(jpeg_bytes).decode("ascii")
    prompt = (
        f"Locate the desk item matching '{query}' in this robot workspace image. "
        "Ignore orange/coral robot arm links, servo horns, and gripper fingers unless explicitly queried. "
        "Return JSON with: found (bool), label (str), point ([y, x] grasp center normalized 0-1000), "
        "box_2d ([ymin, xmin, ymax, xmax] normalized 0-1000), yaw_deg (principal axis angle -90..90), "
        "and confidence (0.0..1.0). If not visible, set found=false."
    )
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "image", "data": b64_jpeg, "mime_type": "image/jpeg"},
            {"type": "text", "text": prompt},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": GeminiPoint2DResult.model_json_schema(),
        },
        store=False,
    )
    parsed = GeminiPoint2DResult.model_validate_json(interaction.output_text)
    pt = parsed.point if len(parsed.point) == 2 else [500, 500]
    box = parsed.box_2d if len(parsed.box_2d) == 4 else [0, 0, 0, 0]
    return {
        "found": bool(parsed.found),
        "label": parsed.label or query,
        "point": [_clamp_norm(pt[0]), _clamp_norm(pt[1])],
        "box_2d": [_clamp_norm(v) for v in box],
        "yaw_deg": float(parsed.yaw_deg),
        "confidence": float(parsed.confidence),
    }


async def localize_object_3d(
    jpeg_bytes: bytes,
    query: str,
    calib: CalibrationStore,
    camera: str = "top",
    q_rad: list[float] | None = None,
    z_m: float | None = None,
    use_gemini: bool = True,
    timeout_s: float = 6.0,
    model: str | None = None,
) -> dict[str, Any]:
    """Localize `query` in `jpeg_bytes` via Gemini 4 Hydrogen (with OpenCV fallback) and project to 3D."""
    bgr = _decode_bgr(jpeg_bytes)
    h, w = bgr.shape[:2]

    backend = "opencv-fallback"
    fallback_reason: str | None = None
    det: dict[str, Any] | None = None

    if use_gemini:
        chosen_model = model or GEMINI_VISION_MODEL
        try:
            det = await asyncio.wait_for(
                asyncio.to_thread(_call_gemini_interactions_sync, jpeg_bytes, query, chosen_model),
                timeout=timeout_s,
            )
            backend = chosen_model
        except Exception as exc:
            fallback_reason = str(exc) or exc.__class__.__name__
            det = None
    else:
        fallback_reason = "use_gemini=False"

    if det is None:
        det = await asyncio.to_thread(localize_opencv, bgr, query)
        backend = "opencv-fallback"

    y_norm, x_norm = det["point"]
    ymin, xmin, ymax, xmax = det["box_2d"]
    px = (x_norm / 1000.0) * w
    py = (y_norm / 1000.0) * h

    depth_m = calib.get_metric_depth(px, py, camera=camera, q_rad=q_rad)
    wx, wy, wz = calib.pixel_to_world(px, py, z_m=z_m, camera=camera, q_rad=q_rad)

    K = calib._K(camera)
    box_w_px = max(0.0, ((xmax - xmin) / 1000.0) * w)
    box_h_px = max(0.0, ((ymax - ymin) / 1000.0) * h)
    w_m = round((box_w_px / max(1.0, float(K[0, 0]))) * depth_m, 4)
    h_m = round((box_h_px / max(1.0, float(K[1, 1]))) * depth_m, 4)

    yaw_deg = float(det.get("yaw_deg", 0.0))
    half_yaw = math.radians(yaw_deg) * 0.5
    half_pitch = math.pi / 4.0
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    qx = -sp * sy
    qy = sp * cy
    qz = cp * sy
    qw = cp * cy

    return {
        "query": query,
        "found": bool(det["found"]),
        "backend": backend,
        "fallback_reason": fallback_reason,
        "camera": camera,
        "label": det["label"],
        "point_norm": [int(y_norm), int(x_norm)],
        "box_2d": [int(ymin), int(xmin), int(ymax), int(xmax)],
        "pixel": [round(px, 2), round(py, 2)],
        "image_size": [w, h],
        "world": {"x": wx, "y": wy, "z": wz},
        "pose_6d": {
            "x": wx,
            "y": wy,
            "z": wz,
            "qx": round(qx, 8),
            "qy": round(qy, 8),
            "qz": round(qz, 8),
            "qw": round(qw, 8),
        },
        "depth_m": depth_m,
        "size_m": [w_m, h_m],
        "confidence": round(float(det.get("confidence", 0.0)), 3),
    }
