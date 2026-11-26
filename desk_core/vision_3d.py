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
import json
import math
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from .calib_store import CalibrationStore

GEMINI_VISION_MODEL = os.environ.get("DESK_VISION_MODEL", "gemini-4-hydrogen").strip() or "gemini-4-hydrogen"
_scene_provider_lock = threading.Lock()

DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    "gemini": GEMINI_VISION_MODEL,
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-6.1-terra",
    "qwen": "qwen4-max",
    "deepseek": "deepseek-flash",
}

VISION_MODELS_BY_PROVIDER: dict[str, list[str]] = {
    "gemini": [
        "gemini-4-hydrogen",
        "gemini-4-argon",
        "gemini-robotics-er-2-preview",
        "gemini-3.8-pro",
        "gemini-3.8-flash",
    ],
    "anthropic": [
        "claude-fable-5-2",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    ],
    "openai": [
        "gpt-6.2-astra",
        "gpt-6.2-sol",
        "gpt-6.1-terra",
        "gpt-6-luna",
        "gpt-5",
        "gpt-4.1",
    ],
    "qwen": ["qwen4-max", "qwen3.8-max", "qwen3-vl-plus", "qwen-vl-max", "qwen2.5-vl-72b-instruct"],
    "deepseek": ["deepseek-flash", "deepseek-vl2", "deepseek-vl2-small"],
}

# Documented capability tiers (no fabricated benchmark percentages; use /api/vision/verify for live measured accuracy)
MODEL_RATINGS: dict[str, dict[str, Any]] = {
    "gemini-4-hydrogen": {"tier": "Robotics Pointing", "color": "green", "note": "Gemini 4 frontier vision"},
    "gemini-4-argon": {"tier": "Frontier Vision", "color": "green", "note": "Gemini 4 Argon multimodal"},
    "gemini-robotics-er-2-preview": {"tier": "Robotics Pointing", "color": "green", "note": "Embodied [y,x] pointing"},
    "gemini-3.8-pro": {"tier": "Frontier Vision", "color": "green", "note": "High-accuracy multimodal"},
    "gemini-3.8-flash": {"tier": "Fast Vision", "color": "amber", "note": "Low-latency multimodal"},
    "claude-fable-5-2": {"tier": "Frontier Vision", "color": "green", "note": "Flagship agentic vision"},
    "claude-opus-5-5": {"tier": "Frontier Vision", "color": "green", "note": "Strongest Claude 5.5"},
    "claude-sonnet-5-5": {"tier": "Balanced Vision", "color": "green", "note": "Speed + accuracy (5.5)"},
    "claude-haiku-5-5": {"tier": "Fast Vision", "color": "amber", "note": "Low-latency (5.5)"},
    "claude-opus-4-6": {"tier": "Legacy Vision", "color": "amber", "note": "Legacy 4.6 series"},
    "claude-sonnet-4-6": {"tier": "Legacy Vision", "color": "amber", "note": "Legacy 4.6 series"},
    "gpt-6.2-astra": {"tier": "Frontier Vision", "color": "green", "note": "Flagship reasoning + vision"},
    "gpt-6.2-sol": {"tier": "Balanced Vision", "color": "green", "note": "High efficiency (6.2)"},
    "gpt-6.1-terra": {"tier": "Visual Grounding", "color": "green", "note": "Computer-use & visual reasoning"},
    "gpt-6-luna": {"tier": "Fast Vision", "color": "amber", "note": "High-volume fast vision"},
    "gpt-5": {"tier": "Legacy Vision", "color": "amber", "note": "Retiring Dec 2026"},
    "gpt-4.1": {"tier": "Legacy Vision", "color": "amber", "note": "Deprecated 4.1 series"},
    "qwen4-max": {"tier": "Frontier Vision", "color": "green", "note": "Qwen 4 Max (GA Nov 2026)"},
    "qwen3.8-max": {"tier": "Frontier Vision", "color": "green", "note": "Top ERQA Qwen3.8 VL"},
    "qwen3-vl-plus": {"tier": "Balanced Vision", "color": "green", "note": "Qwen3 VL spatial grounding"},
    "qwen-vl-max": {"tier": "Balanced Vision", "color": "amber", "note": "High-res Qwen VL"},
    "qwen2.5-vl-72b-instruct": {"tier": "Open-Weight VL", "color": "amber", "note": "Qwen 2.5 VL 72B"},
    "deepseek-flash": {"tier": "Frontier Vision", "color": "green", "note": "DeepSeek-V4.1-Flash-1017 (Multimodal, 1M)"},
    "deepseek-vl2": {"tier": "VL2 Endpoint", "color": "amber", "note": "Requires VL2-compatible host"},
    "deepseek-vl2-small": {"tier": "Compact VL2", "color": "amber", "note": "Requires VL2-compatible host"},
    "opencv-fallback": {"tier": "Offline Heuristic", "color": "red", "note": "No semantic understanding"},
}

# Explicit HSV bounds for ElRobot 3D-printed / anodized orange & coral servo horns
ORANGE_HORN_LOWER_1 = np.array([5, 95, 70], dtype=np.uint8)
ORANGE_HORN_UPPER_1 = np.array([22, 255, 255], dtype=np.uint8)
ORANGE_HORN_LOWER_2 = np.array([170, 95, 70], dtype=np.uint8)
ORANGE_HORN_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

COLOR_HINTS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "red": (np.array([0, 100, 60], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
    "blue": (np.array([95, 70, 50], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8)),
    "green": (np.array([35, 60, 50], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
    "yellow": (np.array([22, 80, 80], dtype=np.uint8), np.array([36, 255, 255], dtype=np.uint8)),
    "purple": (np.array([130, 50, 50], dtype=np.uint8), np.array([165, 255, 255], dtype=np.uint8)),
    "pink": (np.array([145, 50, 80], dtype=np.uint8), np.array([175, 255, 255], dtype=np.uint8)),
    "brown": (np.array([8, 60, 30], dtype=np.uint8), np.array([24, 220, 160], dtype=np.uint8)),
}

# Semantic keyword -> color hint fallback for offline OpenCV detection
QUERY_COLOR_ALIASES: dict[str, str] = {
    "key": "blue",
    "fob": "blue",
    "marker": "red",
    "pen": "red",
    "mug": "brown",
    "cup": "brown",
    "coffee": "brown",
    "note": "yellow",
    "sticky": "yellow",
    "cube": "green",
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


class Scene2DResult(BaseModel):
    detections: list[GeminiPoint2DResult]


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


def _opencv_candidates(bgr: np.ndarray, query: str | None) -> tuple[list[tuple[float, np.ndarray]], int]:
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    orange_mask = build_orange_horn_mask(hsv)
    q_low = (query or "").lower()
    if query is None:
        brown_mask = cv2.inRange(hsv, *COLOR_HINTS["brown"])
        orange_mask = cv2.bitwise_and(orange_mask, cv2.bitwise_not(brown_mask))

    # Optional explicit color hint (or semantic prop alias) if query specifies a target
    hint_mask: np.ndarray | None = None
    for cname, (lo, hi) in COLOR_HINTS.items():
        if cname in q_low:
            hint_mask = cv2.inRange(hsv, lo, hi)
            break
    if hint_mask is None:
        for kw, cname in QUERY_COLOR_ALIASES.items():
            if kw in q_low and cname in COLOR_HINTS:
                lo, hi = COLOR_HINTS[cname]
                hint_mask = cv2.inRange(hsv, lo, hi)
                break

    if hint_mask is not None:
        # Don't strip brown mugs or red marker caps when explicitly queried
        if any(k in q_low for k in ("mug", "cup", "brown", "red", "marker", "pen")):
            fg_mask = hint_mask
        else:
            fg_mask = cv2.bitwise_and(hint_mask, cv2.bitwise_not(orange_mask))
    else:
        # Estimate desk background from median Lab color & find salient foreground objects
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = np.median(lab.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(lab - bg_lab, axis=2)
        fg_mask = (dist > 22.0).astype(np.uint8) * 255
        if query is not None:
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

    return candidates, int(np.count_nonzero(orange_mask))


def localize_opencv(bgr: np.ndarray, query: str) -> dict[str, Any]:
    """Fast local HSV + contour fallback, explicitly masking orange ElRobot servo horns."""
    candidates, orange_masked_px = _opencv_candidates(bgr, query)
    if not candidates:
        return {
            "found": False,
            "label": query,
            "point": [500, 500],
            "box_2d": [0, 0, 0, 0],
            "yaw_deg": 0.0,
            "confidence": 0.0,
            "orange_masked_px": orange_masked_px,
        }

    # Prefer largest clean foreground contour
    best_area, best_cnt = max(candidates, key=lambda item: item[0])
    return _opencv_detection(best_cnt, best_area, bgr.shape[:2], f"{query} (opencv)", orange_masked_px)


def _opencv_detection(
    contour: np.ndarray, area: float, shape: tuple[int, int], label: str, orange_masked_px: int,
) -> dict[str, Any]:
    h, w = shape
    M = cv2.moments(contour)
    if M["m00"] > 1e-6:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        bx, by, bw, bh = cv2.boundingRect(contour)
        cx, cy = bx + 0.5 * bw, by + 0.5 * bh

    bx, by, bw, bh = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
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

    conf = min(0.85, max(0.35, 0.35 + 2.5 * (area / (h * w))))
    return {
        "found": True,
        "label": label,
        "point": [y_norm, x_norm],
        "box_2d": [ymin, xmin, ymax, xmax],
        "yaw_deg": round(yaw_deg, 2),
        "confidence": round(conf, 3),
        "orange_masked_px": orange_masked_px,
    }


def _localize_scene_opencv(bgr: np.ndarray) -> list[dict[str, Any]]:
    candidates, orange_masked_px = _opencv_candidates(bgr, None)
    return [
        _opencv_detection(contour, area, bgr.shape[:2], "unidentified object (opencv)", orange_masked_px)
        for area, contour in candidates
    ]


def _vision_prompt(query: str | None) -> str:
    if query is None:
        return (
            "Locate each distinct visible desk item in this overhead robot workspace image. "
            "Ignore the table, arm, servo horns, and gripper. Do not guess at hidden items. "
            'Return ONLY a JSON object with a "detections" array, empty if no items are visible. '
            "For each detection provide found=true, a short identifying label including visible distinguishing details, "
            "point ([y, x] center normalized 0-1000), box_2d ([ymin, xmin, ymax, xmax] normalized 0-1000), "
            "yaw_deg (principal axis angle -90..90), and confidence (0.0..1.0)."
        )
    return (
        f"Locate the desk item matching '{query}' in this robot workspace image. "
        "Ignore orange/coral robot arm links, servo horns, and gripper fingers unless explicitly queried. "
        "Return ONLY valid JSON with keys: found (bool), label (str), point ([y, x] grasp center normalized 0-1000), "
        "box_2d ([ymin, xmin, ymax, xmax] normalized 0-1000), yaw_deg (principal axis angle -90..90), "
        "and confidence (0.0..1.0). If not visible, set found=false."
    )


def _parse_vlm_json(raw_text: str, query: str | None) -> list[dict[str, Any]]:
    txt = raw_text.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if m:
        txt = m.group(0)
    if query is None:
        items = json.loads(txt).get("detections")
        if not isinstance(items, list):
            raise ValueError("Scene vision response must contain a detections array.")
        detections = []
        has_valid_item = False
        for item in items:
            try:
                det = GeminiPoint2DResult.model_validate(item)
            except ValidationError:
                continue
            if not det.found:
                has_valid_item = True
                continue
            if (
                det.label.strip()
                and len(det.point) == 2 and len(det.box_2d) == 4
                and {"point", "box_2d"} <= det.model_fields_set
                and math.isfinite(det.yaw_deg) and math.isfinite(det.confidence)
            ):
                detections.append(_normalize_detection(det, det.label.strip()))
                has_valid_item = True
        if items and not has_valid_item:
            raise ValueError("Scene vision response had no valid detections.")
        return detections
    parsed = GeminiPoint2DResult.model_validate_json(txt)
    return [_normalize_detection(parsed, parsed.label or query)]


def _normalize_detection(parsed: GeminiPoint2DResult, label: str) -> dict[str, Any]:
    pt = parsed.point if len(parsed.point) == 2 else [500, 500]
    box = parsed.box_2d if len(parsed.box_2d) == 4 else [0, 0, 0, 0]
    return {
        "found": bool(parsed.found),
        "label": label,
        "point": [_clamp_norm(pt[0]), _clamp_norm(pt[1])],
        "box_2d": [_clamp_norm(v) for v in box],
        "yaw_deg": float(parsed.yaw_deg),
        "confidence": float(parsed.confidence),
    }


def _load_keystore() -> dict[str, str]:
    env_path = os.environ.get("DESK_KEYS_PATH", "").strip()
    path = Path(env_path) if env_path else (Path.home() / ".config" / "desk-core" / "keys.json")
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, str] = {}
        for k, v in raw.items():
            val = str(v).strip()
            if not val:
                continue
            norm_k = "anthropic" if k == "claude" else k.strip().lower()
            out[norm_k] = val
        return out
    except Exception:
        return {}


def _get_api_key(provider: str) -> str:
    p = "anthropic" if provider == "claude" else provider.strip().lower()
    env_map = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "qwen": ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
        "deepseek": ("DEEPSEEK_API_KEY",),
    }
    for var in env_map.get(p, ()):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return _load_keystore().get(p, "")


def _call_gemini_interactions_sync(jpeg_bytes: bytes, query: str | None, model: str) -> list[dict[str, Any]]:
    from google import genai

    api_key = _get_api_key("gemini")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    b64_jpeg = base64.b64encode(jpeg_bytes).decode("ascii")
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "image", "data": b64_jpeg, "mime_type": "image/jpeg"},
            {"type": "text", "text": _vision_prompt(query)},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": (Scene2DResult if query is None else GeminiPoint2DResult).model_json_schema(),
        },
        store=False,
    )
    return _parse_vlm_json(interaction.output_text, query)


def _call_anthropic_sync(jpeg_bytes: bytes, query: str | None, model: str) -> list[dict[str, Any]]:
    api_key = _get_api_key("anthropic")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    b64_jpeg = base64.b64encode(jpeg_bytes).decode("ascii")
    payload = {
        "model": model,
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_jpeg},
                    },
                    {"type": "text", "text": _vision_prompt(query)},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {body[:200]}") from exc
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return _parse_vlm_json(text, query)


def _call_openai_compat_sync(
    jpeg_bytes: bytes,
    query: str | None,
    model: str,
    api_key: str,
    base_url: str,
    provider_label: str,
) -> list[dict[str, Any]]:
    if not api_key:
        raise RuntimeError(f"{provider_label} API key not set")
    b64_jpeg = base64.b64encode(jpeg_bytes).decode("ascii")
    payload = {
        "model": model,
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}},
                    {"type": "text", "text": _vision_prompt(query)},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider_label} HTTP {exc.code}: {body[:200]}") from exc
    text = data["choices"][0]["message"]["content"]
    return _parse_vlm_json(text, query)


def _resolve_provider(provider: str) -> str:
    p = (provider or "auto").strip().lower()
    if p == "claude":
        return "anthropic"
    if p != "auto":
        return p
    env_p = os.environ.get("DESK_VISION_PROVIDER", "").strip().lower()
    if env_p:
        return "anthropic" if env_p == "claude" else env_p
    for cand in ("gemini", "anthropic", "openai", "qwen", "deepseek"):
        if _get_api_key(cand):
            return cand
    return "gemini"


def _call_vlm_sync(jpeg_bytes: bytes, query: str | None, provider: str, model: str) -> list[dict[str, Any]]:
    if provider == "gemini":
        return _call_gemini_interactions_sync(jpeg_bytes, query, model)
    if provider == "anthropic":
        return _call_anthropic_sync(jpeg_bytes, query, model)
    if provider == "openai":
        key = _get_api_key("openai")
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        return _call_openai_compat_sync(jpeg_bytes, query, model, key, url, "OpenAI")
    if provider == "qwen":
        key = _get_api_key("qwen")
        default_qwen_url = (
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
            if key.startswith("sk-sp-")
            else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        url = os.environ.get("QWEN_BASE_URL", default_qwen_url).rstrip("/") + "/chat/completions"
        return _call_openai_compat_sync(jpeg_bytes, query, model, key, url, "Qwen")
    if provider == "deepseek":
        key = _get_api_key("deepseek")
        url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/") + "/chat/completions"
        return _call_openai_compat_sync(jpeg_bytes, query, model, key, url, "DeepSeek")
    raise ValueError(f"Unsupported vision provider: {provider!r}")


def _call_scene_vlm_sync(jpeg_bytes: bytes, query: str | None, provider: str, model: str) -> list[dict[str, Any]]:
    if not _scene_provider_lock.acquire(blocking=False):
        raise RuntimeError("Previous scene vision request is still running.")
    try:
        return _call_vlm_sync(jpeg_bytes, query, provider, model)
    finally:
        _scene_provider_lock.release()


async def localize_object_3d(
    jpeg_bytes: bytes,
    query: str,
    calib: CalibrationStore,
    camera: str = "top",
    q_rad: list[float] | None = None,
    z_m: float | None = None,
    use_gemini: bool = True,
    timeout_s: float = 12.0,
    model: str | None = None,
    provider: str = "auto",
) -> dict[str, Any]:
    """Localize `query` in `jpeg_bytes` via Gemini/Anthropic/OpenAI/Qwen/DeepSeek (with OpenCV fallback) and project to 3D."""
    return (await localize_frame_3d(
        jpeg_bytes, query, calib, camera=camera, q_rad=q_rad, z_m=z_m,
        use_gemini=use_gemini, timeout_s=timeout_s, model=model, provider=provider,
    ))[0]


async def localize_frame_3d(
    jpeg_bytes: bytes,
    query: str | None,
    calib: CalibrationStore,
    camera: str = "top",
    q_rad: list[float] | None = None,
    z_m: float | None = None,
    use_gemini: bool = True,
    timeout_s: float = 12.0,
    model: str | None = None,
    provider: str = "auto",
) -> list[dict[str, Any]]:
    """Localize a requested item or, without a query, all detected items in one frame."""
    bgr = _decode_bgr(jpeg_bytes)

    backend = "opencv-fallback"
    fallback_reason: str | None = None
    detections: list[dict[str, Any]] | None = None

    resolved_provider = _resolve_provider(provider)
    if use_gemini and resolved_provider != "opencv":
        chosen_model = model or DEFAULT_PROVIDER_MODELS.get(resolved_provider, GEMINI_VISION_MODEL)
        allowed = VISION_MODELS_BY_PROVIDER.get(resolved_provider)
        if allowed and model and model not in allowed:
            raise ValueError(f"Model {model!r} is not in vision-capable list for {resolved_provider}: {allowed}")
        call_provider = _call_scene_vlm_sync if query is None else _call_vlm_sync
        try:
            detections = await asyncio.wait_for(
                asyncio.to_thread(call_provider, jpeg_bytes, query, resolved_provider, chosen_model),
                timeout=timeout_s,
            )
            backend = chosen_model if resolved_provider == "gemini" else f"{resolved_provider}:{chosen_model}"
        except Exception as exc:
            fallback_reason = str(exc) or exc.__class__.__name__
    else:
        fallback_reason = "opencv_only" if resolved_provider == "opencv" else "use_gemini=False"

    if detections is None:
        if query is None:
            detections = await asyncio.to_thread(_localize_scene_opencv, bgr)
        else:
            detections = [await asyncio.to_thread(localize_opencv, bgr, query)]

    return [
        _project_detection(det, query, bgr.shape[:2], calib, camera, q_rad, z_m, backend, fallback_reason)
        for det in detections
    ]


def _project_detection(
    det: dict[str, Any], query: str | None, shape: tuple[int, int], calib: CalibrationStore,
    camera: str, q_rad: list[float] | None, z_m: float | None, backend: str, fallback_reason: str | None,
) -> dict[str, Any]:
    h, w = shape
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
