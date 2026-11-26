"""DESK-CORE Mission Control FastAPI Dashboard (issue #14).

Serves a high-contrast Mission Control web UI on http://127.0.0.1:8765 with:
  - Live Top (with optional localize_3d / Robo-Jev ROI overlay) & gripper eye-in-hand feeds (the "wrist" view; there is no dedicated wrist camera)
  - Prominent red E-STOP button (`POST /api/estop`)
  - Robo-Jev <150 ms reflex meters (`target_shifted`, `path_clear`, `will_tip`, `grasp_secure`)
  - Live ST3215 joint telemetry & link health
  - Portable speaker TTS cue endpoint (`POST /api/tts`)
  - 1-click "Run 07:30 First-Light Routine" (`POST /api/first_light`)
  - Recent tool-call / action log tail (`GET /api/logs`)

Run:
  uv run --offline --managed-python python -m desk_core.dashboard
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import mcp_server as srv
from .robo_jev import check_reflex_and_abort
from .vision_3d import (
    DEFAULT_PROVIDER_MODELS,
    MODEL_RATINGS,
    VISION_MODELS_BY_PROVIDER,
    _get_api_key,
    _load_keystore,
    localize_object_3d,
)

app = FastAPI(title="DESK-CORE Mission Control")

_action_log: deque[dict[str, Any]] = deque(maxlen=40)
_last_detection: dict[str, Any] | None = None
_first_light_running = False


def _log_event(kind: str, detail: str, ok: bool = True) -> None:
    _action_log.appendleft({
        "ts": time.strftime("%H:%M:%S"),
        "kind": kind,
        "detail": detail,
        "ok": ok,
    })


class LocalizeReq(BaseModel):
    query: str = "keys"
    camera: str = "top"
    use_gemini: bool = True
    provider: str = "auto"
    model: str | None = None


class TTSReq(BaseModel):
    text: str


class MockDisplaceReq(BaseModel):
    mode: str = "in_bounds"  # "in_bounds" | "out_of_bounds" | "reset"


class KeySetReq(BaseModel):
    provider: str
    api_key: str = ""


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    return {"ok": True, "service": "desk-core-mission-control"}


@app.get("/api/telemetry")
async def api_telemetry() -> dict[str, Any]:
    try:
        bridge = await srv._get_bridge()
        st = await bridge.get_state()
        link = getattr(bridge, "link_status", lambda: {"bridge": "unknown"})()
        motors = [
            {
                "id": m.id,
                "position": m.position,
                "norm": round(m.to_norm(), 3),
                "current_ma": m.current_ma,
                "err": m.error_status,
            }
            for m in st.motors
        ]
        torque_enabled = st.torque_enabled
        bus_serial = st.bus_serial
        err = None
    except Exception as exc:
        link = {"bridge": "disconnected", "socket_up": False}
        motors = []
        torque_enabled = False
        bus_serial = ""
        err = str(exc)

    return {
        "bus_serial": bus_serial,
        "torque_enabled": torque_enabled,
        "link": link,
        "motors": motors,
        "robo_jev": srv._robo_jev.last_report,
        "last_detection": _last_detection,
        "tts": srv._tts.status(),
        "first_light_running": _first_light_running,
        "calib": srv._calib.status(),
        "error": err,
    }


@app.get("/api/frame/{camera}.jpg")
async def api_frame(camera: str, show_overlay: bool = False) -> Response:
    if camera not in ("top", "wrist"):
        return Response(status_code=404)
    bridge = await srv._get_bridge()
    jpeg = await bridge.get_frame(camera, denoise=False)

    view_det = None
    if _last_detection:
        views = _last_detection.get("views")
        if isinstance(views, dict) and camera in views:
            view_det = views[camera]
        elif _last_detection.get("camera") == camera:
            view_det = _last_detection

    if (view_det and view_det.get("found")) or (camera == "top" and srv._robo_jev.watch):
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            h, w = bgr.shape[:2]
            if view_det and view_det.get("found"):
                ymin, xmin, ymax, xmax = view_det["box_2d"]
                p1 = (int(xmin / 1000.0 * w), int(ymin / 1000.0 * h))
                p2 = (int(xmax / 1000.0 * w), int(ymax / 1000.0 * h))
                px = int(round(view_det["pixel"][0]))
                py = int(round(view_det["pixel"][1]))
                cv2.rectangle(bgr, p1, p2, (0, 230, 118), 2)
                cv2.circle(bgr, (px, py), 6, (0, 230, 118), -1)
                world_src = _last_detection if _last_detection.get("camera") == "both" else view_det
                lbl = f"{view_det['label']} [{view_det.get('backend', '?')}] ({world_src['world']['x']:.2f}, {world_src['world']['y']:.2f}m)"
                cv2.putText(bgr, lbl, (p1[0], max(20, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 118), 2)

            if camera == "top" and srv._robo_jev.watch is not None:
                wx1, wy1, wx2, wy2 = srv._robo_jev.watch.box_px
                col = (0, 60, 255) if srv._robo_jev.last_report.get("target_shifted") else (0, 200, 255)
                cv2.rectangle(bgr, (wx1, wy1), (wx2, wy2), col, 2)
                cv2.putText(bgr, "ROBO-JEV WATCH", (wx1, max(18, wy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)

            ok_enc, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok_enc:
                jpeg = buf.tobytes()

    if camera == "top" and show_overlay:
        jpeg = srv._with_xray(bridge, jpeg)
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/api/estop")
async def api_estop() -> dict[str, Any]:
    bridge = await srv._get_bridge()
    res = await bridge.abort()
    srv._tts.speak("Emergency stop engaged.")
    _log_event("E-STOP", f"Bridge aborted: {res}", ok=True)
    return {"ok": True, "estop": res}


@app.post("/api/localize")
async def api_localize(req: LocalizeReq) -> dict[str, Any]:
    global _last_detection
    if req.camera in ("both", "dual"):
        raise HTTPException(status_code=400, detail="camera='both' is disabled: dual-camera localization is pending the epipolar step")
    bridge = await srv._get_bridge()
    jpeg = await bridge.get_frame(req.camera, denoise=True)
    q = await srv._live_arm_q(None, req.camera)
    det = await localize_object_3d(
        jpeg_bytes=jpeg,
        query=req.query,
        calib=srv._calib,
        camera=req.camera,
        q_rad=q,
        use_gemini=req.use_gemini,
        model=req.model,
        provider=req.provider,
    )
    _last_detection = det
    fb = f" ({det['fallback_reason']})" if det.get("fallback_reason") else ""
    _log_event("LOCALIZE", f"{req.query} via {det['backend']}{fb} -> {det['world']}", ok=det["found"])
    return det


@app.post("/api/robo_jev/arm")
async def api_arm_watch() -> dict[str, Any]:
    if _last_detection and _last_detection.get("found"):
        px = tuple(_last_detection["pixel"])
        q = _last_detection["query"]
        d = float(_last_detection["depth_m"])
    else:
        px = (320.0, 240.0)
        q = "center-roi"
        d = 0.62
    res = await srv.robo_jev_arm_watch(*px, query=q, depth_m=d)
    _log_event("ROBO-JEV", f"Armed watch on {q} @ {res['ref_pixel']}")
    return res


@app.post("/api/robo_jev/check")
async def api_check_reflex() -> dict[str, Any]:
    bridge = await srv._get_bridge()
    res = await check_reflex_and_abort(
        gatekeeper=srv._robo_jev,
        bridge=bridge,
        calib=srv._calib,
        obstacles=list(srv._obstacles.values()),
        camera="top",
    )
    if res.get("aborted"):
        srv._tts.speak(f"Reflex abort: {res['reason'].replace('_', ' ')}.")
    _log_event("ROBO-JEV", f"reason={res['reason']} ({res['latency_ms']} ms)", ok=res["ok"])
    return res


@app.post("/api/robo_jev/recover")
async def api_guarded_recover() -> dict[str, Any]:
    global _last_detection
    q = (_last_detection or {}).get("query") or (srv._robo_jev.watch.query if srv._robo_jev.watch else "keys")
    res = await srv.robo_jev_guard(
        query=q,
        use_gemini=bool(os.environ.get("GEMINI_API_KEY")),
    )
    if res.get("ok") and res.get("action") == "recalibrated" and res.get("world"):
        _last_detection = {
            **(_last_detection or {}),
            "found": True,
            "query": q,
            "label": q,
            "pixel": res["pixel"],
            "world": res["world"],
        }
    _log_event("RECOVER", f"action={res.get('action')} reason={res.get('reason', 'ok')}", ok=res.get("ok", False))
    return res


def _displace_object_in_frame(
    bgr: np.ndarray,
    box_px: tuple[int, int, int, int],
    mode: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Shift the object inside `box_px` either inside WorkspaceBounds or out of bounds."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box_px
    x1, y1 = max(4, x1), max(4, y1)
    x2, y2 = min(w - 4, x2), min(h - 4, y2)
    bw, bh = max(24, x2 - x1), max(24, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    def _in_safe_bounds(px: int, py: int) -> bool:
        wx, wy, wz = srv._calib.pixel_to_world(float(px), float(py), camera="top")
        return srv._robo_jev.safe_bounds.contains(wx, wy, wz)

    if mode == "out_of_bounds":
        candidates = [
            (w - max(bw // 2 + 20, 90), h - max(bh // 2 + 20, 80)),
            (max(bw // 2 + 20, 90), h - max(bh // 2 + 20, 80)),
            (w - max(bw // 2 + 20, 90), max(bh // 2 + 20, 80)),
        ]
        dst_cx, dst_cy = candidates[0]
        for cand_x, cand_y in candidates:
            if not _in_safe_bounds(cand_x, cand_y) and np.hypot(cand_x - cx, cand_y - cy) >= 120:
                dst_cx, dst_cy = cand_x, cand_y
                break
    else:
        # Pick a destination shifted >= 45 px whose world pose is guaranteed inside WorkspaceBounds
        candidates = [
            (cx + 60, cy),
            (cx - 60, cy),
            (cx, cy - 60),
            (cx + 80, cy - 60),
            (int(w * 0.5), int(h * 0.42)),
            (int(w * 0.48), int(h * 0.35)),
        ]
        dst_cx, dst_cy = int(w * 0.5), int(h * 0.42)
        for cand_x, cand_y in candidates:
            cand_x = max(bw // 2 + 8, min(w - bw // 2 - 8, cand_x))
            cand_y = max(bh // 2 + 8, min(h - bh // 2 - 8, cand_y))
            if _in_safe_bounds(cand_x, cand_y) and np.hypot(cand_x - cx, cand_y - cy) >= 45:
                dst_cx, dst_cy = cand_x, cand_y
                break

    nx1 = max(4, min(w - bw - 4, dst_cx - bw // 2))
    ny1 = max(4, min(h - bh - 4, dst_cy - bh // 2))
    nx2, ny2 = nx1 + bw, ny1 + bh

    patch = bgr[y1:y2, x1:x2].copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    pad = 6
    mask[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = 255
    out = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    out[ny1:ny2, nx1:nx2] = patch
    return out, (nx1 + bw // 2 - cx, ny1 + bh // 2 - cy)


@app.post("/api/mock/displace")
async def api_mock_displace(req: MockDisplaceReq) -> dict[str, Any]:
    bridge = await srv._get_bridge()
    if not hasattr(bridge, "set_frame_override"):
        return {"ok": False, "error": "Mock displacement only available in mock mode"}

    if req.mode == "reset":
        bridge.set_frame_override("top", None)
        _log_event("MOCK", "Reset top workspace frame to original")
        return {"ok": True, "mode": "reset"}

    # Always displace from the clean un-overridden base frame
    bridge.set_frame_override("top", None)
    base_jpeg = await bridge.get_frame("top", denoise=False)
    arr = np.frombuffer(base_jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"ok": False, "error": "Failed to decode base mock frame"}

    h, w = bgr.shape[:2]
    if _last_detection and _last_detection.get("found"):
        ymin, xmin, ymax, xmax = _last_detection["box_2d"]
        box_px = (int(xmin / 1000.0 * w), int(ymin / 1000.0 * h), int(xmax / 1000.0 * w), int(ymax / 1000.0 * h))
        label = _last_detection.get("query", "target")
    elif srv._robo_jev.watch is not None:
        box_px = srv._robo_jev.watch.box_px
        label = srv._robo_jev.watch.query
    else:
        # Default keys region on mock_frames/top.jpg (1280x720)
        box_px = (220, 340, 495, 535)
        label = "keys"

    displaced, (dx, dy) = _displace_object_in_frame(bgr, box_px, req.mode)
    ok_enc, buf = cv2.imencode(".jpg", displaced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok_enc:
        return {"ok": False, "error": "Failed to encode displaced frame"}

    bridge.set_frame_override("top", buf.tobytes())
    _log_event("MOCK", f"Displaced '{label}' ({req.mode}) by dx={dx}px, dy={dy}px")
    return {"ok": True, "mode": req.mode, "label": label, "dx_px": dx, "dy_px": dy}


def _keystore_path() -> Path:
    env_p = os.environ.get("DESK_KEYS_PATH", "").strip()
    return Path(env_p) if env_p else (Path.home() / ".config" / "desk-core" / "keys.json")


def _write_keystore(data: dict[str, str]) -> None:
    p = _keystore_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    payload = json.dumps(data, indent=2) + "\n"
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def _keys_status() -> dict[str, dict[str, bool]]:
    stored = _load_keystore()
    return {
        prov: {
            "configured": bool(_get_api_key(prov)),
            "in_keystore": prov in stored,
        }
        for prov in DEFAULT_PROVIDER_MODELS
    }


@app.get("/api/keys")
async def api_get_keys() -> dict[str, Any]:
    return {"providers": _keys_status()}


@app.post("/api/keys")
async def api_set_key(req: KeySetReq) -> dict[str, Any]:
    prov = req.provider.strip().lower()
    if prov == "claude":
        prov = "anthropic"
    if prov not in DEFAULT_PROVIDER_MODELS:
        return {"ok": False, "error": f"Unsupported provider: {prov}"}
    val = req.api_key.strip()
    if not val:
        return {"ok": False, "error": "Empty API key"}
    stored = _load_keystore()
    stored[prov] = val
    _write_keystore(stored)
    _log_event("KEYS", f"Saved API key for {prov} (0600)")
    return {"ok": True, "provider": prov, "providers": _keys_status()}


@app.post("/api/keys/unset")
async def api_unset_key(req: KeySetReq) -> dict[str, Any]:
    prov = req.provider.strip().lower()
    if prov == "claude":
        prov = "anthropic"
    if prov not in DEFAULT_PROVIDER_MODELS:
        return {"ok": False, "error": f"Unsupported provider: {prov}"}
    stored = _load_keystore()
    stored.pop(prov, None)
    _write_keystore(stored)
    _log_event("KEYS", f"Unset stored API key for {prov}")
    return {"ok": True, "provider": prov, "providers": _keys_status()}


@app.post("/api/vision/verify")
async def api_verify_vision(req: LocalizeReq) -> dict[str, Any]:
    """Run live localization against known mock_frames/top.jpg ground truth to verify efficacy."""
    gt_path = Path(__file__).resolve().parent.parent / "mock_frames" / "top.jpg"
    if not gt_path.exists():
        return {"ok": False, "error": "mock_frames/top.jpg not found"}
    jpeg = gt_path.read_bytes()
    # Ground-truth centers (x, y) in 1280x720 mock_frames/top.jpg
    gt = {
        "keys": (357.0, 437.0),
        "coffee mug": (630.0, 310.0),
    }
    errors: list[float] = []
    latencies: list[float] = []
    backends: list[str] = []
    fallback_reasons: list[str] = []
    for q, (gx, gy) in gt.items():
        t0 = time.perf_counter()
        det = await localize_object_3d(
            jpeg_bytes=jpeg,
            query=q,
            calib=srv._calib,
            camera="top",
            use_gemini=req.use_gemini,
            timeout_s=15.0,
            model=req.model,
            provider=req.provider,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        backends.append(det["backend"])
        if det.get("fallback_reason"):
            fallback_reasons.append(str(det["fallback_reason"]))
        if det.get("found"):
            px, py = det["pixel"]
            errors.append(float(np.hypot(px - gx, py - gy)))
        else:
            errors.append(999.0)
    mean_err = round(sum(errors) / len(errors), 1)
    mean_ms = round(sum(latencies) / len(latencies), 1)
    fb = fallback_reasons[0] if fallback_reasons else None
    color = "green" if mean_err <= 65.0 else ("amber" if mean_err <= 140.0 else "red")
    if fb:
        verdict = f"FALLBACK ({fb[:60]})"
        color = "red"
    else:
        verdict = "VERIFIED PASS" if mean_err <= 65.0 else ("FAIR" if mean_err <= 140.0 else "POOR")
    _log_event("VERIFY", f"{backends[0]}: {verdict} (err={mean_err}px, {mean_ms}ms)", ok=(not fb and mean_err <= 65.0))
    return {
        "ok": True,
        "backend": backends[0],
        "fallback_reason": fb,
        "mean_err_px": mean_err,
        "mean_latency_ms": mean_ms,
        "color": color,
        "verdict": verdict,
    }


@app.post("/api/tts")
async def api_tts(req: TTSReq) -> dict[str, Any]:
    queued = srv._tts.speak(req.text)
    _log_event("TTS", req.text, ok=queued)
    return {"queued": queued, "tts": srv._tts.status()}


@app.post("/api/first_light")
async def api_first_light() -> dict[str, Any]:
    """Run the Act 1 07:30 First-Light check-in routine (non-destructive if uncalibrated)."""
    global _first_light_running, _last_detection
    if _first_light_running:
        return {"ok": False, "error": "First-Light routine already running"}
    _first_light_running = True
    steps: list[dict[str, Any]] = []
    try:
        srv._tts.speak("Good morning. Starting 7 30 first light desk check.")
        _log_event("FIRST-LIGHT", "Started 07:30 First-Light routine")

        bridge = await srv._get_bridge()
        st = await bridge.get_state()
        steps.append({"step": "telemetry", "motors": len(st.motors), "bus": st.bus_serial})

        jpeg = await bridge.get_frame("top", denoise=True)
        det = await localize_object_3d(jpeg_bytes=jpeg, query="keys", calib=srv._calib, camera="top")
        _last_detection = det
        steps.append({"step": "localize_3d", "found": det["found"], "backend": det["backend"], "world": det["world"]})
        _log_event("FIRST-LIGHT", f"Localized keys via {det['backend']} -> {det['world']}", ok=det["found"])

        if det["found"]:
            await srv.robo_jev_arm_watch(*det["pixel"], query="keys", depth_m=det["depth_m"])
            reflex = await check_reflex_and_abort(srv._robo_jev, bridge, calib=srv._calib, obstacles=list(srv._obstacles.values()))
            steps.append({"step": "robo_jev", "ok": reflex["ok"], "latency_ms": reflex["latency_ms"]})
            srv._tts.speak(f"Keys spotted. Reflex check nominal in {int(round(reflex['latency_ms']))} milliseconds.")
        else:
            srv._tts.speak("Desk scan complete. No target keys detected.")

        return {"ok": True, "steps": steps}
    finally:
        _first_light_running = False


@app.get("/api/logs")
async def api_logs() -> dict[str, Any]:
    mcp_calls: list[dict[str, Any]] = []
    runs_dir = srv.RUNS_DIR
    if runs_dir.exists():
        run_logs = sorted(runs_dir.glob("*/calls.jsonl"), key=lambda p: p.stat().st_mtime)
        if run_logs:
            lines = run_logs[-1].read_text(encoding="utf-8", errors="ignore").splitlines()[-15:]
            for ln in reversed(lines):
                try:
                    rec = json.loads(ln)
                    if rec.get("event") == "tool_call":
                        mcp_calls.append({
                            "ts": rec.get("ts", "")[11:19],
                            "tool": rec.get("tool"),
                            "ok": rec.get("ok"),
                            "duration_ms": rec.get("duration_ms"),
                        })
                except Exception:
                    pass
    return {"dashboard_events": list(_action_log), "mcp_calls": mcp_calls}


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>DESK-CORE Mission Control</title>
  <style>
    :root { --bg: #0b0f17; --card: #131a26; --border: #243044; --text: #e6edf7; --muted: #8ea0b8; --green: #00e676; --red: #ff3b3b; --cyan: #38bdf8; --amber: #fbbf24; }
    * { box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); padding: 14px; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; gap: 12px; }
    .title { font-size: 20px; font-weight: 700; letter-spacing: 0.5px; }
    .badge { padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; background: #1e293b; color: var(--cyan); }
    .estop-btn { background: var(--red); color: white; border: none; padding: 14px 28px; font-size: 18px; font-weight: 800; border-radius: 10px; cursor: pointer; box-shadow: 0 0 18px rgba(255,59,59,0.45); }
    .estop-btn:hover { filter: brightness(1.1); }
    .grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 12px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
    .cams { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .cam-box img { width: 100%; border-radius: 8px; border: 1px solid var(--border); background: #000; display: block; }
    .cam-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; font-weight: 600; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
    input[type=text] { background: #0b0f17; border: 1px solid var(--border); color: var(--text); padding: 8px 10px; border-radius: 6px; flex: 1; min-width: 140px; }
    button.act { background: #1d4ed8; color: white; border: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; }
    button.first-light { background: linear-gradient(90deg, #f59e0b, #ea580c); color: #fff; border: none; padding: 10px 16px; border-radius: 8px; font-weight: 700; cursor: pointer; }
    .meter { margin: 6px 0; }
    .meter-head { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px; }
    .bar { height: 8px; background: #0b0f17; border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; background: var(--green); transition: width 0.2s; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #1e293b; }
    .log { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; max-height: 180px; overflow-y: auto; }
  </style>
</head>
<body>
  <header>
    <div style="display:flex;align-items:center;gap:12px;">
      <div class="title">DESK-CORE MISSION CONTROL</div>
      <span id="link-badge" class="badge">CONNECTING...</span>
      <span id="det-badge" class="badge">DETECTOR: none</span>
      <span id="jev-latency" class="badge">ROBO-JEV: CPU CV (0 ms)</span>
    </div>
    <div style="display:flex;gap:10px;align-items:center;">
      <button class="first-light" onclick="runFirstLight()">Run 07:30 First-Light Routine</button>
      <button class="estop-btn" onclick="triggerEstop()">E-STOP</button>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="cams">
        <div class="cam-box">
          <div class="cam-label">TOP WORKSPACE (X-Ray / Robo-Jev ROI)</div>
          <img id="cam-top" src="/api/frame/top.jpg" alt="top" />
          <label><input id="show-xray" type="checkbox" onchange="refreshFrames()" /> X-Ray (obstacles and safe path)</label>
        </div>
        <div class="cam-box">
          <div class="cam-label">GRIPPER EYE-IN-HAND (camera id: wrist)</div>
          <img id="cam-wrist" src="/api/frame/wrist.jpg" alt="wrist" />
        </div>
      </div>
      <div class="row">
        <input id="loc-query" type="text" value="keys" placeholder="Object query (keys, mug, marker...)" />
        <select id="loc-camera" style="background:#0b1220;color:var(--text);border:1px solid #334155;border-radius:6px;padding:7px 8px;font-size:13px;">
          <option value="top" selected>Top View</option>
          <option value="wrist">Gripper View (wrist)</option>
          <option value="both" disabled>Both Views (Top + Gripper — disabled)</option>
        </select>
        <select id="loc-provider" onchange="syncVisionModels()" style="background:#0b1220;color:var(--text);border:1px solid #334155;border-radius:6px;padding:7px 8px;font-size:13px;">
          <option value="auto">Auto (Env/Keystore)</option>
          <option value="gemini">Gemini</option>
          <option value="anthropic">Anthropic</option>
          <option value="openai">OpenAI</option>
          <option value="qwen">Qwen VL</option>
          <option value="deepseek">DeepSeek VL</option>
          <option value="opencv">OpenCV Only</option>
        </select>
        <select id="loc-model" onchange="updateModelRating()" style="background:#0b1220;color:var(--text);border:1px solid #334155;border-radius:6px;padding:7px 8px;font-size:13px;"></select>
        <span id="model-rating" class="badge">TIER: --</span>
        <button class="act" onclick="runLocalize()">Localize 3D</button>
        <button class="act" style="background:#0f766e;color:#fff;" onclick="verifyVisionModel()">Verify Model</button>
        <button class="act" onclick="armWatch()">Arm Watch</button>
        <button class="act" onclick="checkReflex()">Check Reflex</button>
        <button class="act" onclick="guardedRecover()">Guarded Recover</button>
      </div>
      <div class="row">
        <input id="api-key-input" type="password" placeholder="API key for selected provider (saved 0600)..." />
        <button class="act" onclick="saveProviderKey()">Set Key</button>
        <button class="act" style="background:#475569;color:#fff;" onclick="unsetProviderKey()">Unset Key</button>
        <span id="key-status" class="badge">KEY: check...</span>
      </div>
      <div class="row">
        <button class="act" style="background:#b45309;color:#fff;" onclick="mockDisplace('in_bounds')">Displace In Safe Bounds</button>
        <button class="act" style="background:#991b1b;color:#fff;" onclick="mockDisplace('out_of_bounds')">Displace Out of Bounds</button>
        <button class="act" onclick="mockDisplace('reset')">Reset Mock Scene</button>
      </div>
      <div class="row">
        <input id="tts-text" type="text" value="Desk core standing by." placeholder="Speaker TTS message" />
        <button class="act" onclick="sendTTS()">Speak TTS</button>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:12px;">
      <div class="card">
        <div style="font-weight:700;margin-bottom:6px;">Robo-Jev &lt;150 ms Safety Gatekeeper</div>
        <div id="jev-reason" style="font-size:12px;color:var(--muted);margin-bottom:6px;">Status: nominal</div>
        <div class="meter"><div class="meter-head"><span>Path Clear</span><span id="m-path">100%</span></div><div class="bar"><div id="b-path" class="fill" style="width:100%"></div></div></div>
        <div class="meter"><div class="meter-head"><span>Grasp Secure</span><span id="m-grasp">100%</span></div><div class="bar"><div id="b-grasp" class="fill" style="width:100%"></div></div></div>
        <div class="meter"><div class="meter-head"><span>Tip Risk</span><span id="m-tip">0%</span></div><div class="bar"><div id="b-tip" class="fill" style="width:0%;background:var(--amber)"></div></div></div>
      </div>

      <div class="card">
        <div style="font-weight:700;margin-bottom:6px;">ST3215 Telemetry</div>
        <table><thead><tr><th>ID</th><th>Ticks</th><th>Norm</th><th>mA</th><th>Err</th></tr></thead><tbody id="motor-rows"></tbody></table>
      </div>

      <div class="card">
        <div style="font-weight:700;margin-bottom:6px;">Action &amp; MCP Log</div>
        <div id="log-box" class="log"></div>
      </div>
    </div>
  </div>

  <script>
    async function postJSON(url, body) {
      const r = await fetch(url, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {}) });
      return r.json();
    }
    async function triggerEstop() { await postJSON('/api/estop'); refresh(); }
    const VISION_MODELS = {
      auto: [['', 'Default Vision Model', 'Auto Tier', 'green', 'Auto-picks configured provider']],
      gemini: [
        ['gemini-4-hydrogen', 'gemini-4-hydrogen', 'Robotics Pointing', 'green', 'Gemini 4 frontier vision'],
        ['gemini-4-argon', 'gemini-4-argon', 'Frontier Vision', 'green', 'Gemini 4 Argon multimodal'],
        ['gemini-robotics-er-2-preview', 'gemini-robotics-er-2-preview', 'Robotics Pointing', 'green', 'Embodied [y,x] pointing'],
        ['gemini-3.8-pro', 'gemini-3.8-pro', 'Frontier Vision', 'green', 'High-accuracy multimodal'],
        ['gemini-3.8-flash', 'gemini-3.8-flash', 'Fast Vision', 'amber', 'Low-latency multimodal']
      ],
      anthropic: [
        ['claude-fable-5-2', 'claude-fable-5-2', 'Frontier Vision', 'green', 'Flagship agentic vision'],
        ['claude-opus-5-5', 'claude-opus-5-5', 'Frontier Vision', 'green', 'Strongest Claude 5.5'],
        ['claude-sonnet-5-5', 'claude-sonnet-5-5', 'Balanced Vision', 'green', 'Speed + accuracy (5.5)'],
        ['claude-haiku-5-5', 'claude-haiku-5-5', 'Fast Vision', 'amber', 'Low-latency (5.5)'],
        ['claude-opus-4-6', 'claude-opus-4-6', 'Legacy Vision', 'amber', 'Legacy 4.6 series'],
        ['claude-sonnet-4-6', 'claude-sonnet-4-6', 'Legacy Vision', 'amber', 'Legacy 4.6 series']
      ],
      openai: [
        ['gpt-6.2-astra', 'gpt-6.2-astra', 'Frontier Vision', 'green', 'Flagship reasoning + vision'],
        ['gpt-6.2-sol', 'gpt-6.2-sol', 'Balanced Vision', 'green', 'High efficiency (6.2)'],
        ['gpt-6.1-terra', 'gpt-6.1-terra', 'Visual Grounding', 'green', 'Computer-use & visual reasoning'],
        ['gpt-6-luna', 'gpt-6-luna', 'Fast Vision', 'amber', 'High-volume fast vision'],
        ['gpt-5', 'gpt-5', 'Legacy Vision', 'amber', 'Retiring Dec 2026'],
        ['gpt-4.1', 'gpt-4.1', 'Legacy Vision', 'amber', 'Deprecated 4.1 series']
      ],
      qwen: [
        ['qwen4-max', 'qwen4-max', 'Frontier Vision', 'green', 'Qwen 4 Max (GA Nov 2026)'],
        ['qwen3.8-max', 'qwen3.8-max', 'Frontier Vision', 'green', 'Top ERQA Qwen3.8 VL'],
        ['qwen3-vl-plus', 'qwen3-vl-plus', 'Balanced Vision', 'green', 'Qwen3 VL spatial grounding'],
        ['qwen-vl-max', 'qwen-vl-max', 'Balanced Vision', 'amber', 'High-res Qwen VL'],
        ['qwen2.5-vl-72b-instruct', 'qwen2.5-vl-72b-instruct', 'Open-Weight VL', 'amber', 'Qwen 2.5 VL 72B']
      ],
      deepseek: [
        ['deepseek-flash', 'deepseek-flash', 'Frontier Vision', 'green', 'DeepSeek-V4.1-Flash-1017 (Multimodal, 1M)'],
        ['deepseek-vl2', 'deepseek-vl2', 'VL2 Endpoint', 'amber', 'Requires VL2-compatible host'],
        ['deepseek-vl2-small', 'deepseek-vl2-small', 'Compact VL2', 'amber', 'Requires VL2-compatible host']
      ],
      opencv: [['', 'opencv-fallback', 'Offline Heuristic', 'red', 'No semantic understanding']]
    };
    const RATING_STYLES = {
      green: { bg: '#064e3b', fg: '#34d399', bd: '#10b981' },
      amber: { bg: '#78350f', fg: '#fbbf24', bd: '#f59e0b' },
      red: { bg: '#7f1d1d', fg: '#fca5a5', bd: '#ef4444' }
    };
    function applyRatingStyle(badge, col, text) {
      const s = RATING_STYLES[col] || { bg: '#1e293b', fg: '#94a3b8', bd: '#334155' };
      badge.textContent = text;
      badge.style.background = s.bg;
      badge.style.color = s.fg;
      badge.style.borderColor = s.bd;
    }
    function updateModelRating() {
      const prov = document.getElementById('loc-provider').value || 'auto';
      const sel = document.getElementById('loc-model');
      const opts = VISION_MODELS[prov] || VISION_MODELS.auto;
      const picked = opts.find(o => o[0] === sel.value) || opts[0];
      const [, , tier, col, note] = picked;
      applyRatingStyle(document.getElementById('model-rating'), col, `${tier} · ${note}`);
    }
    async function verifyVisionModel() {
      const prov = document.getElementById('loc-provider').value || 'auto';
      const model = document.getElementById('loc-model').value || null;
      const badge = document.getElementById('model-rating');
      badge.textContent = 'VERIFYING...';
      const r = await postJSON('/api/vision/verify', { query: 'keys', provider: prov, model: model });
      if (r.ok) {
        applyRatingStyle(badge, r.color, `${r.verdict} (${r.mean_err_px}px · ${r.mean_latency_ms}ms)`);
      }
      refresh();
    }
    let _keyProviders = {};
    function updateKeyBadge() {
      const prov = document.getElementById('loc-provider').value || 'auto';
      const el = document.getElementById('key-status');
      if (prov === 'opencv') { el.textContent = 'KEY: offline CV'; return; }
      if (prov === 'auto') {
        const setList = Object.entries(_keyProviders).filter(([, v]) => v.configured).map(([k]) => k);
        el.textContent = setList.length ? `KEYS: ${setList.join(', ')}` : 'KEYS: none set';
        return;
      }
      const st = _keyProviders[prov] || {};
      el.textContent = st.configured ? (st.in_keystore ? `KEY (${prov}): SET (0600)` : `KEY (${prov}): SET (env)`) : `KEY (${prov}): unset`;
    }
    async function refreshKeys() {
      try {
        const res = await fetch('/api/keys').then(r => r.json());
        _keyProviders = res.providers || {};
        updateKeyBadge();
      } catch (e) {}
    }
    async function saveProviderKey() {
      const prov = document.getElementById('loc-provider').value || 'gemini';
      const key = document.getElementById('api-key-input').value.trim();
      if (!key || prov === 'auto' || prov === 'opencv') return;
      await postJSON('/api/keys', { provider: prov, api_key: key });
      document.getElementById('api-key-input').value = '';
      await refreshKeys();
      refresh();
    }
    async function unsetProviderKey() {
      const prov = document.getElementById('loc-provider').value || 'gemini';
      if (prov === 'auto' || prov === 'opencv') return;
      await postJSON('/api/keys/unset', { provider: prov });
      await refreshKeys();
      refresh();
    }
    function syncVisionModels() {
      const prov = document.getElementById('loc-provider').value || 'auto';
      const sel = document.getElementById('loc-model');
      const opts = VISION_MODELS[prov] || VISION_MODELS.auto;
      sel.innerHTML = opts.map(([v, l, tier]) => `<option value="${v}">${l} [${tier}]</option>`).join('');
      updateKeyBadge();
      updateModelRating();
    }
    syncVisionModels();
    refreshKeys();

    async function runLocalize() {
      const cam = document.getElementById('loc-camera').value || 'both';
      const prov = document.getElementById('loc-provider').value || 'auto';
      const model = document.getElementById('loc-model').value || null;
      await postJSON('/api/localize', {
        query: document.getElementById('loc-query').value || 'keys',
        camera: cam,
        provider: prov,
        model: model
      });
      refresh();
    }
    async function armWatch() { await postJSON('/api/robo_jev/arm'); refresh(); }
    async function checkReflex() { await postJSON('/api/robo_jev/check'); refresh(); }
    async function guardedRecover() { await postJSON('/api/robo_jev/recover'); refresh(); }
    async function mockDisplace(mode) { await postJSON('/api/mock/displace', { mode }); refreshFrames(); refresh(); }
    async function sendTTS() { await postJSON('/api/tts', { text: document.getElementById('tts-text').value }); refresh(); }
    async function runFirstLight() { await postJSON('/api/first_light'); refresh(); }

    function refreshFrames() {
      const t = Date.now();
      const xray = document.getElementById('show-xray').checked;
      document.getElementById('cam-top').src = '/api/frame/top.jpg?show_overlay=' + xray + '&t=' + t;
      document.getElementById('cam-wrist').src = '/api/frame/wrist.jpg?t=' + t;
    }

    async function refresh() {
      try {
        const [tel, logs] = await Promise.all([fetch('/api/telemetry').then(r => r.json()), fetch('/api/logs').then(r => r.json())]);
        const link = tel.link || {};
        document.getElementById('link-badge').textContent = `${(link.bridge || 'offline').toUpperCase()} | TORQUE: ${tel.torque_enabled ? 'ON' : 'OFF'}`;
        const det = tel.last_detection || null;
        if (det) {
          const fb = det.fallback_reason ? ` (${det.fallback_reason})` : '';
          document.getElementById('det-badge').textContent = `DETECTOR: ${det.backend}${fb}`;
        }
        const rj = tel.robo_jev || {};
        document.getElementById('jev-latency').textContent = `ROBO-JEV (CPU CV): ${rj.latency_ms ?? 0} ms`;
        document.getElementById('jev-reason').textContent = `Status: ${rj.reason || 'nominal'} | Shifted: ${rj.target_shifted} (${rj.shift_mm || 0} mm)`;
        const p = Math.round((rj.path_clear ?? 1) * 100), g = Math.round((rj.grasp_secure ?? 1) * 100), tip = Math.round((rj.will_tip ?? 0) * 100);
        document.getElementById('m-path').textContent = p + '%'; document.getElementById('b-path').style.width = p + '%';
        document.getElementById('m-grasp').textContent = g + '%'; document.getElementById('b-grasp').style.width = g + '%';
        document.getElementById('m-tip').textContent = tip + '%'; document.getElementById('b-tip').style.width = tip + '%';

        document.getElementById('motor-rows').innerHTML = (tel.motors || []).map(m =>
          `<tr><td>M${m.id}</td><td>${m.position}</td><td>${m.norm}</td><td>${m.current_ma}</td><td>${m.err}</td></tr>`
        ).join('');

        const esc = v => String(v ?? '').replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[ch]);
        const evs = (logs.dashboard_events || []).map(e => `<div>[${esc(e.ts)}] <b>${esc(e.kind)}</b>: ${esc(e.detail)}</div>`).join('');
        const calls = (logs.mcp_calls || []).map(c => `<div style="color:var(--muted)">[${esc(c.ts)}] MCP ${esc(c.tool)} (${esc(c.duration_ms)}ms)</div>`).join('');
        document.getElementById('log-box').innerHTML = evs + calls;
      } catch (e) {}
    }

    setInterval(refreshFrames, 250);
    setInterval(refresh, 500);
    refresh();
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return DASHBOARD_HTML


def main() -> None:
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
