"""Secure vision-only LLM plugin registry, 0600 keystore, and multimodal agent loop.

Only vision-capable providers (Gemini, Anthropic Claude, OpenAI) and their
image-capable models are permitted. API keys are stored locally in
`~/.config/desk-core/keys.json` with `0o600` permissions (never logged or committed).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class VisionModelSpec:
    provider: str
    model_id: str
    label: str
    supports_images: bool = True


# Strict allow-list: ONLY providers and models that support image inputs
VISION_PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {"name": "Google Gemini", "env_var": "GEMINI_API_KEY", "alt_env_var": "GOOGLE_API_KEY"},
    "claude": {"name": "Anthropic Claude", "env_var": "ANTHROPIC_API_KEY"},
    "openai": {"name": "OpenAI", "env_var": "OPENAI_API_KEY"},
    "qwen": {"name": "Qwen VL", "env_var": "QWEN_API_KEY", "alt_env_var": "DASHSCOPE_API_KEY"},
    "deepseek": {"name": "DeepSeek VL", "env_var": "DEEPSEEK_API_KEY"},
}

VISION_MODELS: list[VisionModelSpec] = [
    # Google Gemini
    VisionModelSpec("gemini", "gemini-4-hydrogen", "Gemini 4 Hydrogen (Robotics Pointing)"),
    VisionModelSpec("gemini", "gemini-4-argon", "Gemini 4 Argon (Frontier Vision)"),
    VisionModelSpec("gemini", "gemini-robotics-er-2-preview", "Gemini Robotics ER 2 Preview (Embodied Pointing)"),
    VisionModelSpec("gemini", "gemini-3.8-pro", "Gemini 3.8 Pro (Frontier Vision)"),
    VisionModelSpec("gemini", "gemini-3.8-flash", "Gemini 3.8 Flash (Fast Vision)"),
    # Anthropic Claude
    VisionModelSpec("claude", "claude-fable-5-2", "Claude Fable 5.2 (Frontier Vision)"),
    VisionModelSpec("claude", "claude-opus-5-5", "Claude Opus 5.5 (Frontier Vision)"),
    VisionModelSpec("claude", "claude-sonnet-5-5", "Claude Sonnet 5.5 (Balanced Vision)"),
    VisionModelSpec("claude", "claude-haiku-5-5", "Claude Haiku 5.5 (Fast Vision)"),
    VisionModelSpec("claude", "claude-opus-4-6", "Claude Opus 4.6 (Legacy Vision)"),
    VisionModelSpec("claude", "claude-sonnet-4-6", "Claude Sonnet 4.6 (Legacy Vision)"),
    # OpenAI
    VisionModelSpec("openai", "gpt-6.2-astra", "GPT-6.2 Astra (Frontier Vision)"),
    VisionModelSpec("openai", "gpt-6.2-sol", "GPT-6.2 Sol (Balanced Vision)"),
    VisionModelSpec("openai", "gpt-6.1-terra", "GPT-6.1 Terra (Visual Grounding)"),
    VisionModelSpec("openai", "gpt-6-luna", "GPT-6 Luna (Fast Vision)"),
    VisionModelSpec("openai", "gpt-5", "GPT-5 (Legacy Vision)"),
    VisionModelSpec("openai", "gpt-4.1", "GPT-4.1 (Legacy Vision)"),
    # Qwen VL
    VisionModelSpec("qwen", "qwen4-max", "Qwen 4 Max (Frontier Vision)"),
    VisionModelSpec("qwen", "qwen3.8-max", "Qwen 3.8 Max (Frontier Vision)"),
    VisionModelSpec("qwen", "qwen3-vl-plus", "Qwen3 VL Plus (Balanced Vision)"),
    VisionModelSpec("qwen", "qwen-vl-max", "Qwen VL Max (Balanced Vision)"),
    VisionModelSpec("qwen", "qwen2.5-vl-72b-instruct", "Qwen 2.5 VL 72B (Open-Weight VL)"),
    # DeepSeek VL
    VisionModelSpec("deepseek", "deepseek-flash", "DeepSeek-V4.1-Flash-1017 (Frontier Vision)"),
    VisionModelSpec("deepseek", "deepseek-vl2", "DeepSeek-VL2 (VL2 Endpoint)"),
    VisionModelSpec("deepseek", "deepseek-vl2-small", "DeepSeek-VL2 Small (Compact VL2)"),
]

VISION_MODEL_MAP: dict[str, VisionModelSpec] = {
    f"{m.provider}:{m.model_id}": m for m in VISION_MODELS
}


def _normalize_provider(provider: str) -> str:
    p = provider.strip().lower()
    return "claude" if p == "anthropic" else p


def validate_vision_plugin(provider: str, model_id: str | None = None) -> VisionModelSpec | None:
    """Ensure only image-capable providers & models can be registered or selected."""
    p = _normalize_provider(provider)
    if p not in VISION_PROVIDERS:
        raise ValueError(
            f"Provider {provider!r} is not supported or does not support vision. "
            f"Allowed vision providers: {sorted(VISION_PROVIDERS.keys())}"
        )
    if model_id is None:
        return None
    key = f"{p}:{model_id.strip()}"
    if key not in VISION_MODEL_MAP:
        allowed = [m.model_id for m in VISION_MODELS if m.provider == p]
        raise ValueError(
            f"Model {model_id!r} is not in the vision-capable allow-list for {p}. "
            f"Allowed: {allowed}"
        )
    return VISION_MODEL_MAP[key]


class SecureKeyStore:
    """Local 0600 JSON keystore for vision LLM API keys."""

    def __init__(self, path: Path | None = None) -> None:
        env_path = os.environ.get("DESK_KEYS_PATH", "").strip()
        self.path = path or (Path(env_path) if env_path else (Path.home() / ".config" / "desk-core" / "keys.json"))
        self._keys: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._keys = {
                    k: str(v).strip()
                    for k, v in raw.items()
                    if k in VISION_PROVIDERS and str(v).strip()
                }
            except Exception:
                self._keys = {}
        else:
            self._keys = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        payload = json.dumps(self._keys, indent=2) + "\n"
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def set_key(self, provider: str, api_key: str) -> None:
        validate_vision_plugin(provider)
        p = provider.strip().lower()
        clean = api_key.strip()
        if clean:
            self._keys[p] = clean
        else:
            self._keys.pop(p, None)
        self._save()

    def unset_key(self, provider: str, clear_env: bool = True) -> None:
        """Remove a provider's API key from the local keystore (and current process env)."""
        validate_vision_plugin(provider)
        p = provider.strip().lower()
        self._keys.pop(p, None)
        self._save()
        if clear_env:
            os.environ.pop(VISION_PROVIDERS[p]["env_var"], None)

    def get_key(self, provider: str) -> str:
        p = _normalize_provider(provider)
        if p not in VISION_PROVIDERS:
            return ""
        if self._keys.get(p):
            return self._keys[p]
        meta = VISION_PROVIDERS[p]
        val = os.environ.get(meta["env_var"], "").strip()
        if val:
            return val
        alt = meta.get("alt_env_var", "")
        return os.environ.get(alt, "").strip() if alt else ""

    def status(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for p, meta in VISION_PROVIDERS.items():
            val = self.get_key(p)
            masked = f"••••{val[-4:]}" if len(val) >= 4 else ("••••" if val else "")
            out[p] = {
                "name": meta["name"],
                "configured": bool(val),
                "masked": masked,
                "source": "keystore" if self._keys.get(p) else ("env" if val else "none"),
            }
        return out

    def available_dropdown_options(self) -> list[tuple[str, str]]:
        """Return `(display_label, 'provider:model_id')` for vision-capable models."""
        st = self.status()
        configured = [m for m in VISION_MODELS if st[m.provider]["configured"]]
        unconfigured = [m for m in VISION_MODELS if not st[m.provider]["configured"]]
        options: list[tuple[str, str]] = []
        for m in configured:
            options.append((f"[{m.provider.upper()}] {m.label}", f"{m.provider}:{m.model_id}"))
        for m in unconfigured:
            options.append((f"[{m.provider.upper()} - add key] {m.label}", f"{m.provider}:{m.model_id}"))
        return options


class AgentToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentStepResponse(BaseModel):
    thought: str = Field(default="", description="Brief reasoning about what you see and next step.")
    tool_calls: list[AgentToolCall] = Field(default_factory=list, description="MCP tools to invoke.")
    reply: str = Field(default="", description="Concise user-facing response.")
    done: bool = Field(default=True, description="True if finished, False if more steps needed after tool results.")


def _build_system_prompt(tools_summary: str) -> str:
    return (
        "You are DESK-CORE, a multimodal robot arm operator.\n"
        "You receive live camera image(s) from the robot workspace and user commands.\n"
        "Respond ONLY with valid JSON matching:\n"
        '{"tool_calls": [{"tool": "name", "args": {...}}], "reply": "...", "done": true/false}\n'
        "Rules:\n"
        "- Do NOT output or narrate your thoughts, reasoning, coordinates, or step-by-step plan.\n"
        "- While `done` is false, set `reply` to \"\".\n"
        "- When `done` is true, set `reply` to ONE short, direct confirmation or answer (under 15 words, e.g. 'Moved the pen onto the mug.').\n"
        "- To locate objects before picking/moving, call `localize_3d(query=..., camera='top')` to get the object's 2D box and marks the live overlays. Its 3D point is a placeholder on the table plane, so it is only approximate.\n"
        "- Call `pick_at(x, y, z)` and `place_at(x, y, z)` directly once coordinates are known (they handle reachable grasp orientation); do NOT loop on `solve_ik` guessing quaternions.\n"
        "- Never report success if a motion tool failed or was blocked; report the failure briefly.\n"
        "- Batch tool calls in one step when safe (e.g., `pick_at` + `place_at` when target coordinates are known) "
        "and avoid redundant `observe`/`get_state` calls since live camera images are already attached.\n"
        f"Available MCP tools:\n{tools_summary}"
    )


def _extract_json_dict(raw_text: str) -> dict[str, Any]:
    import re as _re

    txt = (raw_text or "").strip()
    if not txt:
        raise RuntimeError("Model returned an empty response.")
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    for m in _re.finditer(r"```(?:json)?\s*([\s\S]*?)```", txt, flags=_re.IGNORECASE):
        candidate = m.group(1).strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(txt):
        if ch == "{":
            try:
                obj, _end = decoder.raw_decode(txt, idx)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    return {"thought": "", "tool_calls": [], "reply": txt, "done": True}


def _call_gemini_step(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    images_jpeg: list[tuple[str, bytes]],
) -> AgentStepResponse:
    from google import genai

    client = genai.Client(api_key=api_key)
    content: list[dict[str, Any]] = []
    for _cam, jpg in images_jpeg:
        content.append({
            "type": "image",
            "data": base64.b64encode(jpg).decode("ascii"),
            "mime_type": "image/jpeg",
        })
    content.append({"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"})
    interaction = client.interactions.create(
        model=model_id,
        input=content,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": AgentStepResponse.model_json_schema(),
        },
        store=False,
    )
    raw = getattr(interaction, "output_text", None) or ""
    if not raw.strip() and getattr(interaction, "outputs", None):
        raw = "\n".join(str(getattr(o, "text", "") or "") for o in interaction.outputs)
    return AgentStepResponse.model_validate(_extract_json_dict(raw))


def _http_post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float = 25.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {err_body[:300]}") from exc


def _call_claude_step(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    images_jpeg: list[tuple[str, bytes]],
) -> AgentStepResponse:
    content: list[dict[str, Any]] = []
    for _cam, jpg in images_jpeg:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(jpg).decode("ascii"),
            },
        })
    content.append({"type": "text", "text": user_prompt})
    res = _http_post_json(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        body={
            "model": model_id,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        },
    )
    text_parts = [b.get("text", "") for b in res.get("content", []) if b.get("type") == "text"]
    return AgentStepResponse.model_validate(_extract_json_dict("\n".join(text_parts)))


def _call_openai_compat_step(
    model_id: str,
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    images_jpeg: list[tuple[str, bytes]],
) -> AgentStepResponse:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for _cam, jpg in images_jpeg:
        b64 = base64.b64encode(jpg).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    res = _http_post_json(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        body={
            "model": model_id,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        },
    )
    txt = res["choices"][0]["message"]["content"]
    return AgentStepResponse.model_validate(_extract_json_dict(txt))


def call_vision_model_step(
    provider: str,
    model_id: str,
    api_key: str,
    tools_summary: str,
    user_prompt: str,
    images_jpeg: list[tuple[str, bytes]],
) -> AgentStepResponse:
    validate_vision_plugin(provider, model_id)
    if not api_key.strip():
        raise RuntimeError(f"No API key configured for {provider}. Press Ctrl+K to add one.")
    sys_prompt = _build_system_prompt(tools_summary)
    p = _normalize_provider(provider)
    if p == "gemini":
        return _call_gemini_step(model_id, api_key, sys_prompt, user_prompt, images_jpeg)
    if p == "claude":
        return _call_claude_step(model_id, api_key, sys_prompt, user_prompt, images_jpeg)
    if p == "openai":
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return _call_openai_compat_step(model_id, api_key, url, sys_prompt, user_prompt, images_jpeg)
    if p == "qwen":
        default_qwen = (
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
            if api_key.strip().startswith("sk-sp-")
            else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        url = os.environ.get("QWEN_BASE_URL", default_qwen)
        return _call_openai_compat_step(model_id, api_key, url, sys_prompt, user_prompt, images_jpeg)
    if p == "deepseek":
        url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        return _call_openai_compat_step(model_id, api_key, url, sys_prompt, user_prompt, images_jpeg)
    raise ValueError(f"Unsupported provider: {provider}")


def _pick_macos_audio_index() -> str:
    override = os.environ.get("DESK_AUDIO_DEVICE", "").strip()
    if override:
        return override.lstrip(":")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        txt = proc.stderr.decode("utf-8", errors="ignore")
        in_audio = False
        candidates: list[tuple[str, str]] = []
        virtual_kw = ("teams", "blackhole", "soundflower", "zoom", "loopback", "virtual", "obs")
        for line in txt.splitlines():
            if "AVFoundation audio devices:" in line:
                in_audio = True
                continue
            if in_audio and "AVFoundation" in line:
                import re as _re
                m = _re.search(r"\[(\d+)\]\s+(.+)$", line)
                if m:
                    idx, name = m.group(1), m.group(2).strip()
                    if not any(k in name.lower() for k in virtual_kw):
                        candidates.append((idx, name))
        for idx, name in candidates:
            if "macbook" in name.lower() or "built-in" in name.lower():
                return idx
        if candidates:
            return candidates[0][0]
    except Exception:
        pass
    return "0"


def _wav_has_speech(wav_bytes: bytes, min_rms: float = 60.0) -> bool:
    import io as _io
    import struct as _struct
    import wave as _wave

    try:
        with _wave.open(_io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        if len(frames) < 320:
            return False
        samples = _struct.unpack(f"<{len(frames) // 2}h", frames[: (len(frames) // 2) * 2])
        mean = sum(samples) / len(samples)
        rms = (sum((s - mean) ** 2 for s in samples) / len(samples)) ** 0.5
        return rms >= min_rms
    except Exception:
        return True


def _normalize_pcm_wav(wav_bytes: bytes) -> bytes:
    """Fast in-memory DC-removal and gentle gain normalization for 16-bit mono WAV."""
    import io as _io
    import struct as _struct
    import wave as _wave

    try:
        with _wave.open(_io.BytesIO(wav_bytes), "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())
        if len(frames) < 320 or params.sampwidth != 2:
            return wav_bytes
        samples = _struct.unpack(f"<{len(frames) // 2}h", frames[: (len(frames) // 2) * 2])
        mean = sum(samples) / len(samples)
        centered = [s - mean for s in samples]
        peak = max(abs(x) for x in centered) if centered else 0.0
        if peak < 150.0:
            return wav_bytes
        gain = min(12.0, 20000.0 / peak) if peak < 20000.0 else 1.0
        normed = [max(-32767, min(32767, int(round(x * gain)))) for x in centered]
        out = _io.BytesIO()
        with _wave.open(out, "wb") as wf_out:
            wf_out.setparams(params)
            wf_out.writeframes(_struct.pack(f"<{len(normed)}h", *normed))
        return out.getvalue()
    except Exception:
        return wav_bytes


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    import io as _io
    import wave as _wave

    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _chunk_rms(chunk: bytes) -> float:
    import struct as _struct

    n = len(chunk) // 2
    if n <= 0:
        return 0.0
    samples = _struct.unpack(f"<{n}h", chunk[: n * 2])
    mean = sum(samples) / n
    return (sum((s - mean) ** 2 for s in samples) / n) ** 0.5


def record_wav_sync(seconds: float = 6.0, vad_auto_stop: bool = True) -> bytes:
    """Record mono 16 kHz WAV clip with VAD auto-stop so short commands finish immediately."""
    if shutil.which("ffmpeg"):
        if sys.platform == "darwin":
            dev_idx = _pick_macos_audio_index()
            in_fmt, in_dev = "avfoundation", f":{dev_idx}"
        else:
            in_fmt, in_dev = "alsa", "default"

        dur = str(max(1, int(round(seconds))))
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-f", in_fmt, "-i", in_dev,
            "-t", dur, "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        chunks: list[bytes] = []
        chunk_bytes = 3200  # 100ms at 16kHz * 2 bytes
        speech_started = False
        silent_chunks = 0
        noise_samples: list[float] = []
        min_chunks = 10  # at least 1.0s
        trailing_silence_chunks = 7  # stop 0.7s after user finishes speaking

        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(chunk_bytes)
                if not chunk:
                    break
                chunks.append(chunk)
                if vad_auto_stop:
                    rms = _chunk_rms(chunk)
                    if len(chunks) <= 3:
                        noise_samples.append(rms)
                    noise_floor = (sum(noise_samples) / len(noise_samples)) if noise_samples else 80.0
                    thresh = max(140.0, noise_floor * 2.5)
                    if rms >= thresh:
                        speech_started = True
                        silent_chunks = 0
                    elif speech_started:
                        silent_chunks += 1
                        if len(chunks) >= min_chunks and silent_chunks >= trailing_silence_chunks:
                            proc.terminate()
                            break
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        pcm = b"".join(chunks)
        if len(pcm) < 320:
            err = proc.stderr.read().decode("utf-8", errors="ignore")[-200:] if proc.stderr else ""
            raise RuntimeError(f"Microphone capture failed: {err or 'no audio data'}")
        return _normalize_pcm_wav(_pcm_to_wav(pcm))

    if sys.platform.startswith("linux") and shutil.which("arecord"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = Path(tf.name)
        try:
            dur = str(max(1, int(round(seconds))))
            cmd = ["arecord", "-d", dur, "-f", "S16_LE", "-r", "16000", "-c", "1", str(wav_path)]
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=seconds + 6)
            if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 100:
                err = proc.stderr.decode("utf-8", errors="ignore")[-200:]
                raise RuntimeError(f"Microphone capture failed: {err or 'no audio data'}")
            return _normalize_pcm_wav(wav_path.read_bytes())
        finally:
            wav_path.unlink(missing_ok=True)

    raise RuntimeError("No audio recorder found (install `alsa-utils` for `arecord` or `ffmpeg`).")


def transcribe_wav_sync(wav_bytes: bytes, keystore: SecureKeyStore) -> str:
    """Transcribe WAV audio using Gemini (preferred if key set) or OpenAI Whisper."""
    if not _wav_has_speech(wav_bytes):
        return ""
    prompt = (
        "Transcribe the spoken robot arm command verbatim in English. "
        "Common words: keys, mug, pen, wallet, airpods, wired earphones, mobile phone, "
        "red cube, green block, pick, place, move, closer to, on top of, home, stop. "
        "Return only the transcript text, or empty string if only background noise."
    )
    gem_key = keystore.get_key("gemini")
    if gem_key:
        from google import genai

        client = genai.Client(api_key=gem_key)
        res = client.interactions.create(
            model="gemini-3.8-flash",
            input=[
                {"type": "audio", "data": base64.b64encode(wav_bytes).decode("ascii"), "mime_type": "audio/wav"},
                {"type": "text", "text": prompt},
            ],
            store=False,
        )
        return (res.output_text or "").strip()

    oai_key = keystore.get_key("openai")
    if oai_key:
        boundary = "----DeskCoreAudioBoundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="cmd.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {oai_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            return json.loads(resp.read().decode("utf-8")).get("text", "").strip()

    raise RuntimeError("Voice transcription requires a Gemini or OpenAI API key (press Ctrl+K).")


VOICE_PRESETS: dict[str, tuple[str, str]] = {
    "female": ("en-US-AvaNeural", "Samantha"),
    "male": ("en-US-AndrewNeural", "Daniel"),
    "female_uk": ("en-GB-SoniaNeural", "Karen"),
    "male_uk": ("en-GB-RyanNeural", "Rishi"),
}


def _resolve_voice(voice: str | None) -> tuple[str, str]:
    v = (voice or os.environ.get("DESK_TTS_VOICE", "female")).strip()
    if v.lower() in VOICE_PRESETS:
        return VOICE_PRESETS[v.lower()]
    mac_fallback = os.environ.get("DESK_MAC_VOICE", "Samantha").strip() or "Samantha"
    return (v or "en-US-AvaNeural", mac_fallback)


def _edge_tts_speak_sync(clean: str, edge_voice: str) -> bool:
    if os.environ.get("DESK_TTS_LOCAL") == "1":
        return False
    player = shutil.which("afplay") or shutil.which("ffplay") or shutil.which("mpg123")
    if not player:
        return False
    try:
        import asyncio as _asyncio
        import edge_tts

        async def _synth(out_path: str) -> None:
            comm = edge_tts.Communicate(clean, edge_voice)
            await comm.save(out_path)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            mp3_path = tf.name
        try:
            _asyncio.run(_synth(mp3_path))
            cmd = [player, "-nodisp", "-autoexit", mp3_path] if player.endswith("ffplay") else [player, mp3_path]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        finally:
            try:
                os.unlink(mp3_path)
            except OSError:
                pass
    except Exception:
        return False


def speak_text_sync(text: str, voice: str | None = None) -> bool:
    """Speak `text` using natural neural TTS (`edge-tts`), with `say` / `pyttsx3` fallback."""
    if os.environ.get("DESK_TTS", "1") == "0":
        return False
    clean = " ".join(str(text).split()).strip()
    if not clean:
        return False
    edge_voice, mac_voice = _resolve_voice(voice)
    if _edge_tts_speak_sync(clean, edge_voice):
        return True
    if sys.platform == "darwin" and shutil.which("say"):
        subprocess.run(
            ["say", "-v", mac_voice, "-r", "175", clean],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    try:
        import pyttsx3

        eng = pyttsx3.init()
        eng.setProperty("rate", 175)
        eng.say(clean)
        eng.runAndWait()
        return True
    except Exception:
        return False


def _sanitize_reply(reply: str, thought: str = "") -> str:
    """Strip internal thinking/narration so only the concise final reply is shown or spoken."""
    import re as _re

    txt = _re.sub(r"<thinking>[\s\S]*?</thinking>", "", reply or "", flags=_re.IGNORECASE).strip()
    if thought and txt.startswith(thought.strip()):
        txt = txt[len(thought.strip()):].strip(" \n:-")
    txt = " ".join(txt.split())
    if not txt:
        return ""
    if len(txt.split()) > 18:
        sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
        if len(sentences) > 1:
            txt = sentences[-1]
    return txt


async def run_agent_command(
    command_text: str,
    provider: str,
    model_id: str,
    keystore: SecureKeyStore,
    fetch_images_fn: Callable[[], Awaitable[list[tuple[str, bytes]]]],
    tools_summary: str,
    call_tool_fn: Callable[[str, dict[str, Any]], Awaitable[Any]],
    max_steps: int = 8,
    step_fn: Callable[..., AgentStepResponse] | None = None,
) -> dict[str, Any]:
    """Run up to `max_steps` multimodal tool-calling turns for a user voice/text command."""
    import asyncio

    fn = step_fn or call_vision_model_step
    api_key = keystore.get_key(provider)
    history: list[str] = [f"User command: {command_text}"]
    executed: list[dict[str, Any]] = []
    final_reply = ""
    completed = False
    last_error = ""

    for _ in range(max_steps):
        images = await fetch_images_fn()
        prompt = "\n\n".join(history)
        step = await asyncio.to_thread(
            fn, provider, model_id, api_key, tools_summary, prompt, images
        )
        if step.thought:
            history.append(f"Thought: {step.thought}")

        step_failed = False
        for tc in step.tool_calls:
            try:
                res = await call_tool_fn(tc.tool, tc.args)
            except Exception as exc:
                res = {"ok": False, "error": str(exc)}
            failed = (
                res is False
                or (isinstance(res, dict) and res.get("ok") is False)
                or (isinstance(res, str) and "Error" in res)
            )
            if failed:
                step_failed = True
                if isinstance(res, dict):
                    last_error = str(res.get("error") or res.get("reason") or f"{tc.tool} failed")
                elif isinstance(res, str):
                    last_error = res[:180]
                else:
                    last_error = f"{tc.tool} returned False"
            executed.append({"tool": tc.tool, "args": tc.args, "result": res})
            history.append(f"Tool {tc.tool}({json.dumps(tc.args)}) -> {json.dumps(res, default=str)[:1200]}")
            if failed:
                break

        if not step_failed and step.reply and (step.done or not step.tool_calls):
            cleaned = _sanitize_reply(step.reply, step.thought)
            if cleaned:
                final_reply = cleaned

        if not step.tool_calls or (step.done and not step_failed):
            completed = True
            break

    if not final_reply:
        if completed:
            final_reply = "Done."
        elif last_error:
            final_reply = f"Failed before completion: {last_error}"
        else:
            final_reply = "Stopped after reaching step limit before completing the task."

    return {
        "reply": final_reply,
        "steps": executed,
        "provider": provider,
        "model": model_id,
    }
