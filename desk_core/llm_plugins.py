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
    "gemini": {"name": "Google Gemini", "env_var": "GEMINI_API_KEY"},
    "claude": {"name": "Anthropic Claude", "env_var": "ANTHROPIC_API_KEY"},
    "openai": {"name": "OpenAI", "env_var": "OPENAI_API_KEY"},
}

VISION_MODELS: list[VisionModelSpec] = [
    VisionModelSpec("gemini", "gemini-4-hydrogen", "Gemini 4 Hydrogen (Vision + Robotics)"),
    VisionModelSpec("gemini", "gemini-3.8-flash", "Gemini 3.8 Flash (Vision)"),
    VisionModelSpec("gemini", "gemini-3.8-pro", "Gemini 3.8 Pro (Vision)"),
    VisionModelSpec("claude", "claude-sonnet-4-6", "Claude Sonnet 4.6 (Vision)"),
    VisionModelSpec("claude", "claude-opus-4-6", "Claude Opus 4.6 (Vision)"),
    VisionModelSpec("claude", "claude-haiku-4-5", "Claude Haiku 4.5 (Vision)"),
    VisionModelSpec("openai", "gpt-4.1", "GPT-4.1 (Vision)"),
    VisionModelSpec("openai", "gpt-4.1-mini", "GPT-4.1 Mini (Vision)"),
    VisionModelSpec("openai", "gpt-4o", "GPT-4o (Vision)"),
    VisionModelSpec("openai", "o4-mini", "o4-mini (Vision)"),
]

VISION_MODEL_MAP: dict[str, VisionModelSpec] = {
    f"{m.provider}:{m.model_id}": m for m in VISION_MODELS
}


def validate_vision_plugin(provider: str, model_id: str | None = None) -> VisionModelSpec | None:
    """Ensure only image-capable providers & models can be registered or selected."""
    p = provider.strip().lower()
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

    def get_key(self, provider: str) -> str:
        p = provider.strip().lower()
        if p not in VISION_PROVIDERS:
            return ""
        if self._keys.get(p):
            return self._keys[p]
        env_var = VISION_PROVIDERS[p]["env_var"]
        return os.environ.get(env_var, "").strip()

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
        '{"thought": "...", "tool_calls": [{"tool": "name", "args": {...}}], "reply": "...", "done": true/false}\n'
        "Rules:\n"
        "- Inspect the image(s) carefully before acting.\n"
        "- Prefer safe perception/check tools (`localize_3d`, `observe`, `get_state`, `robo_jev_guard`) before motion.\n"
        "- Keep `tool_calls` empty and set `done=true` when answering a visual question or after finishing.\n"
        f"Available MCP tools:\n{tools_summary}"
    )


def _extract_json_dict(raw_text: str) -> dict[str, Any]:
    txt = raw_text.strip()
    if txt.startswith("```"):
        lines = txt.splitlines()
        txt = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    return json.loads(txt)


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
    return AgentStepResponse.model_validate_json(interaction.output_text)


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


def _call_openai_step(
    model_id: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    images_jpeg: list[tuple[str, bytes]],
) -> AgentStepResponse:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for _cam, jpg in images_jpeg:
        b64 = base64.b64encode(jpg).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    res = _http_post_json(
        "https://api.openai.com/v1/chat/completions",
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
    p = provider.strip().lower()
    if p == "gemini":
        return _call_gemini_step(model_id, api_key, sys_prompt, user_prompt, images_jpeg)
    if p == "claude":
        return _call_claude_step(model_id, api_key, sys_prompt, user_prompt, images_jpeg)
    if p == "openai":
        return _call_openai_step(model_id, api_key, sys_prompt, user_prompt, images_jpeg)
    raise ValueError(f"Unsupported provider: {provider}")


def record_wav_sync(seconds: float = 4.0) -> bytes:
    """Record short mono 16 kHz WAV clip using `arecord` (Linux) or `ffmpeg` (macOS/Linux)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    try:
        dur = str(max(1, int(round(seconds))))
        if sys.platform.startswith("linux") and shutil.which("arecord"):
            cmd = ["arecord", "-d", dur, "-f", "S16_LE", "-r", "16000", "-c", "1", str(wav_path)]
        elif shutil.which("ffmpeg"):
            if sys.platform == "darwin":
                cmd = ["ffmpeg", "-y", "-f", "avfoundation", "-i", ":0", "-t", dur, "-ar", "16000", "-ac", "1", str(wav_path)]
            else:
                cmd = ["ffmpeg", "-y", "-f", "alsa", "-i", "default", "-t", dur, "-ar", "16000", "-ac", "1", str(wav_path)]
        else:
            raise RuntimeError("No audio recorder found (install `alsa-utils` for `arecord` or `ffmpeg`).")

        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=seconds + 5)
        if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 100:
            err = proc.stderr.decode("utf-8", errors="ignore")[-200:]
            raise RuntimeError(f"Microphone capture failed: {err or 'no audio data'}")
        return wav_path.read_bytes()
    finally:
        wav_path.unlink(missing_ok=True)


def transcribe_wav_sync(wav_bytes: bytes, keystore: SecureKeyStore) -> str:
    """Transcribe WAV audio using Gemini (preferred if key set) or OpenAI Whisper."""
    gem_key = keystore.get_key("gemini")
    if gem_key:
        from google import genai

        client = genai.Client(api_key=gem_key)
        res = client.interactions.create(
            model="gemini-3.8-flash",
            input=[
                {"type": "audio", "data": base64.b64encode(wav_bytes).decode("ascii"), "mime_type": "audio/wav"},
                {"type": "text", "text": "Transcribe the spoken robot command verbatim. Return only the transcript text."},
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


async def run_agent_command(
    command_text: str,
    provider: str,
    model_id: str,
    keystore: SecureKeyStore,
    fetch_images_fn: Callable[[], Awaitable[list[tuple[str, bytes]]]],
    tools_summary: str,
    call_tool_fn: Callable[[str, dict[str, Any]], Awaitable[Any]],
    max_steps: int = 5,
    step_fn: Callable[..., AgentStepResponse] | None = None,
) -> dict[str, Any]:
    """Run up to `max_steps` multimodal tool-calling turns for a user voice/text command."""
    import asyncio

    fn = step_fn or call_vision_model_step
    api_key = keystore.get_key(provider)
    history: list[str] = [f"User command: {command_text}"]
    executed: list[dict[str, Any]] = []
    final_reply = ""

    for _ in range(max_steps):
        images = await fetch_images_fn()
        prompt = "\n\n".join(history)
        step = await asyncio.to_thread(
            fn, provider, model_id, api_key, tools_summary, prompt, images
        )
        if step.thought:
            history.append(f"Thought: {step.thought}")
        if step.reply:
            final_reply = step.reply

        if not step.tool_calls or step.done:
            break

        for tc in step.tool_calls:
            res = await call_tool_fn(tc.tool, tc.args)
            executed.append({"tool": tc.tool, "args": tc.args, "result": res})
            history.append(f"Tool {tc.tool}({json.dumps(tc.args)}) -> {json.dumps(res, default=str)[:1200]}")

    return {
        "reply": final_reply or "Done.",
        "steps": executed,
        "provider": provider,
        "model": model_id,
    }
