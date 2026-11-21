"""Per-run tool-call logging for replay (DESK-CORE issue #3).

Every desk_core MCP server session gets its own directory:

    runs/<UTC %Y%m%dT%H%M%SZ>-<pid>/
        calls.jsonl   — one JSON record per line, in call order
        frames/       — JPEGs returned by observe(), kept out of the log

calls.jsonl doubles as a replay recipe: feed each `tool_call` record back
through an MCP `ClientSession.call_tool(tool, args)` in order to reproduce
the run against any bridge (mock or live). Lines are flushed + fsynced as
they are written, so a crash mid-session still leaves a readable trail.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_JSON_CHARS = 8000


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(obj: Any) -> Any:
    """Best-effort JSON-safe copy; exotic or huge payloads become short strings."""
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        return repr(obj)[:MAX_JSON_CHARS]
    if len(s) > MAX_JSON_CHARS:
        return {"_truncated": True, "_preview": s[: MAX_JSON_CHARS // 2], "_chars": len(s)}
    return json.loads(s)


class RunLogger:
    """Append-only JSONL recorder. The run directory is created lazily on the
    first call so handshake-only sessions (agents that never touch a tool) do
    not litter runs/ with empty dirs."""

    def __init__(self, runs_dir: str | os.PathLike, meta: dict[str, Any] | None = None):
        self._runs_dir = Path(runs_dir)
        self._meta = meta or {}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self._runs_dir / f"{stamp}-{os.getpid()}"
        self._fh = None
        self._seq = 0

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        (self.run_dir / "frames").mkdir(parents=True, exist_ok=True)
        self._fh = (self.run_dir / "calls.jsonl").open("a", encoding="utf-8")
        self._emit({"event": "run_start", "ts": _iso(), **self._meta})

    def record_call(
        self,
        tool: str,
        args: dict,
        result: Any = None,
        error: BaseException | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        self._ensure_open()
        self._seq += 1
        rec: dict[str, Any] = {
            "event": "tool_call",
            "seq": self._seq,
            "ts": _iso(),
            "tool": tool,
            "args": _jsonable(args),
            "ok": error is None,
            "duration_ms": round(duration_ms, 1),
        }
        if error is not None:
            rec["error"] = repr(error)
        else:
            payload, images = self._encode(result, tool)
            rec["result"] = payload
            if images:
                rec["images"] = images
        self._emit(rec)

    def _encode(self, obj: Any, tool: str, suffix: str = "") -> tuple[Any, list[str]]:
        """JSON-safe copy of a result; fastmcp Image blocks are saved under frames/."""
        from fastmcp.utilities.types import Image  # local import: module usable without a server

        if isinstance(obj, Image):
            rel = f"frames/{self._seq:03d}_{tool}{suffix}.jpg"
            data = obj.data
            if data is None and obj.path is not None:
                try:
                    data = Path(obj.path).read_bytes()
                except OSError:
                    data = None
            if data:
                (self.run_dir / rel).write_bytes(data)
                return {"_image": rel, "bytes": len(data)}, [rel]
            return {"_image": None}, []
        if isinstance(obj, (list, tuple)):
            items, imgs = [], []
            for i, el in enumerate(obj):
                enc, sub = self._encode(el, tool, f"_{i:02d}")
                items.append(enc)
                imgs.extend(sub)
            return items, imgs
        if isinstance(obj, dict):
            items, imgs = {}, []
            for k, v in obj.items():
                enc, sub = self._encode(v, tool, f"_{str(k)[:24]}")
                items[str(k)] = enc
                imgs.extend(sub)
            return items, imgs
        return _jsonable(obj), []

    def _emit(self, rec: dict) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            self._emit({"event": "run_end", "ts": _iso(), "calls": self._seq, "run_dir": str(self.run_dir)})
        finally:
            self._fh.close()
            self._fh = None
