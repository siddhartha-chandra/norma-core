"""Minimal TypeSafe Jev (System One) client over the public HTTP API.

Reads the key from `TYPE_SAFE_API_KEY` (set it in `.env`). Uses only the standard library.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"


class JevUnavailable(RuntimeError):
    """Raised when the Jev key is missing or the service cannot be reached."""


def jev_api_key() -> str | None:
    return os.environ.get("TYPE_SAFE_API_KEY") or None


def noul_question(instructions: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def choice_question(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def system_one(state: str, questions: dict[str, dict[str, Any]], timeout_s: float = 8.0) -> dict[str, Any]:
    """Ask Jev `questions` about `state`; returns the `answers` mapping keyed by question id."""
    key = jev_api_key()
    if not key:
        raise JevUnavailable("TYPE_SAFE_API_KEY is not set")
    body = json.dumps({"state": state, "model": JEV_MODEL, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(
        JEV_URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise JevUnavailable(f"Jev request failed: {exc}") from exc
    return payload.get("answers", {})
