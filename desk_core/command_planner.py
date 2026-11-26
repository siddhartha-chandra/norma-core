"""Route a natural-language command to a fixed motion template, or to a non-motion path.

Jev makes the judgments (does this need arm motion? which template fits?). Code extracts the
object names, grounds them with `localize`, and expands the chosen template into literal
`execute_sequence` steps. Planning never moves the arm.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable

from .jev_client import JevUnavailable, choice_question, noul_question, system_one

MIN_CONFIDENCE = 0.5
NEXT_TO_OFFSET_M = (0.0, 0.08)  # placeholder: 8 cm from the reference centre on the table plane

TEMPLATES: dict[str, str] = {
    "pick_and_place": "Pick up one object and put it on or in another object.",
    "place_next_to": "Put one object next to, beside, or near another object.",
    "go_home": "Return the robot arm to its resting home pose.",
    "sort_hive": "Sort the Hive board game tiles by color, then stack each insect type.",
    "none": "Not a physical action the arm can do with these templates, e.g. a question about the scene or small talk.",
}

_PLACE_NEXT_TO = re.compile(
    r"^\s*(?:please\s+)?(?:place|put|set|move)\s+(?:the\s+)?(?P<target>.+?)\s+(?:next to|beside|near)\s+(?:the\s+)?(?P<reference>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_PICK_AND_PLACE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:pick up|pick|grab|take)\s+(?:the\s+)?(?P<target>.+?)\s+(?:and\s+)?(?:put|place)\s+(?:it\s+)?(?:on|in|into|onto)"
    r"|(?:place|put)\s+(?:the\s+)?(?P<target2>.+?)\s+(?:on|in|into|onto)\s+)\s*(?:the\s+)?(?P<destination>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_SORT_HIVE = re.compile(r"(?:please\s+)?sort\s+(?:the\s+)?hive\s+tiles[.!]?", re.IGNORECASE)
_HEADING_OUT = re.compile(r"i(?:['’]m| am)\s+heading\s+out[.!]?", re.IGNORECASE)

Localize = Callable[[str], Awaitable[dict[str, Any]]]


def extract_slots(template: str, command: str) -> dict[str, str] | None:
    """Pull object names out of the command with fixed patterns. None when the command does not fit."""
    if template in {"go_home", "sort_hive"}:
        return {}
    if template == "place_next_to":
        m = _PLACE_NEXT_TO.match(command)
        return {"target": m.group("target").strip(), "reference": m.group("reference").strip()} if m else None
    if template == "pick_and_place":
        m = _PICK_AND_PLACE.match(command)
        if not m:
            return None
        target = m.group("target") or m.group("target2")
        return {"target": target.strip(), "destination": m.group("destination").strip()}
    return None


def _point(found: dict[str, Any]) -> dict[str, float]:
    w = found["world"]
    return {"x": float(w["x"]), "y": float(w["y"]), "z": float(w["z"])}


def _steps(template: str, grounded: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    if template == "go_home":
        return [{"tool": "go_home", "args": {"look_at_table": True}}]
    if template == "sort_hive":
        return [{"tool": "hive_sort", "args": {"execute": True}}]
    if template == "pick_and_place":
        t, d = grounded["target"], grounded["destination"]
        return [
            {"tool": "pick_at", "args": {"x": t["x"], "y": t["y"], "z": t["z"]}},
            {"tool": "place_at", "args": {"x": d["x"], "y": d["y"], "z": d["z"]}},
        ]
    if template == "place_next_to":
        t, r = grounded["target"], grounded["reference"]
        dx, dy = NEXT_TO_OFFSET_M
        return [
            {"tool": "pick_at", "args": {"x": t["x"], "y": t["y"], "z": t["z"]}},
            {"tool": "place_at", "args": {"x": r["x"] + dx, "y": r["y"] + dy, "z": r["z"]}},
        ]
    raise ValueError(f"no steps for template {template!r}")


def _unpack(answers: dict[str, Any]) -> tuple[bool, str, float]:
    motion = float(answers["needs_motion"]["noul"]) >= MIN_CONFIDENCE
    tpl = answers["template"]
    return motion, str(tpl.get("choice", "none")), float(tpl.get("confidence", 0.0))


async def plan_command(command: str, localize: Localize, timeout_s: float = 8.0) -> dict[str, Any]:
    """Return {route, ...}. route is 'motion' (with template, slots, grounding, steps), or
    'no_motion', 'unsupported', 'not_found', or 'unavailable' (caller should use its LLM path)."""
    if _SORT_HIVE.fullmatch(command.strip()):
        return {"route": "motion", "template": "sort_hive", "slots": {}, "grounding": {},
                "steps": _steps("sort_hive", {})}
    if _HEADING_OUT.fullmatch(command.strip()):
        return {"route": "motion", "template": "heading_out", "slots": {}, "grounding": {},
                "steps": [{"tool": "heading_out", "args": {"execute": True}}]}

    questions = {
        "needs_motion": noul_question("The command asks the robot arm to physically move, pick up, or place an object."),
        "template": choice_question("Which robot action template matches the command?", TEMPLATES),
    }
    try:
        answers = await asyncio.to_thread(system_one, command, questions, timeout_s)
    except JevUnavailable as exc:
        return {"route": "unavailable", "reason": str(exc)}

    motion, template, confidence = _unpack(answers)
    if not motion:
        return {"route": "no_motion"}
    if template == "none" or confidence < MIN_CONFIDENCE:
        return {"route": "unsupported", "reason": f"template {template!r} (confidence {confidence:.2f})"}

    slots = extract_slots(template, command)
    if slots is None:
        return {"route": "unsupported", "reason": f"could not read object names for {template!r}"}

    grounded: dict[str, dict[str, float]] = {}
    watch_step = []
    for role, name in slots.items():
        found = await localize(name)
        if not found.get("found"):
            return {"route": "not_found", "reason": f"could not locate {name!r}"}
        if found.get("backend") == "opencv-fallback":
            return {"route": "not_found", "reason": f"could not semantically confirm {name!r} (opencv-fallback)"}
        grounded[role] = _point(found)
        if role == "target" and found.get("pixel") is not None:
            x, y = found["pixel"]
            watch_step = [{"tool": "robo_jev_arm_watch", "args": {"x": float(x), "y": float(y), "query": name}}]

    return {
        "route": "motion",
        "template": template,
        "slots": slots,
        "grounding": grounded,
        "steps": watch_step + _steps(template, grounded),
    }
