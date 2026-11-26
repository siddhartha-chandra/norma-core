"""desk_core.command_planner: Jev routing, slot extraction, template expansion (Jev and localizer stubbed)."""
import asyncio

import pytest

from desk_core import command_planner as cp
from desk_core.jev_client import JevUnavailable


def _answers(motion: float, choice: str, confidence: float = 0.9) -> dict:
    return {
        "needs_motion": {"type": "noul", "noul": motion},
        "template": {"type": "choice", "choice": choice, "confidence": confidence},
    }


def _localizer(table: dict[str, tuple[float, float, float] | None]):
    calls: list[str] = []

    async def localize(query: str) -> dict:
        calls.append(query)
        xyz = table.get(query)
        if xyz is None:
            return {"found": False}
        return {"found": True, "world": {"x": xyz[0], "y": xyz[1], "z": xyz[2]}}

    localize.calls = calls
    return localize


def _run(command: str, answers: dict | None, localize, monkeypatch) -> dict:
    if answers is not None:
        monkeypatch.setattr(cp, "system_one", lambda state, questions, timeout_s: answers)
    else:
        def _raise(*a, **k):
            raise JevUnavailable("TYPE_SAFE_API_KEY is not set")
        monkeypatch.setattr(cp, "system_one", _raise)
    return asyncio.run(cp.plan_command(command, localize=localize))


@pytest.mark.parametrize(
    "template, command, expected",
    [
        ("place_next_to", "place keys next to the red box", {"target": "keys", "reference": "red box"}),
        ("place_next_to", "Please put the mug beside the tray.", {"target": "mug", "reference": "tray"}),
        ("pick_and_place", "pick up the pen and put it on the tray", {"target": "pen", "destination": "tray"}),
        ("pick_and_place", "put the wallet in the blue bin", {"target": "wallet", "destination": "blue bin"}),
        ("go_home", "go home", {}),
    ],
)
def test_extract_slots(template, command, expected):
    assert cp.extract_slots(template, command) == expected


def test_extract_slots_rejects_unparseable_command():
    assert cp.extract_slots("place_next_to", "place the keys") is None


def test_place_next_to_expands_to_pick_then_offset_place(monkeypatch):
    loc = _localizer({"keys": (0.30, 0.10, 0.05), "red box": (0.40, -0.02, 0.05)})
    plan = _run("place keys next to the red box", _answers(0.95, "place_next_to"), loc, monkeypatch)

    assert plan["route"] == "motion"
    assert plan["template"] == "place_next_to"
    assert loc.calls == ["keys", "red box"]
    dx, dy = cp.NEXT_TO_OFFSET_M
    assert plan["steps"] == [
        {"tool": "pick_at", "args": {"x": 0.30, "y": 0.10, "z": 0.05}},
        {"tool": "place_at", "args": {"x": pytest.approx(0.40 + dx), "y": pytest.approx(-0.02 + dy), "z": 0.05}},
    ]


def test_pick_and_place_uses_destination_point(monkeypatch):
    loc = _localizer({"pen": (0.21, -0.02, 0.05), "tray": (0.34, 0.09, 0.05)})
    plan = _run("pick up the pen and put it on the tray", _answers(0.9, "pick_and_place"), loc, monkeypatch)

    assert plan["route"] == "motion"
    assert [s["tool"] for s in plan["steps"]] == ["pick_at", "place_at"]
    assert plan["steps"][1]["args"] == {"x": 0.34, "y": 0.09, "z": 0.05}


def test_grounded_pick_arms_the_target_crop_before_moving(monkeypatch):
    async def localize(query):
        return {"found": True, "pixel": [320.0, 240.0], "world": {"x": 0.3, "y": 0.1, "z": 0.05}}

    plan = _run("place keys next to the red box", _answers(0.95, "place_next_to"), localize, monkeypatch)
    assert plan["steps"][0] == {
        "tool": "robo_jev_arm_watch", "args": {"x": 320.0, "y": 240.0, "query": "keys"},
    }
    assert plan["steps"][1]["tool"] == "pick_at"


def test_go_home_needs_no_grounding(monkeypatch):
    loc = _localizer({})
    plan = _run("go home", _answers(0.97, "go_home"), loc, monkeypatch)

    assert plan["route"] == "motion"
    assert plan["steps"] == [{"tool": "go_home", "args": {"look_at_table": True}}]
    assert loc.calls == []


def test_non_motion_command_skips_grounding(monkeypatch):
    loc = _localizer({})
    plan = _run("what do you see on the desk?", _answers(0.05, "none"), loc, monkeypatch)

    assert plan == {"route": "no_motion"}
    assert loc.calls == []


def test_low_confidence_template_is_unsupported(monkeypatch):
    loc = _localizer({"keys": (0.3, 0.1, 0.05), "red box": (0.4, 0.0, 0.05)})
    plan = _run("place keys next to the red box", _answers(0.9, "place_next_to", confidence=0.3), loc, monkeypatch)

    assert plan["route"] == "unsupported"
    assert loc.calls == []


def test_unparseable_command_is_unsupported(monkeypatch):
    loc = _localizer({})
    plan = _run("make the arm dance", _answers(0.8, "pick_and_place"), loc, monkeypatch)

    assert plan["route"] == "unsupported"
    assert loc.calls == []


def test_missing_object_is_not_found(monkeypatch):
    loc = _localizer({"keys": (0.3, 0.1, 0.05)})
    plan = _run("place keys next to the red box", _answers(0.95, "place_next_to"), loc, monkeypatch)

    assert plan["route"] == "not_found"
    assert "red box" in plan["reason"]


def test_missing_key_reports_unavailable(monkeypatch):
    monkeypatch.delenv("TYPE_SAFE_API_KEY", raising=False)
    loc = _localizer({})
    plan = asyncio.run(cp.plan_command("go home", localize=loc))

    assert plan["route"] == "unavailable"
    assert loc.calls == []


def test_opencv_fallback_detection_is_rejected(monkeypatch):
    async def loc(name: str):
        return {
            "found": True,
            "backend": "opencv-fallback",
            "world": {"x": 0.3, "y": 0.1, "z": 0.05},
        }

    plan = _run("place keys next to the red box", _answers(0.95, "place_next_to"), loc, monkeypatch)
    assert plan["route"] == "not_found"
    assert "opencv-fallback" in plan["reason"]
