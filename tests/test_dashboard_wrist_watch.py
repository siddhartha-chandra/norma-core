"""Dashboard wrist watch latch: arm motion makes a wrist watch stale until Localize + Arm Watch."""
import pytest
from fastapi.testclient import TestClient

from desk_core import dashboard
from desk_core import mcp_server as srv


@pytest.fixture(autouse=True)
def _mock_bridge_only(monkeypatch):
    """Never reach a live Station: these tests drive Check Reflex, which can abort the arm."""
    monkeypatch.setenv("DESK_MOCK", "1")
    monkeypatch.setenv("STATION_HOST", "")
    monkeypatch.setattr(srv, "_bridge", None)
    yield


def _arm_wrist_watch(client: TestClient) -> None:
    assert client.post("/api/localize", json={"query": "mug", "camera": "wrist", "use_gemini": False}).status_code == 200
    dashboard._last_detection = {**dashboard._last_detection, "found": True, "pixel": [320.0, 240.0],
                                 "depth_m": 0.3, "camera": "wrist"}
    assert client.post("/api/robo_jev/arm").status_code == 200


def test_wrist_watch_latches_stale_until_relocalized():
    client = TestClient(dashboard.app)
    _arm_wrist_watch(client)
    armed = dict(dashboard._armed_positions)

    dashboard._armed_positions = {mid: pos + 100 for mid, pos in armed.items()}
    assert client.post("/api/robo_jev/check").json()["reason"] == "wrist_moved_relocalize"
    assert client.get("/api/telemetry").json()["robo_jev"]["reason"] == "wrist_moved_relocalize"

    dashboard._armed_positions = armed
    assert client.post("/api/robo_jev/check").json()["reason"] == "wrist_moved_relocalize"
    assert client.post("/api/robo_jev/arm").status_code == 409

    _arm_wrist_watch(client)
    assert client.post("/api/robo_jev/check").json()["reason"] != "wrist_moved_relocalize"


def test_arm_watch_refuses_wrist_detection_after_arm_motion():
    client = TestClient(dashboard.app)
    _arm_wrist_watch(client)
    dashboard._detection_positions = {mid: pos + 100 for mid, pos in dashboard._detection_positions.items()}
    res = client.post("/api/robo_jev/arm")
    assert res.status_code == 409
    assert "Localize" in res.json()["detail"]


def test_jaw_motion_does_not_stale_wrist_watch():
    client = TestClient(dashboard.app)
    _arm_wrist_watch(client)
    jaw_id = max(dashboard._armed_positions) + 1
    dashboard._armed_positions = {**dashboard._armed_positions, jaw_id: -10_000}
    assert client.post("/api/robo_jev/check").json()["reason"] != "wrist_moved_relocalize"
