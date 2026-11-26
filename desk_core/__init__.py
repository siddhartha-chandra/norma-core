"""DESK-CORE — Tripartite Embodied Desk Concierge for NormaCore Station."""

import os
from pathlib import Path

from dotenv import dotenv_values

# Load .env before any submodule reads its import-time settings (e.g. ROBOT_MAX_STEP_TICKS).
if os.environ.get("DESK_NO_DOTENV") != "1":
    for _k, _v in dotenv_values(Path(__file__).resolve().parents[1] / ".env").items():
        if _v is not None and not os.environ.get(_k, "").strip():
            os.environ[_k] = _v

from .homography import HomographyFit, fit_plane_homography, reanchor_plane
from .station_bridge import (
    LiveStationBridge,
    MockStationBridge,
    MotorState,
    RobotState,
    StationBridge,
    create_bridge,
)

__all__ = [
    "StationBridge",
    "MockStationBridge",
    "LiveStationBridge",
    "MotorState",
    "RobotState",
    "create_bridge",
    "HomographyFit",
    "fit_plane_homography",
    "reanchor_plane",
]
