"""Shared datatypes and Protocol contracts for DESK-CORE.

Allows parallel development across the team:
  - Abhimanyu + Scot: rig setup, camera calibration, homography, ray-plane depth (`calib/rig_calib.json`)
  - Suhail: Pink (Pinocchio) IK solver (`desk_core/ik_pink.py` or `PINK_IK_URL`)
  - Siddhartha: MCP server, Gemini 4 Hydrogen localize_3d, Robo-Jev, Dashboard
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Pose6D:
    x: float
    y: float
    z: float
    alpha: float = 0.0  # roll (rad)
    beta: float = 0.0   # pitch (rad)
    gamma: float = 0.0  # yaw (rad)


@dataclass
class SceneObstacle:
    name: str
    shape: str = "cylinder"  # cylinder | box
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius_m: float = 0.045
    height_m: float = 0.10
    size_xyz: tuple[float, float, float] | None = None


@dataclass
class Detection:
    label: str
    u_px: float
    v_px: float
    box_2d: tuple[int, int, int, int] | None = None  # [ymin, xmin, ymax, xmax] 0..1000 or px
    confidence: float = 1.0
    source: str = "gemini"
    pose6d: Pose6D | None = None


@dataclass
class IKRequest:
    x: float
    y: float
    z: float
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    q_curr_rad: list[float] = field(default_factory=list)
    obstacles: list[SceneObstacle] = field(default_factory=list)


@runtime_checkable
class IKSolver(Protocol):
    """Contract for Suhail's Pink IK solver (or fallback solver)."""

    def solve_ik(
        self,
        x: float,
        y: float,
        z: float,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        q_curr_rad: list[float] | None = None,
        obstacles: list[SceneObstacle] | None = None,
    ) -> list[float]:
        """Return target arm joint angles [j_1, ..., j_N] in radians."""
        ...


@runtime_checkable
class DepthEstimator(Protocol):
    """Contract for Scot/Abhimanyu's metric depth & pixel->world projection."""

    def get_metric_depth(self, x: float, y: float, camera: str = "top") -> float:
        """Return metric depth z (meters) along camera ray or height above camera for pixel (x, y)."""
        ...

    def pixel_to_world(self, x: float, y: float, z_m: float | None = None, camera: str = "top") -> tuple[float, float, float]:
        """Return 3D metric (X, Y, Z) in robot base frame."""
        ...
