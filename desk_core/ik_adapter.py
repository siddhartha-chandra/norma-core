"""IK solver adapter: plugs straight into a Pink IK solver when available.

Resolution order:
  1. External HTTP service if `PINK_IK_URL` env var is set (remote Pink IK server).
  2. Local Python module `desk_core/ik_pink.py` exporting
     `solve_ik(x, y, z, qx, qy, qz, qw, seed_joints_rad, obstacles) -> list[float]`.
  3. If neither backend is available, `solve_ik` raises an error.

KINEMATICS CONTRACT:
  - URDF: hardware/elrobot/simulation/elrobot_follower.urdf
  - World frame: base_link
  - EE frame: Gripper_Base_v1_1
  - Arm joints: ["rev_motor_01", "rev_motor_02", ..., "rev_motor_07"] (meters, radians)
  - Target orientation: quaternion (qx, qy, qz, qw); current seed: seed_joints_rad
  - URDF q=[0]*7 pose aligns with joint_zero_ticks + joint_signs in calibration JSON
"""
from __future__ import annotations

import importlib
import json
import math
import os
import urllib.request
from typing import Any

from .interfaces import SceneObstacle
from .station_bridge import REPO_ROOT

# Default arm joints matching URDF kinematic chain
ARM_JOINT_NAMES = (
    "rev_motor_01",
    "rev_motor_02",
    "rev_motor_03",
    "rev_motor_04",
    "rev_motor_05",
    "rev_motor_06",
    "rev_motor_07",
)


def _normalize_quaternion(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[float, float, float, float]:
    quaternion = (qx, qy, qz, qw)
    if not all(math.isfinite(value) for value in quaternion):
        raise ValueError("Target orientation quaternion must be finite")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError("Target orientation quaternion must be nonzero")
    return tuple(value / norm for value in quaternion)


class IKAdapter:
    def __init__(self) -> None:
        self._pink_fn = None
        self.reload()

    def reload(self) -> None:
        self._pink_fn = None
        if (REPO_ROOT / "desk_core" / "ik_pink.py").exists():
            try:
                mod = importlib.import_module("desk_core.ik_pink")
                importlib.reload(mod)
                self._pink_fn = getattr(mod, "solve_ik", None)
            except Exception:
                self._pink_fn = None

    def backend_name(self) -> str:
        if os.environ.get("PINK_IK_URL", "").strip():
            return f"http:{os.environ['PINK_IK_URL'].strip()}"
        if self._pink_fn is not None:
            try:
                self._pink_fn(0, 0, 0.1)  # Quick probe
                return "pink_local"
            except NotImplementedError:
                return "unavailable (local Pink solver not implemented)"
            except Exception:
                return "pink_local"
        return "unavailable (no local Pink solver or PINK_IK_URL)"

    def solve_ik(
        self,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
        seed_joints_rad: list[float] | None = None,
        obstacles: list[SceneObstacle] | None = None,
        n_arm: int = 7,
    ) -> list[float]:
        """Solve IK for an xyzw target quaternion and optional joint seed."""
        qx, qy, qz, qw = _normalize_quaternion(qx, qy, qz, qw)
        seed = list(seed_joints_rad) if seed_joints_rad is not None else [0.0] * n_arm
        if len(seed) != n_arm:
            raise ValueError(f"Expected {n_arm} seed joint angles, got {len(seed)}")
        if not all(math.isfinite(value) for value in seed):
            raise ValueError("Seed joint angles must be finite")

        url = os.environ.get("PINK_IK_URL", "").strip()
        if url:
            payload = json.dumps({
                "x": x,
                "y": y,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "seed_joints_rad": seed,
                "obstacles": [vars(o) for o in (obstacles or ())],
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data: Any = json.loads(resp.read().decode("utf-8"))
                angles = data["joints"] if isinstance(data, dict) else data
                result = [float(v) for v in angles]
                if len(result) != n_arm or not all(math.isfinite(v) for v in result):
                    raise ValueError(f"IK service must return {n_arm} finite joint angles")
                return result

        if self._pink_fn is None:
            raise RuntimeError(
                "No IK backend configured. Implement desk_core/ik_pink.py or set PINK_IK_URL."
            )
        try:
            angles = self._pink_fn(
                x=x,
                y=y,
                z=z,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
                seed_joints_rad=seed,
                obstacles=obstacles or [],
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                "Local Pink IK solver is not implemented. Implement desk_core/ik_pink.py "
                "or set PINK_IK_URL."
            ) from exc
        result = [float(v) for v in angles]
        if len(result) != n_arm or not all(math.isfinite(v) for v in result):
            raise ValueError(f"Pink solver must return {n_arm} finite joint angles")
        return result
