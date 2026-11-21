"""IK solver adapter: plugs straight into Suhail's Pink solver when available.

Resolution order:
  1. External HTTP service if `PINK_IK_URL` env var is set (lets Suhail run Pink on his own laptop!).
  2. Local Python module `desk_core/ik_pink.py` exporting
     `solve_ik(x, y, z, alpha, beta, gamma, q_curr_rad=..., obstacles=...) -> list[float]`.
  3. Built-in 7-DoF damped kinematic fallback so offline MCP tool chains
     (`solve_ik -> move_joint`, `move_to_pose`, `pick_at`) run cleanly before Pink lands.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import urllib.request
from typing import Any

import numpy as np

from .interfaces import SceneObstacle
from .station_bridge import REPO_ROOT


def _fk_and_jac_nominal(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smooth 7-DoF serial arm forward kinematics + 6xN finite-diff Jacobian (nominal link lengths).

    Used ONLY when Suhail's Pink solver (`desk_core/ik_pink.py` or `PINK_IK_URL`) is absent.
    Joint convention: j1 yaw, j2 shoulder pitch, j3 elbow pitch, j4 pitch, j5 roll, j6 wrist pitch, j7 wrist roll.
    """
    n = len(q)
    # Pad/trim to 7 internally
    qq = np.zeros(7, dtype=float)
    qq[: min(7, n)] = q[: min(7, n)]

    def pose6(angles: np.ndarray) -> np.ndarray:
        j1, j2, j3, j4, j5, j6, j7 = angles
        l1, l2, l3, l4 = 0.08, 0.16, 0.16, 0.09
        r = l2 * math.cos(j2) + l3 * math.cos(j2 + j3) + l4 * math.cos(j2 + j3 + j4)
        z = l1 + l2 * math.sin(j2) + l3 * math.sin(j2 + j3) + l4 * math.sin(j2 + j3 + j4)
        x = r * math.cos(j1)
        y = r * math.sin(j1)
        alpha = j5 + j7
        beta = j2 + j3 + j4 + j6
        gamma = j1
        return np.array([x, y, z, alpha, beta, gamma], dtype=float)

    p0 = pose6(qq)
    J = np.zeros((6, n), dtype=float)
    eps = 1e-4
    for i in range(n):
        dq = qq.copy()
        dq[i] += eps
        J[:, i] = (pose6(dq) - p0) / eps
    return p0, J


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _fallback_dls_ik(
    x: float,
    y: float,
    z: float,
    alpha: float,
    beta: float,
    gamma: float,
    q_curr_rad: list[float],
    n_arm: int,
    bounds_rad: list[tuple[float, float]] | None = None,
) -> list[float]:
    q = np.array(q_curr_rad[:n_arm] if len(q_curr_rad) >= n_arm else [0.0] * n_arm, dtype=float)
    lo = np.array([b[0] for b in bounds_rad[:n_arm]], dtype=float) if bounds_rad else np.full(n_arm, -1.55)
    hi = np.array([b[1] for b in bounds_rad[:n_arm]], dtype=float) if bounds_rad else np.full(n_arm, 1.55)
    target = np.array([x, y, z, alpha, beta, gamma], dtype=float)
    lam = 1e-2
    for _ in range(40):
        curr, J = _fk_and_jac_nominal(q)
        err = target - curr
        err[3] = _wrap_angle(err[3])
        err[4] = _wrap_angle(err[4])
        err[5] = _wrap_angle(err[5])
        if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 2e-2:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(6), err)
        q = np.clip(q + 0.5 * dq, lo, hi)
    return [round(float(v), 5) for v in q]


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
            return "pink_local"
        return "nominal_dls_fallback"

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
        n_arm: int = 7,
        bounds_rad: list[tuple[float, float]] | None = None,
    ) -> list[float]:
        seed = list(q_curr_rad) if q_curr_rad else [0.0] * n_arm

        url = os.environ.get("PINK_IK_URL", "").strip()
        if url:
            payload = json.dumps({
                "x": x,
                "y": y,
                "z": z,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "q_curr": seed,
                "obstacles": [vars(o) for o in (obstacles or ())],
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data: Any = json.loads(resp.read().decode("utf-8"))
                angles = data["joints"] if isinstance(data, dict) else data
                return [float(v) for v in angles]

        if self._pink_fn is not None:
            angles = self._pink_fn(
                x=x,
                y=y,
                z=z,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                q_curr_rad=seed,
                obstacles=obstacles or [],
            )
            return [float(v) for v in angles]

        return _fallback_dls_ik(x, y, z, alpha, beta, gamma, seed, n_arm, bounds_rad=bounds_rad)
