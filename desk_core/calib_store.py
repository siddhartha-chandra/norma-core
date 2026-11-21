"""Calibration store, encoder<->radian conversion, and metric depth / ray-plane projection.

Reads `calib/rig_calib.json` (when dropped by Abhimanyu/Scot from #19 / #4 / #6).
If not yet present on disk, provides clean nominal defaults for 7-DoF arm + gripper
so MCP tools and pipelines remain runnable end-to-end offline.
"""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .station_bridge import MAX_ANGLE_STEP, REPO_ROOT, MotorState

TICKS_PER_RAD = (MAX_ANGLE_STEP + 1) / (2.0 * math.pi)
DEFAULT_CALIB_PATH = Path(os.environ.get("DESK_CALIB_PATH") or (REPO_ROOT / "calib" / "rig_calib.json"))


class CalibrationStore:
    def __init__(self, path: Path = DEFAULT_CALIB_PATH):
        self.path = Path(path)
        self.data: dict[str, Any] = {}
        self._custom_depth_fn = None
        self.reload()

    def reload(self) -> dict[str, Any]:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {}
        self._load_depth_hook()
        return self.status()

    def _load_depth_hook(self) -> None:
        self._custom_depth_fn = None
        hook = os.environ.get("DESK_DEPTH_HOOK", "").strip()
        if not hook and Path(REPO_ROOT / "desk_core" / "geom_depth.py").exists():
            hook = "desk_core.geom_depth:get_metric_depth"
        if not hook:
            return
        mod_name, _, fn_name = hook.partition(":")
        fn_name = fn_name or "get_metric_depth"
        try:
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)
            self._custom_depth_fn = getattr(mod, fn_name, None)
        except Exception:
            self._custom_depth_fn = None

    def num_arm_joints(self, total_motors: int) -> int:
        if "num_arm_joints" in self.data:
            return int(self.data["num_arm_joints"])
        env_n = os.environ.get("ROBOT_NUM_ARM_JOINTS", "").strip()
        if env_n:
            return int(env_n)
        # Convention: last motor is gripper (ElRobot: 7+1=8, SO-101: 5+1=6)
        return max(1, total_motors - 1)

    def _sign(self, motor_id: int) -> float:
        signs = self.data.get("joint_signs")
        if isinstance(signs, dict):
            return float(signs.get(str(motor_id), 1.0))
        if isinstance(signs, list) and 0 <= motor_id - 1 < len(signs):
            return float(signs[motor_id - 1])
        return 1.0

    def _zero_tick(self, m: MotorState) -> float:
        zeros = self.data.get("joint_zero_ticks")
        if isinstance(zeros, dict) and str(m.id) in zeros:
            return float(zeros[str(m.id)])
        if isinstance(zeros, list) and 0 <= m.id - 1 < len(zeros):
            return float(zeros[m.id - 1])
        return 0.5 * (m.range_min + m.range_max)

    def ticks_to_rad(self, motors: list[MotorState]) -> list[float]:
        n_arm = self.num_arm_joints(len(motors))
        out: list[float] = []
        for m in motors[:n_arm]:
            delta = (m.position - self._zero_tick(m)) / TICKS_PER_RAD
            out.append(round(self._sign(m.id) * delta, 6))
        return out

    def bounds_rad(self, motors: list[MotorState]) -> list[tuple[float, float]]:
        n_arm = self.num_arm_joints(len(motors))
        out: list[tuple[float, float]] = []
        for m in motors[:n_arm]:
            lo_t, hi_t = (m.range_min, m.range_max) if m.range_max > m.range_min else (0, MAX_ANGLE_STEP)
            z = self._zero_tick(m)
            s = self._sign(m.id)
            a = s * (lo_t - z) / TICKS_PER_RAD
            b = s * (hi_t - z) / TICKS_PER_RAD
            out.append((min(a, b), max(a, b)))
        return out

    def rad_to_ticks(self, joints_rad: list[float], motors: list[MotorState]) -> dict[int, int]:
        n_arm = self.num_arm_joints(len(motors))
        if len(joints_rad) != n_arm:
            raise ValueError(f"Expected {n_arm} arm joint angles, got {len(joints_rad)}")
        out: dict[int, int] = {}
        for m, angle in zip(motors[:n_arm], joints_rad):
            raw = self._zero_tick(m) + self._sign(m.id) * float(angle) * TICKS_PER_RAD
            tick = int(round(raw))
            lo, hi = (m.range_min, m.range_max) if m.range_max > m.range_min else (0, MAX_ANGLE_STEP)
            out[m.id] = max(lo, min(hi, tick))
        return out

    def validate_joint_limits(self, joints_rad: list[float], motors: list[MotorState]) -> tuple[bool, str | None]:
        n_arm = self.num_arm_joints(len(motors))
        if len(joints_rad) != n_arm:
            return False, f"expected {n_arm} angles, got {len(joints_rad)}"
        for m, angle in zip(motors[:n_arm], joints_rad):
            if not math.isfinite(angle):
                return False, f"non-finite angle for motor {m.id}"
            raw = self._zero_tick(m) + self._sign(m.id) * float(angle) * TICKS_PER_RAD
            lo, hi = (m.range_min, m.range_max) if m.range_max > m.range_min else (0, MAX_ANGLE_STEP)
            if raw < lo - 40 or raw > hi + 40:
                return False, f"motor {m.id} target tick {raw:.0f} outside calibrated range [{lo}, {hi}]"
        return True, None

    def _K(self, camera: str) -> np.ndarray:
        key = f"K_{camera}"
        if key in self.data:
            return np.asarray(self.data[key], dtype=float).reshape(3, 3)
        # Reasonable nominal pinhole K for 640x480 UVC webcam (~65 deg HFOV)
        return np.array([[560.0, 0.0, 320.0], [0.0, 560.0, 240.0], [0.0, 0.0, 1.0]], dtype=float)

    def _T_world_cam(self, camera: str) -> np.ndarray | None:
        key = f"T_world_cam_{camera}"
        if key in self.data:
            return np.asarray(self.data[key], dtype=float).reshape(4, 4)
        return None

    def z_table(self) -> float:
        return float(self.data.get("z_table", 0.0))

    def get_metric_depth(self, x: float, y: float, camera: str = "top") -> float:
        """Estimate metric depth z (meters from camera frame) for pixel (x, y).

        Uses custom depth hook if plugged in, else exact ray-plane intersection
        with calibrated T_world_cam + z_table, else nominal camera-to-tabletop ray geometry.
        """
        if self._custom_depth_fn is not None:
            return float(self._custom_depth_fn(x, y, camera=camera))

        K = self._K(camera)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        ray_cam = np.array([(float(x) - cx) / fx, (float(y) - cy) / fy, 1.0], dtype=float)

        T = self._T_world_cam(camera)
        if T is not None:
            R = T[:3, :3]
            t = T[:3, 3]
            ray_world = R @ ray_cam
            if abs(ray_world[2]) > 1e-6:
                s = (self.z_table() - t[2]) / ray_world[2]
                if s > 0:
                    p_cam = s * ray_cam
                    return round(float(p_cam[2]), 5)

        # Nominal tabletop fallback when hand-eye extrinsics aren't calibrated yet
        h_cam = float(self.data.get(f"cam_height_{camera}_m", 0.62 if camera == "top" else 0.25))
        return round(float(h_cam), 5)

    def pixel_to_world(self, x: float, y: float, z_m: float | None = None, camera: str = "top") -> tuple[float, float, float]:
        """Project 2D pixel (x, y) into metric robot base-frame (X, Y, Z) in meters."""
        H = self.data.get("H_table")
        if H is not None and camera == "top" and z_m is None:
            H_mat = np.asarray(H, dtype=float).reshape(3, 3)
            pt = H_mat @ np.array([float(x), float(y), 1.0])
            if abs(pt[2]) > 1e-8:
                return (round(float(pt[0] / pt[2]), 5), round(float(pt[1] / pt[2]), 5), round(self.z_table(), 5))

        K = self._K(camera)
        ray_cam = np.array([(float(x) - K[0, 2]) / K[0, 0], (float(y) - K[1, 2]) / K[1, 1], 1.0], dtype=float)
        T = self._T_world_cam(camera)
        if T is not None:
            R, t = T[:3, :3], T[:3, 3]
            target_z = self.z_table() if z_m is None else float(z_m)
            ray_w = R @ ray_cam
            s = (target_z - t[2]) / ray_w[2] if abs(ray_w[2]) > 1e-6 else self.get_metric_depth(x, y, camera)
            pw = t + max(0.05, s) * ray_w
            return (round(float(pw[0]), 5), round(float(pw[1]), 5), round(float(pw[2]), 5))

        # Nominal overhead pixel scaling fallback (~1.2 mm/px at 0.62 m above center)
        depth = self.get_metric_depth(x, y, camera)
        X_robot = 0.25 - ((float(y) - K[1, 2]) / K[1, 1]) * depth
        Y_robot = -((float(x) - K[0, 2]) / K[0, 0]) * depth
        Z_robot = self.z_table() if z_m is None else float(z_m)
        return (round(float(X_robot), 5), round(float(Y_robot), 5), round(float(Z_robot), 5))

    def status(self) -> dict[str, Any]:
        return {
            "calib_file": str(self.path),
            "calib_loaded": self.path.exists(),
            "arm_type": self.data.get("arm_type", "elrobot-7dof-nominal"),
            "has_K_top": "K_top" in self.data,
            "has_T_world_cam_top": "T_world_cam_top" in self.data,
            "has_H_table": "H_table" in self.data,
            "has_depth_hook": self._custom_depth_fn is not None,
            "z_table": self.z_table(),
        }
