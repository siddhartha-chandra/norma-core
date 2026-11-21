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

import cv2
import numpy as np

from .station_bridge import MAX_ANGLE_STEP, REPO_ROOT, MotorState
from .urdf_fk import DEFAULT_ARM_JOINTS, DEFAULT_ELROBOT_URDF, load_urdf

TICKS_PER_RAD = (MAX_ANGLE_STEP + 1) / (2.0 * math.pi)
RIG_CALIB_PATH = REPO_ROOT / "calib" / "rig_calib.json"


def resolve_default_calib_path() -> Path:
    """Resolve active calibration JSON path.

    Priority when `DESK_CALIB_PATH` env var is unset:
      1. `calib/rig_calib.json` if it exists and has `calibrated == true`.
      2. Newest `calib_frames/*/camera_calibration.json` if present (Scot's intrinsics output).
      3. `calib/rig_calib.json` (nominal tracked contract).
    """
    env_p = os.environ.get("DESK_CALIB_PATH", "").strip()
    if env_p:
        return Path(env_p)
    if RIG_CALIB_PATH.exists():
        try:
            rig_raw = json.loads(RIG_CALIB_PATH.read_text(encoding="utf-8"))
            if rig_raw.get("calibrated"):
                return RIG_CALIB_PATH
        except Exception:
            pass
    cam_calibs = sorted(
        (REPO_ROOT / "calib_frames").glob("*/camera_calibration.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if cam_calibs:
        return cam_calibs[-1]
    return RIG_CALIB_PATH


DEFAULT_CALIB_PATH = resolve_default_calib_path()


def _normalize_calib_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize either `rig_calib.json` or nested `{wrist, scene}` `camera_calibration.json`."""
    data = dict(raw)
    cam_aliases = {"top": ("top", "scene"), "wrist": ("wrist",)}
    for target_cam, aliases in cam_aliases.items():
        for alias in aliases:
            block = raw.get(alias)
            if not isinstance(block, dict):
                continue
            if f"K_{target_cam}" not in data and "camera_matrix" in block:
                data[f"K_{target_cam}"] = block["camera_matrix"]
            if f"dist_{target_cam}" not in data and "distortion_coefficients" in block:
                data[f"dist_{target_cam}"] = block["distortion_coefficients"]
            if f"image_size_{target_cam}" not in data and "image_size" in block:
                data[f"image_size_{target_cam}"] = block["image_size"]
            if f"T_world_cam_{target_cam}" not in data and "T_world_cam" in block:
                data[f"T_world_cam_{target_cam}"] = block["T_world_cam"]
            if f"T_ee_cam_{target_cam}" not in data and "T_ee_cam" in block:
                data[f"T_ee_cam_{target_cam}"] = block["T_ee_cam"]
            if target_cam == "top":
                if "z_table" not in data and "z_table" in block:
                    data["z_table"] = block["z_table"]
                if "H_table" not in data and "H_table" in block:
                    data["H_table"] = block["H_table"]
    if "num_arm_joints" not in data:
        data["num_arm_joints"] = 7
    if "arm_type" not in data and ("wrist" in raw or "scene" in raw):
        data["arm_type"] = "elrobot"
    return data


class CalibrationStore:
    def __init__(self, path: Path | None = None):
        self._explicit_path = Path(path) if path is not None else None
        self.path = self._explicit_path or resolve_default_calib_path()
        self.data: dict[str, Any] = {}
        self._custom_depth_fn = None
        self.reload()

    def reload(self) -> dict[str, Any]:
        if self._explicit_path is None:
            self.path = resolve_default_calib_path()
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.data = _normalize_calib_dict(raw)
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

    def hardware_config(self) -> dict[str, str]:
        """Effective hardware routing IDs: non-empty env vars take priority over rig_calib.json."""
        hw = self.data.get("hardware") if isinstance(self.data.get("hardware"), dict) else {}
        return {
            "bus_serial": (os.environ.get("STATION_BUS_SERIAL", "").strip() or str(hw.get("bus_serial", "")).strip()),
            "camera_top": (os.environ.get("CAMERA_TOP", "").strip() or str(hw.get("camera_top", "")).strip()),
            "camera_wrist": (os.environ.get("CAMERA_WRIST", "").strip() or str(hw.get("camera_wrist", "")).strip()),
        }

    def _K(self, camera: str) -> np.ndarray:
        key = f"K_{camera}"
        if key in self.data:
            return np.asarray(self.data[key], dtype=float).reshape(3, 3)
        # Reasonable nominal pinhole K for 640x480 UVC webcam (~65 deg HFOV)
        return np.array([[560.0, 0.0, 320.0], [0.0, 560.0, 240.0], [0.0, 0.0, 1.0]], dtype=float)

    def _dist(self, camera: str) -> np.ndarray:
        key = f"dist_{camera}"
        if key in self.data and self.data[key] is not None:
            return np.asarray(self.data[key], dtype=float).reshape(-1)
        return np.zeros(5, dtype=float)

    def _ray_cam(self, x: float, y: float, camera: str) -> np.ndarray:
        """Normalized unit-Z camera ray for raw image pixel (x, y), undistorted when dist_{camera} is present."""
        K = self._K(camera)
        dist = self._dist(camera)
        if np.any(np.abs(dist) > 1e-12):
            pts = np.array([[[float(x), float(y)]]], dtype=np.float64)
            und = cv2.undistortPoints(pts, K, dist)
            xn, yn = float(und[0, 0, 0]), float(und[0, 0, 1])
        else:
            xn = (float(x) - K[0, 2]) / K[0, 0]
            yn = (float(y) - K[1, 2]) / K[1, 1]
        return np.array([xn, yn, 1.0], dtype=float)

    def wrist_mount_link(self) -> str:
        return str(self.data.get("wrist_mount_link") or self.data.get("ee_link") or "Gripper_Base_v1_1")

    def world_frame(self) -> str:
        return str(self.data.get("world_frame") or "base_link")

    def joint_names(self) -> list[str]:
        names = self.data.get("joint_names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
        return list(DEFAULT_ARM_JOINTS)

    def urdf_path(self) -> Path:
        raw = str(self.data.get("urdf_path") or "").strip()
        if raw:
            p = Path(raw)
            return p if p.is_absolute() else (REPO_ROOT / p)
        return DEFAULT_ELROBOT_URDF

    def fk_ee_pose(self, q_rad: list[float] | np.ndarray, target_link: str | None = None) -> np.ndarray:
        """Compute 4x4 homogeneous transform ``T_world_ee(q_rad)`` from the ElRobot URDF."""
        urdf = load_urdf(self.urdf_path())
        return urdf.fk(
            q_rad=q_rad,
            target_link=target_link or self.wrist_mount_link(),
            base_link=self.world_frame(),
            joint_names=self.joint_names(),
        )

    def T_world_cam(self, camera: str = "top", q_rad: list[float] | np.ndarray | None = None) -> np.ndarray | None:
        """Return 4x4 transform mapping points from camera frame into world (``base_link``).

        - For static cameras (``top``), reads ``T_world_cam_top``.
        - For robot-mounted cameras (``wrist``), composes ``T_world_ee(q) @ T_ee_cam_wrist``
          via URDF forward kinematics unless a static ``T_world_cam_{camera}`` override exists.
        """
        key = f"T_world_cam_{camera}"
        if key in self.data:
            return np.asarray(self.data[key], dtype=float).reshape(4, 4)

        ee_key = f"T_ee_cam_{camera}"
        if ee_key in self.data:
            T_ee_cam = np.asarray(self.data[ee_key], dtype=float).reshape(4, 4)
            n_arm = len(self.joint_names())
            q = list(q_rad) if q_rad is not None else [0.0] * n_arm
            T_world_ee = self.fk_ee_pose(q, target_link=self.wrist_mount_link())
            return T_world_ee @ T_ee_cam
        return None

    def _T_world_cam(self, camera: str, q_rad: list[float] | np.ndarray | None = None) -> np.ndarray | None:
        return self.T_world_cam(camera=camera, q_rad=q_rad)

    def z_table(self) -> float:
        return float(self.data.get("z_table", 0.0))

    def get_metric_depth(
        self,
        x: float,
        y: float,
        camera: str = "top",
        q_rad: list[float] | np.ndarray | None = None,
    ) -> float:
        """Estimate metric depth z (meters from camera frame) for pixel (x, y).

        Uses custom depth hook if plugged in, else undistorted ray-plane intersection
        with calibrated T_world_cam (or T_world_ee(q) @ T_ee_cam_wrist) + z_table,
        else nominal camera-to-tabletop ray geometry.
        """
        if self._custom_depth_fn is not None:
            return float(self._custom_depth_fn(x, y, camera=camera))

        ray_cam = self._ray_cam(x, y, camera)
        T = self.T_world_cam(camera, q_rad=q_rad)
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

    def pixel_to_world(
        self,
        x: float,
        y: float,
        z_m: float | None = None,
        camera: str = "top",
        q_rad: list[float] | np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Project 2D pixel (x, y) into metric robot base-frame (X, Y, Z) in meters."""
        H = self.data.get("H_table")
        if H is not None and camera == "top" and z_m is None:
            H_mat = np.asarray(H, dtype=float).reshape(3, 3)
            pt = H_mat @ np.array([float(x), float(y), 1.0])
            if abs(pt[2]) > 1e-8:
                return (round(float(pt[0] / pt[2]), 5), round(float(pt[1] / pt[2]), 5), round(self.z_table(), 5))

        ray_cam = self._ray_cam(x, y, camera)
        T = self.T_world_cam(camera, q_rad=q_rad)
        if T is not None:
            R, t = T[:3, :3], T[:3, 3]
            target_z = self.z_table() if z_m is None else float(z_m)
            ray_w = R @ ray_cam
            s = (target_z - t[2]) / ray_w[2] if abs(ray_w[2]) > 1e-6 else self.get_metric_depth(x, y, camera, q_rad=q_rad)
            pw = t + max(0.05, s) * ray_w
            return (round(float(pw[0]), 5), round(float(pw[1]), 5), round(float(pw[2]), 5))

        # Nominal overhead pixel scaling fallback (uses undistorted ray when dist_top is calibrated)
        depth = self.get_metric_depth(x, y, camera, q_rad=q_rad)
        X_robot = 0.25 - float(ray_cam[1]) * depth
        Y_robot = -float(ray_cam[0]) * depth
        Z_robot = self.z_table() if z_m is None else float(z_m)
        return (round(float(X_robot), 5), round(float(Y_robot), 5), round(float(Z_robot), 5))

    def project_world_to_pixel(
        self,
        point_world: list[float] | tuple[float, float, float] | np.ndarray,
        camera: str = "top",
        q_rad: list[float] | np.ndarray | None = None,
    ) -> tuple[float, float] | None:
        """Project a 3D world point (``base_link`` meters) to distorted pixel coordinates ``(u, v)``."""
        T_wc = self.T_world_cam(camera, q_rad=q_rad)
        if T_wc is None:
            return None
        T_cw = np.linalg.inv(T_wc)
        pw = np.asarray(point_world, dtype=float).reshape(3)
        p_cam = T_cw[:3, :3] @ pw + T_cw[:3, 3]
        if p_cam[2] <= 1e-4:
            return None
        K = self._K(camera)
        dist = self._dist(camera)
        img_pts, _ = cv2.projectPoints(
            p_cam.reshape(1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            K,
            dist,
        )
        return (float(img_pts[0, 0, 0]), float(img_pts[0, 0, 1]))

    def missing_required_fields(self) -> list[str]:
        missing: list[str] = []
        if "K_top" not in self.data:
            missing.append("K_top (or scene.camera_matrix)")
        if "T_world_cam_top" not in self.data and "H_table" not in self.data:
            missing.append("T_world_cam_top (or H_table)")
        if "z_table" not in self.data:
            missing.append("z_table")
        if "K_wrist" not in self.data:
            missing.append("K_wrist (or wrist.camera_matrix)")
        if "T_ee_cam_wrist" not in self.data:
            missing.append("T_ee_cam_wrist")
        if "joint_zero_ticks" not in self.data:
            missing.append("joint_zero_ticks")
        if "joint_signs" not in self.data:
            missing.append("joint_signs")
        return missing

    def status(self) -> dict[str, Any]:
        missing = self.missing_required_fields()
        calibrated_flag = bool(self.data["calibrated"]) if "calibrated" in self.data else (len(missing) == 0)
        return {
            "calib_file": str(self.path),
            "calib_loaded": self.path.exists(),
            "calibrated": calibrated_flag,
            "arm_type": self.data.get("arm_type", "elrobot-7dof-nominal"),
            "has_K_top": "K_top" in self.data,
            "has_dist_top": "dist_top" in self.data,
            "has_T_world_cam_top": "T_world_cam_top" in self.data,
            "has_H_table": "H_table" in self.data,
            "has_K_wrist": "K_wrist" in self.data,
            "has_dist_wrist": "dist_wrist" in self.data,
            "has_T_ee_cam_wrist": "T_ee_cam_wrist" in self.data,
            "has_urdf_fk": self.urdf_path().exists(),
            "has_depth_hook": self._custom_depth_fn is not None,
            "z_table": self.z_table(),
            "hardware": self.hardware_config(),
            "missing_required": missing,
        }
