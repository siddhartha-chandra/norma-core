"""Station communication layer for NormaCore ST3215 arms and UVC cameras.

Reuses `norma-core`'s built-in `station_py` + `target/gen_python/protobuf`, incorporating
the critical June 2026 hardware lessons:
  1. Live motor pos/current come from `st3215/rx` (works even torque-off); ranges come from `st3215/inference`.
  2. Cameras publish to `usbvideo/<md5(camera_unique_id)>`, discovered remotely via `inference-states`
     or locally via `station_data/`.
  3. Safe-start torque (GoalPosition=PresentPosition before TorqueEnable=1), GoalSpeed/Acc caps,
     server-side calibrated range clamp + MAX_STEP creep limit, and instant E-STOP `abort()`.
  4. Station restarts: station_py reconnects its socket, but NormFS follow-subscriptions die
     with the old connection, so a supervisor task rebuilds them (and blocks motion commands
     while the link is down) instead of steering on stale motor positions.
"""
from __future__ import annotations

import asyncio
import glob
import io
import json
import logging
import math
import os
import random
import re
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image, ImageChops, ImageDraw, ImageStat

from .hive import HIVE_SHAPE, TILE_RADIUS_M, TILE_THICKNESS_M

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_STATION_SHARED = REPO_ROOT / "software" / "station" / "shared"
if str(_STATION_SHARED) not in sys.path:
    sys.path.insert(0, str(_STATION_SHARED))

log = logging.getLogger("desk_core.station")

# ST3215 register addresses (verified in software/station/examples/st3215-remote-teleop-py/state.py)
RAM_TORQUE_ENABLE = 0x28
RAM_ACC = 0x29
RAM_GOAL_POSITION = 0x2A
RAM_GOAL_SPEED = 0x2E
RAM_PRESENT_POSITION = 0x38
RAM_PRESENT_VOLTAGE = 0x3E
RAM_STATUS = 0x40
RAM_PRESENT_CURRENT = 0x45

MAX_ANGLE_STEP = 4095
RANGES_REFRESH_S = 5.0
SIGN_BIT_MASK = 0x8000
FULL_STATE_LEN = 0x47
GRIPPER_SLACK_STEPS = 120
GRIPPER_CONTACT_CURRENT_MA = 20
RAW_RX_MAX_AGE_S = 0.5
PREFLIGHT_DEADBAND_TICKS = 1
DEFAULT_MAX_STEP = int(os.environ.get("ROBOT_MAX_STEP_TICKS", "400"))
DEFAULT_SPEED = int(os.environ.get("ROBOT_GOAL_SPEED", "600"))
DEFAULT_ACCEL = int(os.environ.get("ROBOT_GOAL_ACCEL", "20"))

ELROBOT_MOTOR_RANGES: dict[int, tuple[int, int]] = {
    1: (1088, 3008),
    2: (1001, 3095),
    3: (912, 3184),
    4: (899, 3197),
    5: (154, 3942),
    6: (1026, 3070),
    7: (125, 3971),
    8: (1232, 2864),
}

## added by Abhi
def _calibrated_elrobot_bus() -> str:
    """Return the serial tied to the measured ElRobot limits."""
    from .calib_store import resolve_default_calib_path

    try:
        rig = json.loads(resolve_default_calib_path().read_text())
        if rig.get("calibrated") and rig.get("arm_type") == "elrobot" and rig.get("num_arm_joints") == 7:
            return rig.get("hardware", {}).get("bus_serial", "").strip()
    except (OSError, ValueError, AttributeError):
        pass
    return ""


# Auto-reconnect (issue #3): a live NormFS subscription should see motor
# traffic many times per second; silence this long means the subscription died
# (e.g. Station restart) even if no explicit error surfaced.
RESUBSCRIBE_STALE_S = float(os.environ.get("ROBOT_STALE_RESUB_S", "8"))
# Motion commands are refused when the newest telemetry is older than this,
# so clamping/creep math never runs on a snapshot from before a restart.
CMD_STALE_LIMIT_S = float(os.environ.get("ROBOT_MAX_CMD_STALE_S", "10"))


def _u16(buf: bytes, addr: int) -> int:
    return struct.unpack_from("<H", buf, addr)[0] if len(buf) >= addr + 2 else 0


def _norm_pos(raw: int) -> int:
    if raw & SIGN_BIT_MASK:
        return (MAX_ANGLE_STEP + 1 - (raw & MAX_ANGLE_STEP)) & MAX_ANGLE_STEP
    return raw & MAX_ANGLE_STEP


def _cleanest_frame(frames: list[bytes]) -> bytes:
    """Burst consensus denoising against occasional corrupt USB/usbip JPEGs."""
    pairs = []
    for jb in frames:
        try:
            pairs.append((jb, Image.open(io.BytesIO(jb)).convert("RGB")))
        except Exception:
            pass
    if len(pairs) < 3:
        return frames[-1]
    w, h = pairs[0][1].size
    imgs = [im if im.size == (w, h) else im.resize((w, h)) for _, im in pairs]
    best_i, best_score = 0, None
    for i in range(len(imgs)):
        score = sum(
            sum(ImageStat.Stat(ImageChops.difference(imgs[i], imgs[j])).sum)
            for j in range(len(imgs))
            if i != j
        )
        if best_score is None or score < best_score:
            best_score, best_i = score, i
    return pairs[best_i][0]


@dataclass
class MotorState:
    id: int
    position: int
    current_ma: int
    range_min: int
    range_max: int
    error_status: int = 0

    def to_norm(self) -> float:
        span = self.range_max - self.range_min
        if span <= 0:
            return 0.5
        return max(0.0, min(1.0, (self.position - self.range_min) / span))


@dataclass
class RobotState:
    bus_serial: str
    motors: list[MotorState]
    torque_enabled: bool = True
    link_age_ms: int | None = None  # live bridge: ms since last st3215 telemetry entry; None = mock

    def ranges(self) -> dict[int, tuple[int, int]]:
        return {m.id: (m.range_min, m.range_max) for m in self.motors}

    def positions(self) -> dict[int, int]:
        return {m.id: m.position for m in self.motors}


class EStopLatched(RuntimeError):
    """A motion command was issued while E-STOP is latched; call `rearm()` (via `observe()`) first."""


class PreflightWriteUncertain(RuntimeError):
    """Station did not confirm whether a preflighted target was sent."""


@runtime_checkable
class StationBridge(Protocol):
    async def connect(self) -> None: ...
    async def get_state(self) -> RobotState: ...
    async def get_frame(self, camera: str = "top", denoise: bool = False) -> bytes: ...
    async def get_reflex_frame(self, camera: str) -> bytes: ...
    async def list_cameras(self) -> list[str]: ...
    async def send_joint_targets(self, targets: dict[int, int], max_step: int = DEFAULT_MAX_STEP) -> dict[int, int]: ...
    async def set_torque(self, enable: bool, motor_ids: list[int] | None = None) -> bool: ...
    async def wait_settle(self, targets: dict[int, int], tol_ticks: int = 25, timeout_s: float = 3.0) -> bool: ...
    async def grasp_with_verify(self, close_ticks: int | None = None, open_ticks: int | None = None) -> dict: ...
    async def abort(self, source: str = "manual") -> dict: ...
    @property
    def abort_source(self) -> str | None: ...
    def rearm(self) -> None: ...


class MockStationBridge:
    """Zero-hardware tabletop simulation for parallel team development and offline tests."""

    def __init__(self, num_joints: int | None = None):
        if num_joints is None:
            num_joints = int(os.environ.get("ROBOT_NUM_MOTORS", "8"))
        self.num_joints = num_joints
        self._torque_on = True
        self._aborted = False
        self._abort_source: str | None = None
        self._frame_overrides: dict[str, bytes] = {}
        self._motors: dict[int, MotorState] = {
            mid: MotorState(
                mid,
                2048,
                35,
                *ELROBOT_MOTOR_RANGES.get(mid, (0, MAX_ANGLE_STEP)),
                0,
            )
            for mid in range(1, num_joints + 1)
        }
        self._ee_pos: list[float] = [0.28, 0.0, 0.25]
        self._held_object: str | None = None
        self._held_riders: list[str] = []
        self._scene_objects: dict[str, dict[str, Any]] = {}
        seed = os.environ.get("DESK_MOCK_HIVE_SEED")
        self._hive_seed = int(seed) if seed else None
        self._heading_out_scene = os.environ.get("DESK_MOCK_HEADING_OUT") == "1"
        self.reset_mock_scene()

    def seed_mock_hive(self, seed: int) -> None:
        self._hive_seed = seed
        spots = [(x, y) for x in (0.17, 0.225, 0.28, 0.335, 0.39) for y in (-0.05, 0.0, 0.05)]
        rng = random.Random(seed)
        rng.shuffle(spots)
        self._scene_objects.clear()
        for color in ("cream", "black"):
            for insect in ("ant", "bee", "beetle"):
                for number in (1, 2):
                    x, y = spots.pop()
                    self.add_mock_object(
                        f"{color}-{insect}-{number}", x, y, color=color, shape=HIVE_SHAPE, insect=insect,
                    )

    def reset_mock_scene(self) -> dict[str, Any]:
        self._ee_pos = [0.28, 0.0, 0.25]
        self._held_object = None
        self._held_riders = []
        if self._hive_seed is not None:
            self.seed_mock_hive(self._hive_seed)
            return self.get_mock_scene_state()
        if self._heading_out_scene:
            self._scene_objects.clear()
            self.add_mock_object("keys", -0.09, 0.18, color="blue", radius_m=0.018)
            self.add_mock_object("mug", 0.0, 0.18, color="brown", shape="cylinder", radius_m=0.035)
            self.add_mock_object("departure_tray", 0.09, 0.18, color="green", shape="tray", radius_m=0.035)
            return self.get_mock_scene_state()
        self._scene_objects = {
            "keys": {"x": 0.27, "y": -0.06, "z": 0.0, "color": "blue", "shape": "cube", "radius_m": 0.025},
            "mug": {"x": 0.30, "y": 0.02, "z": 0.0, "color": "brown", "shape": "cylinder", "radius_m": 0.032},
            "red_cube": {"x": 0.18, "y": 0.10, "z": 0.0, "color": "red", "shape": "cube", "radius_m": 0.024},
            "green_block": {"x": 0.36, "y": -0.12, "z": 0.0, "color": "green", "shape": "cube", "radius_m": 0.024},
        }
        return self.get_mock_scene_state()

    def add_mock_object(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.0,
        color: str = "blue",
        shape: str = "cube",
        radius_m: float = 0.025,
        insect: str | None = None,
    ) -> dict[str, Any]:
        shape = shape.strip().lower()
        self._scene_objects[name] = {
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "z": round(float(z), 4),
            "color": color.strip().lower(),
            "shape": shape,
            "radius_m": TILE_RADIUS_M if shape == HIVE_SHAPE else float(radius_m),
        }
        if shape == HIVE_SHAPE:
            self._scene_objects[name]["insect"] = insect
        return self.get_mock_scene_state()

    @staticmethod
    def _obj_height(name: str, obj: dict[str, Any]) -> float:
        heights = {
            "cylinder": 0.09, "mug": 0.09, "pen": 0.012, "wallet": 0.025,
            "airpods": 0.025, "wired_earphones": 0.015, "mobile_phone": 0.010,
            "sphere": 0.048, "cube": 0.045, "tray": 0.018, HIVE_SHAPE: TILE_THICKNESS_M,
        }
        return heights.get("mug" if name == "mug" else obj.get("shape", "cube"), 0.04)

    def set_mock_ee_pose(self, x: float, y: float, z: float) -> None:
        self._ee_pos = [round(float(x), 4), round(float(y), 4), round(float(z), 4)]
        if self._held_object and self._held_object in self._scene_objects:
            obj = self._scene_objects[self._held_object]
            new_z = max(0.0, round(self._ee_pos[2], 4))
            dx = self._ee_pos[0] - obj["x"]
            dy = self._ee_pos[1] - obj["y"]
            dz = new_z - obj.get("z", 0.0)
            obj["x"] = self._ee_pos[0]
            obj["y"] = self._ee_pos[1]
            obj["z"] = new_z
            for r_name in self._held_riders:
                if r_name in self._scene_objects:
                    r_obj = self._scene_objects[r_name]
                    r_obj["x"] = round(r_obj["x"] + dx, 4)
                    r_obj["y"] = round(r_obj["y"] + dy, 4)
                    r_obj["z"] = max(0.0, round(r_obj.get("z", 0.0) + dz, 4))

    def push_mock_pens(self, start: tuple[float, float], end: tuple[float, float], jaw_z_m: float) -> None:
        """Approximate a completed sweep in the synthetic scene, preserving lateral offsets."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        for name, obj in self._scene_objects.items():
            if obj["shape"] != "pen" or not obj["z"] <= jaw_z_m < obj["z"] + self._obj_height(name, obj):
                continue
            x, y = obj["x"] - start[0], obj["y"] - start[1]
            along, across = x * ux + y * uy, y * ux - x * uy
            radius = obj["radius_m"]
            if 0 <= along < length and abs(across) <= radius:
                obj["x"] = round(end[0] - across * uy, 4)
                obj["y"] = round(end[1] + across * ux, 4)

    def release_mock_object(self) -> str | None:
        released = self._held_object
        riders = list(self._held_riders)
        self._held_object = None
        self._held_riders = []
        if released and released in self._scene_objects:
            obj = self._scene_objects.pop(released)
            rider_objs = {r: self._scene_objects.pop(r) for r in riders if r in self._scene_objects}
            support_z = 0.0
            for other_name, other in self._scene_objects.items():
                dx = obj["x"] - other["x"]
                dy = obj["y"] - other["y"]
                if (dx * dx + dy * dy) ** 0.5 < (obj.get("radius_m", 0.025) + other.get("radius_m", 0.025)):
                    oh = self._obj_height(other_name, other)
                    support_z = max(support_z, round(other.get("z", 0.0) + oh, 4))
            obj["z"] = max(round(self._ee_pos[2], 4), support_z)
            # Re-insert last so newly placed object renders on top of co-planar items
            self._scene_objects[released] = obj
            base_top = round(obj["z"] + self._obj_height(released, obj), 4)
            for r_name, r_obj in rider_objs.items():
                r_obj["z"] = max(base_top, round(r_obj.get("z", base_top), 4))
                self._scene_objects[r_name] = r_obj
                base_top = round(r_obj["z"] + self._obj_height(r_name, r_obj), 4)
        return released

    def get_mock_scene_state(self) -> dict[str, Any]:
        return {
            "ee_pos": {"x": self._ee_pos[0], "y": self._ee_pos[1], "z": self._ee_pos[2]},
            "held_object": self._held_object,
            "objects": {k: dict(v) for k, v in self._scene_objects.items()},
        }

    async def connect(self) -> None:
        log.info("MockStationBridge connected (%d-DOF offline desk mode)", self.num_joints)

    async def get_state(self) -> RobotState:
        return RobotState("MOCK-BUS-0001", [self._motors[i] for i in sorted(self._motors)], self._torque_on)

    async def list_cameras(self) -> list[str]:
        return ["mock-top", "mock-wrist"]

    def set_frame_override(self, camera: str, jpeg_bytes: bytes | None) -> None:
        if jpeg_bytes is None:
            self._frame_overrides.pop(camera, None)
        else:
            self._frame_overrides[camera] = jpeg_bytes

    async def get_reflex_frame(self, camera: str) -> bytes:
        return await self.get_frame(camera, denoise=False)

    def _mock_frame_path(self, camera: str) -> Path | None:
        hive_active = any(obj["shape"] == HIVE_SHAPE for obj in self._scene_objects.values())
        if not (hive_active or self._heading_out_scene) and os.environ.get("DESK_MOCK_SYNTHETIC") != "1":
            env_key = "DESK_MOCK_TOP_IMAGE" if camera == "top" else "DESK_MOCK_WRIST_IMAGE"
            candidates = []
            if os.environ.get(env_key, "").strip():
                candidates.append(Path(os.environ[env_key].strip()))
            candidates.append(REPO_ROOT / "mock_frames" / f"{camera}.jpg")
            if os.environ.get("DESK_MOCK_REAL_SNAPS") == "1":
                candidates.append(REPO_ROOT / ("cam_1.jpg" if camera == "top" else "cam_0.jpg"))
            for cand in candidates:
                if cand.exists() and cand.is_file():
                    return cand
        return None

    @staticmethod
    def _project_mock_point(point, cx, cy, cz, focal):
        x, y, z = point
        dz = max(0.06, cz - z)
        return 320.0 - (y - cy) * focal / dz, 240.0 - (x - cx) * focal / dz, dz

    def synthetic_top_projector(self):
        if "top" not in self._frame_overrides and self._mock_frame_path("top") is None:
            return self._project_mock_top
        return None

    def _mock_top_camera(self):
        return (0.0, 0.21, 0.65) if self._heading_out_scene else (0.28, 0.0, 0.65)

    def _project_mock_top(self, point):
        if point[2] >= 0.65:
            return None
        return self._project_mock_point(point, *self._mock_top_camera(), 680.0)[:2]

    async def get_frame(self, camera: str = "top", denoise: bool = False) -> bytes:
        if camera in self._frame_overrides:
            return self._frame_overrides[camera]
        if path := self._mock_frame_path(camera):
            return path.read_bytes()

        # 2. Clean 640x480 synthetic 3D tabletop scene projected into top & gripper ('wrist') views
        color_rgb = {
            "blue": (40, 115, 220),
            "red": (215, 55, 50),
            "green": (45, 175, 90),
            "yellow": (230, 185, 45),
            "purple": (150, 75, 205),
            "pink": (225, 105, 165),
            "brown": (125, 78, 48),
            "orange": (230, 125, 40),
            "cream": (244, 224, 175),
            "black": (38, 38, 42),
        }
        img = Image.new("RGB", (640, 480), (234, 232, 226) if camera == "top" else (34, 38, 46))
        d = ImageDraw.Draw(img)

        if camera == "top":
            cx_cam, cy_cam, cz_cam = self._mock_top_camera()
            f_px = 680.0
            d.rectangle([28, 28, 612, 452], fill=(242, 240, 235), outline=(190, 188, 180), width=2)
        else:
            cx_cam, cy_cam, cz_cam = self._ee_pos[0], self._ee_pos[1], max(0.12, self._ee_pos[2])
            f_px = 520.0
            # Clean desk background inside gripper field of view
            d.rectangle([0, 0, 640, 480], fill=(224, 222, 216))

        def _proj(wx: float, wy: float, wz: float) -> tuple[float, float, float]:
            return self._project_mock_point((wx, wy, wz), cx_cam, cy_cam, cz_cam, f_px)

        hive_layers: dict[tuple[float, float], int] = {}
        for name, obj in sorted(self._scene_objects.items(), key=lambda kv: kv[1].get("z", 0.0)):
            u, v, dz = _proj(obj["x"], obj["y"], obj["z"])
            r_px = max(8, int(round((obj.get("radius_m", 0.025) / dz) * f_px)))
            if u < -r_px * 2 or u > 640 + r_px * 2 or v < -r_px * 2 or v > 480 + r_px * 2:
                continue
            rgb = color_rgb.get(obj.get("color", "blue"), (60, 130, 210))
            shape = obj.get("shape", "cube")
            # Subtle drop shadow
            d.ellipse([u - r_px + 4, v - r_px + 5, u + r_px + 4, v + r_px + 5], fill=(185, 183, 177))
            if shape == HIVE_SHAPE:
                d.regular_polygon((u, v, r_px), 6, fill=rgb, outline=(70, 65, 58))
                key = (obj["x"], obj["y"])
                layers = hive_layers[key] = hive_layers.get(key, 0) + 1
                ink = (255, 255, 255) if obj["color"] == "black" else (35, 35, 35)
                d.text((u, v - 5 if layers > 1 else v), obj.get("insect") or "?", anchor="mm", fill=ink)
                if layers > 1:
                    d.text((u, v + 6), f"x{layers}", anchor="mm", fill=ink)
                continue
            elif shape == "cylinder" or name == "mug":
                d.ellipse([u - r_px, v - r_px, u + r_px, v + r_px], fill=rgb, outline=(40, 40, 40), width=2)
                inner = max(3, int(r_px * 0.6))
                d.ellipse([u - inner, v - inner, u + inner, v + inner], fill=(70, 42, 26))
            elif shape == "sphere":
                d.ellipse([u - r_px, v - r_px, u + r_px, v + r_px], fill=rgb, outline=(40, 40, 40), width=2)
            elif shape == "pen":
                pw = max(14, int(r_px * 1.8))
                ph = max(4, int(r_px * 0.32))
                d.rounded_rectangle([u - pw, v - ph, u + pw, v + ph], radius=3, fill=(35, 38, 44), outline=(140, 145, 155), width=1)
                d.rectangle([u + pw - max(6, pw // 3), v - ph, u + pw, v + ph], fill=rgb)
            elif shape == "wallet":
                ww = max(14, int(r_px * 1.35))
                wh = max(10, int(r_px * 0.95))
                d.rounded_rectangle([u - ww, v - wh, u + ww, v + wh], radius=5, fill=(105, 62, 38), outline=(55, 32, 20), width=2)
                d.line([(u - ww + 4, v), (u + ww - 4, v)], fill=(140, 90, 55), width=1)
            elif shape == "airpods":
                aw = max(10, int(r_px * 0.95))
                ah = max(8, int(r_px * 0.8))
                d.rounded_rectangle([u - aw, v - ah, u + aw, v + ah], radius=7, fill=(248, 249, 250), outline=(185, 190, 198), width=2)
                d.ellipse([u - 2, v - 2, u + 3, v + 3], fill=(45, 205, 95))
            elif shape == "wired_earphones":
                d.arc([u - r_px, v - r_px, u + r_px, v + r_px], start=20, end=310, fill=(240, 242, 246), width=3)
                d.ellipse([u - r_px - 3, v - 4, u - r_px + 5, v + 4], fill=(248, 249, 250), outline=(150, 155, 165), width=1)
                d.ellipse([u + r_px - 5, v - 4, u + r_px + 3, v + 4], fill=(248, 249, 250), outline=(150, 155, 165), width=1)
            elif shape == "mobile_phone":
                mw = max(11, int(r_px * 0.85))
                mh = max(18, int(r_px * 1.55))
                d.rounded_rectangle([u - mw, v - mh, u + mw, v + mh], radius=6, fill=(26, 30, 38), outline=(110, 120, 135), width=2)
                d.rounded_rectangle([u - mw + 3, v - mh + 4, u + mw - 3, v + mh - 4], radius=3, fill=(45, 65, 95))
            else:
                d.rounded_rectangle([u - r_px, v - r_px, u + r_px, v + r_px], radius=4, fill=rgb, outline=(40, 40, 40), width=2)
            d.text((u - r_px, max(4, v - r_px - 14)), name, fill=(35, 35, 35))

        if camera == "top":
            eu, ev, _ = _proj(self._ee_pos[0], self._ee_pos[1], self._ee_pos[2])
            d.ellipse([eu - 10, ev - 10, eu + 10, ev + 10], outline=(230, 110, 30), width=3)
            d.line([(eu - 14, ev), (eu + 14, ev)], fill=(230, 110, 30), width=2)
            d.line([(eu, ev - 14), (eu, ev + 14)], fill=(230, 110, 30), width=2)
        else:
            # Gripper jaws overlay at bottom of eye-in-hand view + crosshair
            d.polygon([(40, 480), (150, 330), (190, 348), (115, 480)], fill=(32, 34, 40))
            d.polygon([(600, 480), (490, 330), (450, 348), (525, 480)], fill=(32, 34, 40))
            d.line([(320, 0), (320, 480)], fill=(90, 190, 110), width=1)
            d.line([(0, 240), (640, 240)], fill=(90, 190, 110), width=1)

        cam_lbl = "TOP" if camera == "top" else "GRIPPER (wrist)"
        d.text((12, 10), f"MOCK {cam_lbl} | EE=({self._ee_pos[0]:.2f},{self._ee_pos[1]:.2f},{self._ee_pos[2]:.2f})", fill=(35, 35, 35))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    async def send_joint_targets(self, targets: dict[int, int], max_step: int = DEFAULT_MAX_STEP) -> dict[int, int]:
        if self._aborted:
            raise EStopLatched("E-STOP latched: call observe() before moving")
        applied: dict[int, int] = {}
        for mid, raw_tgt in targets.items():
            if mid not in self._motors:
                continue
            m = self._motors[mid]
            clamped = max(m.range_min, min(m.range_max, int(raw_tgt)))
            delta = max(-max_step, min(max_step, clamped - m.position))
            m.position += delta
            applied[mid] = m.position
        return applied

    async def set_torque(self, enable: bool, motor_ids: list[int] | None = None) -> bool:
        self._torque_on = enable
        return True

    async def wait_settle(self, targets: dict[int, int], tol_ticks: int = 25, timeout_s: float = 3.0) -> bool:
        return True

    def get_mock_riders(self, name: str) -> list[str]:
        held = self._scene_objects[name]
        min_z = held["z"] + self._obj_height(name, held) * 0.5
        return [
            other_name
            for other_name, other in sorted(self._scene_objects.items(), key=lambda kv: kv[1].get("z", 0.0))
            if other_name != name and other["z"] >= min_z
            and math.hypot(other["x"] - held["x"], other["y"] - held["y"])
            < held["radius_m"] + other["radius_m"]
        ]

    async def grasp_with_verify(self, close_ticks: int | None = None, open_ticks: int | None = None) -> dict:
        gid = max(self._motors)
        m = self._motors[gid]
        target = close_ticks if close_ticks is not None else (m.range_min + GRIPPER_SLACK_STEPS + 80)
        m.position = max(m.range_min, min(m.range_max, target))
        gap = abs(m.position - m.range_min)
        best_name = None
        best_dist_3d = 0.12
        for name, obj in sorted(self._scene_objects.items(), key=lambda kv: kv[1].get("z", 0.0)):
            dx = obj["x"] - self._ee_pos[0]
            dy = obj["y"] - self._ee_pos[1]
            dz = obj["z"] - self._ee_pos[2]
            d_xy = (dx * dx + dy * dy) ** 0.5
            d_3d = (dx * dx + dy * dy + dz * dz) ** 0.5
            if d_xy < 0.08 and abs(dz) < 0.09 and d_3d < best_dist_3d:
                best_dist_3d = d_3d
                best_name = name
        self._held_object = best_name
        self._held_riders = self.get_mock_riders(best_name) if best_name is not None else []
        if best_name is not None:
            m.position = max(m.position, min(m.range_max, m.range_min + GRIPPER_SLACK_STEPS + 80))
            gap = abs(m.position - m.range_min)
        return {
            "ok": True,
            "holding": bool(best_name is not None or gap > GRIPPER_SLACK_STEPS),
            "held_object": best_name,
            "gripper_id": gid,
            "position": m.position,
            "current_ma": 180,
        }

    async def abort(self, source: str = "manual") -> dict:
        if not self._aborted or source == "manual":
            self._abort_source = source
        self._aborted = True
        return {"aborted": True, "source": self._abort_source, "positions": {k: v.position for k, v in self._motors.items()}}

    @property
    def abort_source(self) -> str | None:
        return self._abort_source

    def rearm(self) -> None:
        self._aborted = False
        self._abort_source = None

    def link_status(self) -> dict:
        return {"bridge": "mock", "station": None, "socket_up": True, "resyncing": False, "rx_age_ms": None}


class LiveStationBridge:
    """Async client for live NormaCore Station over NormFS TCP (:8888).

    Survives Station restarts: station_py's Client reconnects the socket on
    its own, but follow-subscriptions are per-connection, so a supervisor
    task detects dead streams (error queues or stalled telemetry) and
    re-issues them. Motion commands are refused while the link is down or
    stale instead of steering on pre-restart positions.
    """

    ET_FRAMES = 0
    RECONNECT_WAIT_S = 30.0
    SUPERVISE_TICK_S = 1.0

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8888,
        target_bus: str | None = None,
        cam_top: str | None = None,
        cam_wrist: str | None = None,
        *,
        follow_motor_rx: bool = False,
    ):
        self.host = host
        self.port = port
        self.target_bus = (target_bus or "").strip() or None
        self._elrobot_range_bus = _calibrated_elrobot_bus()
        self.client = None
        self._st3215 = None
        self._usbvideo = None
        self._inference_pb = None
        self._commands = None
        self._drivers = None
        self._send_commands = None
        self._motor_state: dict[str, dict[int, MotorState]] = {}
        self._bus_voltage: dict[str, float] = {}
        self._torque_started: set[int] = set()
        self._aborted = False
        self._abort_source: str | None = None
        self._follow_motor_rx = follow_motor_rx
        self._station_ranges: dict[str, dict[int, tuple[int, int]]] = {}
        self._ranges_polled_mono: float | None = None
        self.rx_received_at: dict[tuple[str, int], float] = {}
        self.frame_received_at: dict[str, float] = {}
        self._video_entry_ids: dict[str, bytes] = {}
        self._frames: dict[str, bytes] = {}
        self._frame_buf: dict[str, deque[bytes]] = {}
        self._frame_order: list[str] = []
        self._followed_video_qids: set[str] = set()
        self._key_to_qid: dict[str, str] = {}
        self._probe_tasks: dict[str, asyncio.Task] = {}
        self._tasks: list[asyncio.Task] = []
        self._supervisor: asyncio.Task | None = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._follow_errors: dict[str, asyncio.Queue] = {}
        self._last_rx_mono: float | None = None
        self._resyncing = False
        self._cam_map = {
            "top": (cam_top if cam_top is not None else os.environ.get("CAMERA_TOP", "")).strip(),
            "wrist": (cam_wrist if cam_wrist is not None else os.environ.get("CAMERA_WRIST", "")).strip(),
        }

    def apply_hardware_config(self, bus_serial: str | None = None, cam_top: str | None = None, cam_wrist: str | None = None) -> None:
        self._elrobot_range_bus = _calibrated_elrobot_bus()
        if bus_serial is not None:
            self.target_bus = bus_serial.strip() or None
        if cam_top is not None:
            self._cam_map["top"] = cam_top.strip()
        if cam_wrist is not None:
            self._cam_map["wrist"] = cam_wrist.strip()

    async def connect(self) -> None:
        from station_py import new_station_client, send_commands  # type: ignore
        from target.gen_python.protobuf.drivers.st3215 import st3215 as st3215_pb2  # type: ignore
        from target.gen_python.protobuf.drivers.usbvideo import usbvideo as usbvideo_pb2  # type: ignore
        from target.gen_python.protobuf.station import (  # type: ignore
            commands as commands_pb2,
            drivers as drivers_pb2,
            inference as inference_pb2,
        )

        self._st3215 = st3215_pb2
        self._usbvideo = usbvideo_pb2
        self._inference_pb = inference_pb2
        self._commands = commands_pb2
        self._drivers = drivers_pb2
        self._send_commands = send_commands

        self.client = await new_station_client(f"{self.host}:{self.port}", log)
        log.info("Connected to Station at %s:%s", self.host, self.port)

        self._make_followers()
        self._start_workers()
        await self._warmup()
        if self._supervisor is None or self._supervisor.done():
            self._supervisor = asyncio.create_task(self._supervise())

    # -- subscription lifecycle (rebuilt on every Station restart) -------------

    def _make_followers(self) -> None:
        """Fresh follow queues + follow() calls for the base streams.

        station_py's Client re-establishes the TCP socket itself after a
        Station restart, but follow streams do NOT survive the old
        connection: _follow_async pushes its error onto the returned
        *error queue* and returns, leaving the data queue silent forever.
        We therefore keep the error queues and let _supervise() watch them.
        """
        if self.client is None:
            raise RuntimeError("_make_followers() called before connect()")
        # When read_from_tail is available (live station_py), poll st3215/inference
        # on-demand instead of following the 400-800 Hz st3215/rx & st3215/inference
        # firehoses continuously over Tailscale (which drifts behind by seconds).
        if self._follow_motor_rx:
            # Paired camera captures need receipt times for each raw encoder sample.
            queues = ("st3215/rx", "inference-states")
        else:
            queues = ("inference-states",) if hasattr(self.client, "read_from_tail") else self._BASE_QUEUES
        self._queues = {qid: asyncio.Queue() for qid in queues}
        self._video_q = asyncio.Queue()
        self._follow_errors = {}
        for qid, q in self._queues.items():
            self._follow(qid, q)
        if self.host in ("127.0.0.1", "localhost", "::1"):
            for qid in self._discover_local_video_queues():
                self._follow_video_queue(qid)

    _BASE_QUEUES = ("st3215/inference", "st3215/rx", "inference-states")

    def _follow(self, qid: str, q: asyncio.Queue) -> None:
        err_q = self.client.follow(qid, q)
        if err_q is not None:
            self._follow_errors[qid] = err_q

    def _start_workers(self) -> None:
        tasks = []
        if "st3215/inference" in self._queues:
            tasks.append(asyncio.create_task(self._consume_inference(self._queues["st3215/inference"])))
        if "st3215/rx" in self._queues:
            tasks.append(asyncio.create_task(self._consume_rx(self._queues["st3215/rx"])))
        if "inference-states" in self._queues:
            tasks.append(asyncio.create_task(self._consume_inference_states(self._queues["inference-states"])))
        if not hasattr(self.client, "read_from_tail"):
            tasks.append(asyncio.create_task(self._consume_video(self._video_q)))
        self._tasks = tasks

    def _follow_video_queue(self, qid: str) -> None:
        if qid not in self._followed_video_qids and self.client is not None:
            self._followed_video_qids.add(qid)
            # Fetch camera frames on-demand via read_from_tail instead of continuous 30fps
            # follow(), so ~100 Mbps of MJPEG never congests st3215/rx or commands over Tailscale.
            if hasattr(self.client, "read_from_tail"):
                self._probe_tasks[qid] = asyncio.create_task(self._probe_video_queue(qid))
            else:
                self._follow(qid, self._video_q)
            log.info("Registered camera queue: %s", qid)

    def _discover_local_video_queues(self, max_age_s: float = 180.0) -> list[str]:
        roots = [
            os.environ.get("STATION_DATA_DIR", "").strip(),
            str(REPO_ROOT / "station_data"),
            str(Path.cwd() / "station_data"),
        ]
        found: list[str] = []
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            for d in glob.glob(os.path.join(root, "*", "usbvideo", "*")):
                name = os.path.basename(d)
                if len(name) == 32 and all(c in "0123456789abcdef" for c in name) and os.path.isdir(d):
                    found.append(d)
            if found:
                break
        if not found:
            return []
        now = time.time()
        live = [d for d in found if (now - os.path.getmtime(d)) <= max_age_s]
        return sorted({"usbvideo/" + os.path.basename(d) for d in (live or found)})

    # -- auto-reconnect supervisor ---------------------------------------------

    async def _supervise(self) -> None:
        """Rebuild follow subscriptions whenever Station goes away and comes back.

        station_py's own manage_connection() restores the socket; this task
        restores everything layered on top of it (queues, consumers, camera
        discovery). Runs forever; individual resync failures are logged and
        retried on the next tick.
        """
        while True:
            await asyncio.sleep(self.SUPERVISE_TICK_S)
            if self._resyncing:
                continue
            if (
                hasattr(self.client, "read_from_tail")
                and self._last_rx_mono is not None
                and (time.monotonic() - self._last_rx_mono) > RESUBSCRIBE_STALE_S
                and bool(self.client and self.client.connected and self.client.setup_done)
            ):
                await self._poll_inference_tail(timeout_s=2.0)
            reason = self._dead_reason()
            if reason is None:
                continue
            await self._resync(reason)

    def _dead_reason(self) -> str | None:
        for qid, eq in list(self._follow_errors.items()):
            try:
                err = eq.get_nowait()
            except asyncio.QueueEmpty:
                continue
            return f"follow {qid!r} died: {err!r}"
        for t in self._tasks:
            if t.done():
                exc = None if t.cancelled() else t.exception()
                return f"consumer task exited ({exc!r})"
        if self._last_rx_mono is not None:
            silence = time.monotonic() - self._last_rx_mono
            if silence > RESUBSCRIBE_STALE_S:
                return f"no st3215 telemetry for {silence:.1f}s"
        return None

    async def _resync(self, reason: str) -> None:
        self._resyncing = True
        log.warning("Station link lost (%s) — auto-resubscribing", reason)
        try:
            for t in [*self._tasks, *self._probe_tasks.values()]:
                t.cancel()
            await asyncio.gather(*self._tasks, *self._probe_tasks.values(), return_exceptions=True)
            self._tasks = []
            self._probe_tasks.clear()
            # Block until station_py's manage_connection has re-setup the socket.
            try:
                await self.client.wait_ready(timeout=self.RECONNECT_WAIT_S)
            except TimeoutError:
                log.error("Station socket not back after %.0fs, retrying…", self.RECONNECT_WAIT_S)
                return
            self._followed_video_qids.clear()
            self._make_followers()
            self._start_workers()
            await self._warmup(max_s=8.0)
            log.info("Station link restored (%d bus(es), %d camera(s))",
                     len(self._motor_state), len(self._frame_order))
        except Exception as exc:
            log.error("Resync attempt failed: %s", exc)
        finally:
            self._resyncing = False

    def link_status(self) -> dict:
        up = bool(self.client and self.client.connected and self.client.setup_done)
        age = None
        if self._last_rx_mono is not None:
            age = int((time.monotonic() - self._last_rx_mono) * 1000)
        return {
            "bridge": "live",
            "station": f"{self.host}:{self.port}",
            "socket_up": up,
            "resyncing": self._resyncing,
            "rx_age_ms": age,
        }

    def _require_fresh(self) -> None:
        """Refuse motion commands built on pre-restart / stalled telemetry."""
        if self._last_rx_mono is None:
            raise RuntimeError("No live telemetry yet — refusing to command motion (Station offline or resyncing).")
        age = time.monotonic() - self._last_rx_mono
        if age > CMD_STALE_LIMIT_S:
            raise RuntimeError(
                f"Telemetry stale for {age:.1f}s (> {CMD_STALE_LIMIT_S:.0f}s limit) — "
                "motion is blocked until the Station link is healthy again."
            )

    async def _consume_inference_states(self, q: asyncio.Queue) -> None:
        qdt_video = getattr(self._drivers.QueueDataType, "QDT_USB_VIDEO_FRAMES", 21)
        while True:
            entry = await q.get()
            if entry is None:
                break
            try:
                rx = self._inference_pb.InferenceRxReader(memoryview(bytes(entry.Data)))
                for e in rx.get_entries():
                    if e.get_type() == qdt_video:
                        m = re.search(r"usbvideo/([0-9a-f]{32})", e.get_queue())
                        if m:
                            self._follow_video_queue(f"usbvideo/{m.group(1)}")
            except Exception as exc:
                log.debug("inference-states parse skip: %s", exc)

    def _ingest_inference_bytes(self, raw: bytes | memoryview, ranges_only: bool = False) -> bool:
        # ranges_only keeps positions owned by st3215/rx so encoder samples stay paired with receipt times.
        reader = self._st3215.InferenceStateReader(memoryview(bytes(raw)))
        updated = False
        for bus in reader.get_buses():
            info = bus.get_bus()
            serial = info.get_serial_number() if info else ""
            if not serial:
                continue
            ranges = self._station_ranges.setdefault(serial, {})
            for m in bus.get_motors():
                if m.get_range_max() > m.get_range_min():
                    ranges[m.get_id()] = (m.get_range_min(), m.get_range_max())
            if ranges_only:
                for mid, ms in self._motor_state.get(serial, {}).items():
                    if mid in ranges:
                        ms.range_min, ms.range_max = ranges[mid]
                        updated = True
                continue
            slot = self._motor_state.setdefault(serial, {})
            volts: list[float] = []
            for m in bus.get_motors():
                sb = bytes(m.get_state())
                if len(sb) < FULL_STATE_LEN:
                    continue
                if sb[RAM_PRESENT_VOLTAGE] > 0:
                    volts.append(sb[RAM_PRESENT_VOLTAGE] / 10.0)
                mid = m.get_id()
                pos = _norm_pos(_u16(sb, RAM_PRESENT_POSITION))
                cur = _u16(sb, RAM_PRESENT_CURRENT)
                err = sb[RAM_STATUS]
                if mid in slot:
                    slot[mid].position = pos
                    slot[mid].current_ma = cur
                    slot[mid].error_status = err
                    slot[mid].range_min = m.get_range_min()
                    slot[mid].range_max = m.get_range_max()
                else:
                    slot[mid] = MotorState(
                        mid,
                        pos,
                        cur,
                        m.get_range_min(),
                        m.get_range_max(),
                        err,
                    )
                updated = True
            if volts:
                self._bus_voltage[serial] = sum(volts) / len(volts)
        return updated

    async def _poll_inference_tail(self, timeout_s: float = 2.0) -> bool:
        if self.client is None or not hasattr(self.client, "read_from_tail"):
            return False
        if self._follow_motor_rx:
            # Positions come from st3215/rx here; refresh only the calibrated ranges, and not on every call.
            if self._ranges_polled_mono is not None and time.monotonic() - self._ranges_polled_mono < RANGES_REFRESH_S:
                return False
        if not (self.client.connected and self.client.setup_done):
            return False
        if self._follow_motor_rx:
            self._ranges_polled_mono = time.monotonic()
        try:
            qr = self.client.read_from_tail("st3215/inference", offset=b"\x00", limit=1, step=1, buf_size=4)
            latest_entry = None
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                rem = max(0.05, deadline - time.monotonic())
                entry = await asyncio.wait_for(qr.data.get(), timeout=rem)
                if entry is None:
                    break
                latest_entry = entry
            if latest_entry is not None and self._follow_motor_rx:
                self._ingest_inference_bytes(latest_entry.Data, ranges_only=True)
                return False
            if latest_entry is not None and self._ingest_inference_bytes(latest_entry.Data):
                self._last_rx_mono = time.monotonic()
                return True
        except Exception as exc:
            log.debug("tail inference poll failed: %s", exc)
        return False

    async def _consume_inference(self, q: asyncio.Queue) -> None:
        while True:
            entry = await q.get()
            if entry is None:
                break
            self._last_rx_mono = time.monotonic()
            try:
                self._ingest_inference_bytes(entry.Data)
            except Exception as exc:
                log.debug("inference parse skip: %s", exc)

    async def _consume_rx(self, q: asyncio.Queue) -> None:
        while True:
            entry = await q.get()
            if entry is None:
                break
            self._last_rx_mono = time.monotonic()
            try:
                r = self._st3215.RxEnvelopeReader(memoryview(bytes(entry.Data)))
                sb = bytes(r.get_data())
                if len(sb) < FULL_STATE_LEN:
                    continue
                bus = r.get_bus()
                serial = bus.get_serial_number() if bus else ""
                mid = r.get_motor_id()
                if not serial or mid == 0:
                    continue
                slot = self._motor_state.setdefault(serial, {})
                pos = _norm_pos(_u16(sb, RAM_PRESENT_POSITION))
                self.rx_received_at[serial, mid] = time.monotonic()
                cur = _u16(sb, RAM_PRESENT_CURRENT)
                err = sb[RAM_STATUS]
                if mid in slot:
                    slot[mid].position = pos
                    slot[mid].current_ma = cur
                    slot[mid].error_status = err
                else:
                    lo, hi = self._station_ranges.get(serial, {}).get(mid, (0, MAX_ANGLE_STEP))
                    slot[mid] = MotorState(mid, pos, cur, lo, hi, err)
            except Exception as exc:
                log.debug("rx parse skip: %s", exc)

    def _extract_jpeg(self, entry) -> tuple[str, bytes] | None:
        env = self._usbvideo.RxEnvelopeReader(memoryview(bytes(entry.Data)))
        if env.get_type() != self.ET_FRAMES:
            return None
        cam = env.get_camera()
        key = cam.get_unique_id() or cam.get_serial_number() or cam.get_product() or "cam0"
        frames = env.get_frames().get_frames_data()
        if not frames:
            return None
        jpeg = next(
            (bytes(b) for b in reversed(frames) if len(bytes(b)) > 4 and bytes(b)[:2] == b"\xff\xd8" and bytes(b)[-2:] == b"\xff\xd9"),
            None,
        )
        return (key, jpeg) if jpeg is not None else None

    def _record_frame_receipt(self, key: str, jpeg: bytes, entry_id: bytes | None) -> None:
        if self._follow_motor_rx:
            # Re-reading the queue tail must not make a stalled camera look fresh.
            if entry_id is None:
                if self._frames.get(key) == jpeg:
                    return
            else:
                if self._video_entry_ids.get(key) == entry_id:
                    return
                self._video_entry_ids[key] = entry_id
        self.frame_received_at[key] = time.monotonic()

    async def _fetch_tail_frames(
        self, qid: str, count: int = 1, timeout_s: float = 3.0,
    ) -> list[tuple[str, bytes, bytes | None]]:
        if self.client is None or not hasattr(self.client, "read_from_tail"):
            return []
        offset = (count - 1).to_bytes(8, "little") if count > 1 else b"\x00"
        qr = self.client.read_from_tail(qid, offset=offset, limit=count, step=1, buf_size=max(4, count + 2))
        out: list[tuple[str, bytes, bytes | None]] = []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rem = max(0.05, deadline - time.monotonic())
            try:
                entry = await asyncio.wait_for(qr.data.get(), timeout=rem)
            except asyncio.TimeoutError:
                break
            if entry is None:
                break
            try:
                parsed = self._extract_jpeg(entry)
                if parsed is not None:
                    out.append((*parsed, getattr(getattr(entry, "ID", None), "ID", None)))
            except Exception as exc:
                log.debug("tail video parse skip: %s", exc)
        return out

    async def _probe_video_queue(self, qid: str) -> None:
        try:
            items = await self._fetch_tail_frames(qid, count=1, timeout_s=3.0)
            if not items:
                return
            key, jpeg, entry_id = items[-1]
            self._key_to_qid[key] = qid
            self._record_frame_receipt(key, jpeg, entry_id)
            self._frames[key] = jpeg
            self._frame_buf.setdefault(key, deque(maxlen=7)).append(jpeg)
            if key not in self._frame_order:
                self._frame_order.append(key)
                log.info("Camera discovered: %s (%s)", key, qid)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("camera probe failed for %s: %s", qid, exc)

    async def _consume_video(self, q: asyncio.Queue) -> None:
        while True:
            entry = await q.get()
            if entry is None:
                break
            try:
                parsed = self._extract_jpeg(entry)
                if parsed is None:
                    continue
                key, jpeg = parsed
                self._record_frame_receipt(key, jpeg, getattr(getattr(entry, "ID", None), "ID", None))
                self._frames[key] = jpeg
                self._frame_buf.setdefault(key, deque(maxlen=7)).append(jpeg)
                if key not in self._frame_order:
                    self._frame_order.append(key)
                    log.info("Camera discovered: %s", key)
            except Exception as exc:
                log.debug("video parse skip: %s", exc)

    async def _warmup(self, max_s: float = 5.0) -> None:
        want_cams = 2 if (self._cam_map.get("top") and self._cam_map.get("wrist")) else 1
        settled_since: float | None = None
        for _ in range(int(max_s / 0.1)):
            if not self._motor_state and hasattr(self.client, "read_from_tail"):
                await self._poll_inference_tail(timeout_s=1.0)
            probes_done = all(t.done() for t in self._probe_tasks.values()) if self._probe_tasks else False
            if self._motor_state and len(self._frames) >= want_cams and probes_done:
                break
            if self._motor_state and self._frames:
                settled_since = settled_since or time.monotonic()
                # Wait briefly so both top & wrist cameras finish publishing initial frames
                if time.monotonic() - settled_since >= 1.5:
                    break
            await asyncio.sleep(0.1)

    def _select_bus(self) -> str:
        if self.target_bus and self.target_bus in self._motor_state:
            return self.target_bus
        if not self._motor_state:
            raise RuntimeError("No ST3215 motor state received from Station yet.")

        def score(item: tuple[str, dict[int, MotorState]]):
            serial, ms = item
            # Prefer 12V-powered follower buses (>=10V) over 7.6V/unpowered leader buses when unpinned
            v = self._bus_voltage.get(serial, 0.0)
            powered = 1 if v >= 10.0 else 0
            return (powered, sum(1 for m in ms.values() if m.range_max > m.range_min), len(ms), v)

        return max(self._motor_state.items(), key=score)[0]

    async def get_state(self) -> RobotState:
        for _ in range(50):
            if hasattr(self.client, "read_from_tail"):
                await self._poll_inference_tail(timeout_s=1.5)
            if self._motor_state:
                serial = self._select_bus()
                motors = sorted(self._motor_state[serial].values(), key=lambda m: m.id)
                if motors:
                    age = None
                    if self._last_rx_mono is not None:
                        age = int((time.monotonic() - self._last_rx_mono) * 1000)
                    return RobotState(serial, motors, bool(self._torque_started), age)
            await asyncio.sleep(0.1)
        raise RuntimeError("Timed out waiting for ST3215 joint state from Station.")

    async def list_cameras(self) -> list[str]:
        return list(self._frame_order)

    def _resolve_cam_key(self, camera: str) -> str:
        want = self._cam_map.get(camera, "")
        if want:
            for key in self._frames:
                if want in key:
                    return key
        idx = 1 if (camera == "wrist" and len(self._frame_order) > 1) else 0
        return self._frame_order[idx]

    async def get_reflex_frame(self, camera: str) -> bytes:
        opposite = "top" if camera == "wrist" else "wrist"
        if not self._frame_order or (camera == "wrist" and len(self._frame_order) < 2):
            raise RuntimeError(f"No {camera} reflex camera")
        key = self._resolve_cam_key(camera)
        if self._cam_map.get(camera) and self._cam_map[camera] not in key:
            raise RuntimeError(f"No calibrated {camera} reflex camera")
        if len(self._frame_order) > 1 and key == self._resolve_cam_key(opposite):
            raise RuntimeError(f"Cannot resolve distinct {camera} reflex camera")
        jpeg = await self.get_frame(camera, denoise=False)
        if time.monotonic() - self.frame_received_at.get(key, -math.inf) > 0.1:
            raise RuntimeError(f"Stale {camera} reflex camera")
        return jpeg

    async def get_frame(self, camera: str = "top", denoise: bool = False) -> bytes:
        for _ in range(50):
            if self._frames:
                break
            await asyncio.sleep(0.1)
        if not self._frames:
            raise RuntimeError("No USB video frames received from Station yet.")
        key = self._resolve_cam_key(camera)
        qid = self._key_to_qid.get(key)
        if qid:
            try:
                items = await self._fetch_tail_frames(qid, count=4 if denoise else 1, timeout_s=3.0)
                jpegs = [jb for _key, jb, _entry_id in items]
                if jpegs:
                    self._record_frame_receipt(key, jpegs[-1], items[-1][2])
                    self._frames[key] = jpegs[-1]
                    buf = self._frame_buf.setdefault(key, deque(maxlen=7))
                    buf.extend(jpegs)
                    if denoise and len(jpegs) >= 3:
                        return await asyncio.to_thread(_cleanest_frame, jpegs[-4:])
                    return jpegs[-1]
            except Exception as exc:
                log.debug("on-demand camera fetch failed for %s: %s", key, exc)
        if denoise:
            buf = list(self._frame_buf.get(key, ()))
            if len(buf) >= 3:
                return await asyncio.to_thread(_cleanest_frame, buf)
        return self._frames[key]

    def _sync_write(self, bus_serial: str, address: int, values: dict[int, bytes]):
        st = self._st3215
        sync = st.ST3215SyncWriteCommand(
            address=address,
            motors=[st.ST3215SyncWriteCommand_MotorWrite(motor_id=mid, value=val) for mid, val in values.items()],
        )
        cmd = st.Command(target_bus_serial=bus_serial, sync_write=sync)
        return self._commands.DriverCommand(
            type=self._drivers.StationCommandType.STC_ST3215_COMMAND,
            body=cmd.encode(),
        )

    async def set_torque(self, enable: bool, motor_ids: list[int] | None = None) -> bool:
        if enable:
            if hasattr(self.client, "read_from_tail"):
                await self._poll_inference_tail(timeout_s=1.0)
            # Safe-start writes Goal=Present; both must come from live telemetry.
            self._require_fresh()
        st = await self.get_state()
        if self.target_bus and st.bus_serial != self.target_bus:
            raise RuntimeError(f"Expected follower bus {self.target_bus}, found {st.bus_serial}")
        mids = motor_ids or [m.id for m in st.motors]
        pack = []
        if enable:
            if self._aborted:
                raise EStopLatched("E-STOP latched: call observe() before moving")
            self._require_fresh_motor_rx(st, mids)
            present = {m.id: m.position.to_bytes(2, "little") for m in st.motors if m.id in mids}
            pack.append(self._sync_write(st.bus_serial, RAM_GOAL_POSITION, present))
            pack.append(self._sync_write(st.bus_serial, RAM_GOAL_SPEED, {m: DEFAULT_SPEED.to_bytes(2, "little") for m in mids}))
            pack.append(self._sync_write(st.bus_serial, RAM_ACC, {m: bytes([DEFAULT_ACCEL]) for m in mids}))
            self._torque_started.update(mids)
        else:
            self._torque_started.difference_update(mids)
        val = b"\x01" if enable else b"\x00"
        pack.append(self._sync_write(st.bus_serial, RAM_TORQUE_ENABLE, {m: val for m in mids}))
        await self._send_commands(self.client, pack)
        return True

    ## added by Abhi
    def _require_fresh_motor_rx(self, state: RobotState, motor_ids) -> None:
        if not self._follow_motor_rx:
            return
        now = time.monotonic()
        for mid in motor_ids:
            if not 0 <= now - self.rx_received_at.get((state.bus_serial, int(mid)), -math.inf) <= RAW_RX_MAX_AGE_S:
                raise RuntimeError(f"Motor {mid} raw telemetry is stale")

    ## added by Abhi
    def _effective_motor_range(self, bus_serial: str, motor: MotorState) -> tuple[int, int]:
        lo, hi = motor.range_min, motor.range_max
        if 0 <= lo < hi <= MAX_ANGLE_STEP and (lo, hi) != (0, MAX_ANGLE_STEP):
            return lo, hi
        if bus_serial == self._elrobot_range_bus and motor.id in ELROBOT_MOTOR_RANGES:
            return ELROBOT_MOTOR_RANGES[motor.id]
        return 0, MAX_ANGLE_STEP

    async def send_joint_targets(self, targets: dict[int, int], max_step: int = DEFAULT_MAX_STEP, *,
                                 preflight_start: RobotState | None = None) -> dict[int, int]:
        if self._aborted:
            raise EStopLatched("E-STOP latched: call observe() before moving")
        if hasattr(self.client, "read_from_tail"):
            await self._poll_inference_tail(timeout_s=1.0)
        self._require_fresh()
        st = await self.get_state()
        if self.target_bus and st.bus_serial != self.target_bus:
            raise RuntimeError(f"Expected follower bus {self.target_bus}, found {st.bus_serial}")
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        step_limit = min(max_step, DEFAULT_MAX_STEP)
        by_id = {m.id: m for m in st.motors}
        applied: dict[int, int] = {}
        self._require_fresh_motor_rx(st, targets)
        if preflight_start is not None:
            self._require_fresh_motor_rx(st, preflight_start.positions())
            actual_ranges = {m.id: self._effective_motor_range(st.bus_serial, m) for m in st.motors}
            if (st.bus_serial != preflight_start.bus_serial or st.positions() != preflight_start.positions()
                    or actual_ranges != preflight_start.ranges() or any(m.error_status for m in st.motors)):
                raise RuntimeError("Station no longer matches the preflighted starting state")
            if not targets.keys() <= self._torque_started:
                raise RuntimeError("Enable arm torque before executing a preflighted move")

        for raw_mid, raw_tgt in targets.items():
            mid = int(raw_mid)
            if mid not in by_id:
                if self._follow_motor_rx:
                    raise RuntimeError(f"Missing follower telemetry for motor {mid}")
                continue
            m = by_id[mid]
            if m.error_status:
                raise RuntimeError(f"Motor {mid} reports error status {m.error_status}")
            lo, hi = self._effective_motor_range(st.bus_serial, m)
            clamped = max(lo, min(hi, int(raw_tgt)))
            delta = max(-step_limit, min(step_limit, clamped - m.position))
            stepped = m.position + delta
            if not lo <= stepped <= hi:
                raise RuntimeError(f"Motor {mid} cannot enter its calibrated range within one safe step")
            applied[mid] = stepped

        if preflight_start is not None and (self._aborted or applied != targets):
            raise RuntimeError("Station cannot send the exact preflighted waypoint")
        if applied:
            need_start = [mid for mid in applied if mid not in self._torque_started]
            if need_start:
                await self.set_torque(True, need_start)
            if self._aborted:
                raise EStopLatched("E-STOP latched: call observe() before moving")
            cmd = self._sync_write(
                st.bus_serial,
                RAM_GOAL_POSITION,
                {mid: val.to_bytes(2, "little") for mid, val in applied.items()},
            )
            try:
                await self._send_commands(self.client, [cmd])
            except Exception as exc:
                if preflight_start is None:
                    raise
                raise PreflightWriteUncertain(f"Station waypoint write unconfirmed: {exc}") from exc
        return applied

    async def wait_settle(self, targets: dict[int, int], tol_ticks: int = 25, timeout_s: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            st = await self.get_state()
            pos = st.positions()
            if all(abs(pos.get(mid, 0) - tgt) <= tol_ticks for mid, tgt in targets.items()):
                return True
            await asyncio.sleep(0.08)
        return False

    async def wait_preflight_settle(self, targets: dict[int, int], tol_ticks: int, *,
                                    moving_from: dict[int, int], timeout_s: float = 3.0) -> RobotState | None:
        deadline = time.monotonic() + timeout_s
        previous = None
        while time.monotonic() < deadline and not self._aborted:
            state = await self.get_state()
            self._require_fresh_motor_rx(state, targets)
            pos = state.positions()
            samples = {mid: self.rx_received_at[(state.bus_serial, mid)] for mid in targets}
            if (all(mid in pos and abs(pos[mid] - tick) <= tol_ticks for mid, tick in targets.items())
                    and all(abs(pos[mid] - targets[mid]) <= PREFLIGHT_DEADBAND_TICKS
                            or abs(pos[mid] - targets[mid]) < abs(start - targets[mid])
                            for mid, start in moving_from.items())):
                if previous is None or previous[0] != pos:
                    previous = (pos, samples)
                elif all(stamp > previous[1][mid] for mid, stamp in samples.items()):
                    return state
            else:
                previous = None
            await asyncio.sleep(0.08)
        return None

    ### 7. Verify grip after bounded closing steps
    ## added by Abhi
    def _gripper_rx(self, state: RobotState, bus: str, mid: int) -> tuple[MotorState, float]:
        if state.bus_serial != bus:
            raise RuntimeError(f"Expected gripper bus {bus}, found {state.bus_serial}")
        motor = next((m for m in state.motors if m.id == mid), None)
        if motor is None or motor.error_status:
            raise RuntimeError(f"Gripper motor {mid} is missing or reports an error")
        stamp = self.rx_received_at.get((bus, mid), 0)
        if not 0 <= time.monotonic() - stamp <= RAW_RX_MAX_AGE_S:
            raise RuntimeError(f"Gripper motor {mid} requires fresh raw RX telemetry")
        return motor, stamp

    ### 7. Close and verify the grasp
    async def grasp_with_verify(
        self,
        close_ticks: int | None = None,
        open_ticks: int | None = None,
        clamp_to_range: bool = False,
    ) -> dict:
        result = {"ok": False, "holding": False}
        try:
            state = await self.get_state()
            bus = self.target_bus or state.bus_serial
            mid = 8 if bus == self._elrobot_range_bus else state.motors[-1].id
            motor, stamp = self._gripper_rx(state, bus, mid)
            lo, hi = self._effective_motor_range(bus, motor)
            target = lo if close_ticks is None else int(close_ticks)
            open_target = hi if open_ticks is None else int(open_ticks)
            if clamp_to_range:
                target = max(lo, min(hi, target))
                open_target = max(lo, min(hi, open_target))
            if not lo <= target <= hi or not lo <= open_target <= hi or open_target == target:
                raise ValueError("Gripper targets must be distinct and inside the calibrated range")
            direction = 1 if target > open_target else -1
            initial_position = motor.position
            min_current = max(GRIPPER_CONTACT_CURRENT_MA, motor.current_ma + 10)
            result.update(gripper_id=mid, requested_close=target, position=initial_position, current_ma=motor.current_ma)
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                if abs(motor.position - target) <= 25:
                    return {**result, "ok": True, "target": target, "reason": "closed without contact"}

                commanded = (await self.send_joint_targets({mid: target})).get(mid)
                if commanded is None:
                    raise RuntimeError("The bridge did not send a gripper target")
                stamp = max(stamp, self.rx_received_at.get((bus, mid), 0))
                result["target"] = commanded
                contact_since = None
                contact_position = None
                step_deadline = min(deadline, time.monotonic() + 1.8)
                while time.monotonic() < step_deadline:
                    sample, received = self._gripper_rx(await self.get_state(), bus, mid)
                    if received > stamp:
                        motor, stamp = sample, received
                        result.update(position=motor.position, current_ma=motor.current_ma)
                        if abs(motor.position - commanded) <= 25:
                            # An intermediate creep setpoint is not an indication of contact.
                            break
                        under_load = (
                            direction * (commanded - motor.position) > 25
                            and direction * (target - motor.position) > GRIPPER_SLACK_STEPS
                            and direction * (motor.position - initial_position) > 10
                            and motor.current_ma >= min_current
                        )
                        if not under_load:
                            contact_since = contact_position = None
                        elif contact_since is None or abs(motor.position - contact_position) > 6:
                            contact_since, contact_position = stamp, motor.position
                        elif stamp - contact_since >= 0.3:
                            return {**result, "ok": True, "holding": True}
                    await asyncio.sleep(0.08)
                else:
                    raise RuntimeError("Gripper did not reach the step target or verify contact")
            raise RuntimeError("Timed out closing the gripper")
        except (RuntimeError, ValueError, IndexError) as exc:
            return {**result, "reason": str(exc)}

    async def abort(self, source: str = "manual") -> dict:
        # Latch first so in-flight motion loops cannot send another target while the hold is applied.
        if not self._aborted or source == "manual":
            self._abort_source = source
        self._aborted = True
        # Hold Position uses live PresentPosition when fresh (prevents jumping
        # by unseen drift); if telemetry is stale, drop torque safely instead
        # of refusing an emergency stop.
        if (source != "reflex" or not self._follow_motor_rx) and hasattr(self.client, "read_from_tail"):
            await self._poll_inference_tail(timeout_s=1.0)
        try:
            self._require_fresh()
            if source == "reflex" and self._follow_motor_rx:
                serial = self._select_bus()
                motors = sorted(self._motor_state.get(serial, {}).values(), key=lambda m: m.id)
                if not motors:
                    raise RuntimeError("No follower telemetry for reflex hold")
                st = RobotState(serial, motors, bool(self._torque_started))
            else:
                st = await self.get_state()
            self._require_fresh_motor_rx(st, st.positions())
        except RuntimeError as exc:
            await self.set_torque(False)
            return {"aborted": True, "mode": "torque_off_stale", "reason": str(exc)}
        if self.target_bus and st.bus_serial != self.target_bus:
            raise RuntimeError(f"Expected follower bus {self.target_bus}, found {st.bus_serial}")
        hold = {m.id: m.position.to_bytes(2, "little") for m in st.motors}
        cmd = self._sync_write(st.bus_serial, RAM_GOAL_POSITION, hold)
        await self._send_commands(self.client, [cmd])
        return {"aborted": True, "mode": "hold", "source": self._abort_source, "positions": st.positions()}

    @property
    def abort_source(self) -> str | None:
        return self._abort_source

    def rearm(self) -> None:
        self._aborted = False
        self._abort_source = None


def _load_hardware_fallback() -> dict[str, str]:
    try:
        from .calib_store import CalibrationStore

        return CalibrationStore().hardware_config()
    except Exception:
        return {"bus_serial": "", "camera_top": "", "camera_wrist": ""}


def create_bridge(*, follow_motor_rx: bool = False) -> StationBridge:
    host = os.environ.get("STATION_HOST", "").strip()
    port = int(os.environ.get("STATION_PORT", "8888"))
    if os.environ.get("DESK_MOCK") == "1" or not host:
        return MockStationBridge()
    hw = _load_hardware_fallback()
    bus = os.environ.get("STATION_BUS_SERIAL", "").strip() or hw.get("bus_serial") or None
    cam_top = os.environ.get("CAMERA_TOP", "").strip() or hw.get("camera_top") or ""
    cam_wrist = os.environ.get("CAMERA_WRIST", "").strip() or hw.get("camera_wrist") or ""
    return LiveStationBridge(
        host=host, port=port, target_bus=bus, cam_top=cam_top, cam_wrist=cam_wrist,
        follow_motor_rx=follow_motor_rx,
    )
