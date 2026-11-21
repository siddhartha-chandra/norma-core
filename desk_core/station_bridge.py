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
import logging
import math
import os
import re
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image, ImageChops, ImageDraw, ImageStat

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
SIGN_BIT_MASK = 0x8000
FULL_STATE_LEN = 0x47
GRIPPER_SLACK_STEPS = 120
DEFAULT_MAX_STEP = int(os.environ.get("ROBOT_MAX_STEP_TICKS", "250"))
DEFAULT_SPEED = int(os.environ.get("ROBOT_GOAL_SPEED", "600"))
DEFAULT_ACCEL = int(os.environ.get("ROBOT_GOAL_ACCEL", "20"))

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


@runtime_checkable
class StationBridge(Protocol):
    async def connect(self) -> None: ...
    async def get_state(self) -> RobotState: ...
    async def get_frame(self, camera: str = "top", denoise: bool = False) -> bytes: ...
    async def list_cameras(self) -> list[str]: ...
    async def send_joint_targets(self, targets: dict[int, int], max_step: int = DEFAULT_MAX_STEP) -> dict[int, int]: ...
    async def set_torque(self, enable: bool, motor_ids: list[int] | None = None) -> bool: ...
    async def wait_settle(self, targets: dict[int, int], tol_ticks: int = 25, timeout_s: float = 3.0) -> bool: ...
    async def grasp_with_verify(self, close_ticks: int | None = None, open_ticks: int | None = None) -> dict: ...
    async def abort(self) -> dict: ...


class MockStationBridge:
    """Zero-hardware tabletop simulation for parallel team development and offline tests."""

    def __init__(self, num_joints: int | None = None):
        if num_joints is None:
            num_joints = int(os.environ.get("ROBOT_NUM_MOTORS", "8"))
        self.num_joints = num_joints
        self._torque_on = True
        self._aborted = False
        self._motors: dict[int, MotorState] = {
            mid: MotorState(mid, 2048, 35, 1000, 3100, 0)
            for mid in range(1, num_joints + 1)
        }

    async def connect(self) -> None:
        log.info("MockStationBridge connected (%d-DOF offline desk mode)", self.num_joints)

    async def get_state(self) -> RobotState:
        return RobotState("MOCK-BUS-0001", [self._motors[i] for i in sorted(self._motors)], self._torque_on)

    async def list_cameras(self) -> list[str]:
        return ["mock-top", "mock-wrist"]

    async def get_frame(self, camera: str = "top", denoise: bool = False) -> bytes:
        img = Image.new("RGB", (640, 480), (235, 232, 224) if camera == "top" else (42, 44, 52))
        d = ImageDraw.Draw(img)
        if camera == "top":
            # Draw synthetic Desk-Core props: mug obstacle, blue silicone key fob, chunky marker, departure tray
            d.rectangle([420, 40, 600, 210], fill=(210, 220, 230), outline=(70, 90, 120), width=3)
            d.text((445, 50), "DEPARTURE TRAY", fill=(50, 65, 90))
            d.ellipse([260, 180, 345, 265], fill=(165, 95, 55), outline=(90, 45, 20), width=3)
            d.text((272, 215), "MUG", fill=(255, 255, 255))
            d.polygon([(140, 310), (215, 285), (225, 315), (150, 340)], fill=(35, 115, 220))
            d.text((145, 345), "KEYS FOB", fill=(20, 60, 140))
            d.polygon([(90, 120), (195, 145), (190, 165), (85, 140)], fill=(220, 60, 50))
            d.text((95, 170), "MARKER", fill=(160, 35, 30))
        else:
            d.rectangle([20, 20, 620, 460], outline=(80, 180, 255), width=3)
            d.polygon([(260, 210), (370, 185), (385, 235), (275, 260)], fill=(35, 115, 220))
            d.line([(320, 0), (320, 480)], fill=(100, 220, 120), width=1)
            d.line([(0, 240), (640, 240)], fill=(100, 220, 120), width=1)
        d.text((12, 10), f"MOCK {camera.upper()} | j1={self._motors[1].position}", fill=(20, 20, 20) if camera == "top" else (240, 240, 240))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    async def send_joint_targets(self, targets: dict[int, int], max_step: int = DEFAULT_MAX_STEP) -> dict[int, int]:
        self._aborted = False
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

    async def grasp_with_verify(self, close_ticks: int | None = None, open_ticks: int | None = None) -> dict:
        gid = max(self._motors)
        m = self._motors[gid]
        target = close_ticks if close_ticks is not None else (m.range_min + GRIPPER_SLACK_STEPS + 80)
        m.position = max(m.range_min, min(m.range_max, target))
        gap = abs(m.position - m.range_min)
        return {"ok": True, "holding": gap > GRIPPER_SLACK_STEPS, "gripper_id": gid, "position": m.position, "current_ma": 180}

    async def abort(self) -> dict:
        self._aborted = True
        return {"aborted": True, "positions": {k: v.position for k, v in self._motors.items()}}

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
    ):
        self.host = host
        self.port = port
        self.target_bus = (target_bus or "").strip() or None
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
        self._frames: dict[str, bytes] = {}
        self._frame_buf: dict[str, deque[bytes]] = {}
        self._frame_order: list[str] = []
        self._followed_video_qids: set[str] = set()
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
        self._queues = {qid: asyncio.Queue() for qid in self._BASE_QUEUES}
        self._video_q = asyncio.Queue()
        self._follow_errors = {}
        for qid, q in self._queues.items():
            self._follow(qid, q)
        for qid in self._discover_local_video_queues():
            self._follow_video_queue(qid)

    _BASE_QUEUES = ("st3215/inference", "st3215/rx", "inference-states")

    def _follow(self, qid: str, q: asyncio.Queue) -> None:
        err_q = self.client.follow(qid, q)
        if err_q is not None:
            self._follow_errors[qid] = err_q

    def _start_workers(self) -> None:
        self._tasks = [
            asyncio.create_task(self._consume_inference(self._queues["st3215/inference"])),
            asyncio.create_task(self._consume_rx(self._queues["st3215/rx"])),
            asyncio.create_task(self._consume_inference_states(self._queues["inference-states"])),
            asyncio.create_task(self._consume_video(self._video_q)),
        ]

    def _follow_video_queue(self, qid: str) -> None:
        if qid not in self._followed_video_qids and self.client is not None:
            self._followed_video_qids.add(qid)
            self._follow(qid, self._video_q)
            log.info("Following camera queue: %s", qid)

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
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = []
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

    async def _consume_inference(self, q: asyncio.Queue) -> None:
        while True:
            entry = await q.get()
            if entry is None:
                break
            self._last_rx_mono = time.monotonic()
            try:
                reader = self._st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
                for bus in reader.get_buses():
                    info = bus.get_bus()
                    serial = info.get_serial_number() if info else ""
                    if not serial:
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
                        if mid in slot:
                            slot[mid].range_min = m.get_range_min()
                            slot[mid].range_max = m.get_range_max()
                        else:
                            slot[mid] = MotorState(
                                mid,
                                _norm_pos(_u16(sb, RAM_PRESENT_POSITION)),
                                _u16(sb, RAM_PRESENT_CURRENT),
                                m.get_range_min(),
                                m.get_range_max(),
                                sb[RAM_STATUS],
                            )
                    if volts:
                        self._bus_voltage[serial] = sum(volts) / len(volts)
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
                cur = _u16(sb, RAM_PRESENT_CURRENT)
                err = sb[RAM_STATUS]
                if mid in slot:
                    slot[mid].position = pos
                    slot[mid].current_ma = cur
                    slot[mid].error_status = err
                else:
                    slot[mid] = MotorState(mid, pos, cur, 0, MAX_ANGLE_STEP, err)
            except Exception as exc:
                log.debug("rx parse skip: %s", exc)

    async def _consume_video(self, q: asyncio.Queue) -> None:
        while True:
            entry = await q.get()
            if entry is None:
                break
            try:
                env = self._usbvideo.RxEnvelopeReader(memoryview(bytes(entry.Data)))
                if env.get_type() != self.ET_FRAMES:
                    continue
                cam = env.get_camera()
                key = cam.get_unique_id() or cam.get_serial_number() or cam.get_product() or "cam0"
                frames = env.get_frames().get_frames_data()
                if not frames:
                    continue
                jpeg = next(
                    (bytes(b) for b in reversed(frames) if len(bytes(b)) > 4 and bytes(b)[:2] == b"\xff\xd8" and bytes(b)[-2:] == b"\xff\xd9"),
                    None,
                )
                if jpeg is None:
                    continue
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
            if self._motor_state and len(self._frames) >= want_cams:
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

    async def get_frame(self, camera: str = "top", denoise: bool = False) -> bytes:
        for _ in range(50):
            if self._frames:
                break
            await asyncio.sleep(0.1)
        if not self._frames:
            raise RuntimeError("No USB video frames received from Station yet.")
        key = self._resolve_cam_key(camera)
        if denoise:
            buf = list(self._frame_buf.get(key, ()))
            if len(buf) >= 3:
                return _cleanest_frame(buf)
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
            # Safe-start writes Goal=Present; both must come from live telemetry.
            self._require_fresh()
        st = await self.get_state()
        mids = motor_ids or [m.id for m in st.motors]
        pack = []
        if enable:
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

    async def send_joint_targets(self, targets: dict[int, int], max_step: int = DEFAULT_MAX_STEP) -> dict[int, int]:
        self._require_fresh()
        st = await self.get_state()
        by_id = {m.id: m for m in st.motors}
        applied: dict[int, int] = {}
        need_start = [int(mid) for mid in targets if int(mid) in by_id and int(mid) not in self._torque_started]
        if need_start:
            await self.set_torque(True, need_start)

        for raw_mid, raw_tgt in targets.items():
            mid = int(raw_mid)
            if mid not in by_id:
                continue
            m = by_id[mid]
            lo, hi = (m.range_min, m.range_max) if m.range_max > m.range_min else (0, MAX_ANGLE_STEP)
            clamped = max(lo, min(hi, int(raw_tgt)))
            delta = max(-max_step, min(max_step, clamped - m.position))
            applied[mid] = m.position + delta

        if applied:
            cmd = self._sync_write(
                st.bus_serial,
                RAM_GOAL_POSITION,
                {mid: val.to_bytes(2, "little") for mid, val in applied.items()},
            )
            await self._send_commands(self.client, [cmd])
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

    async def grasp_with_verify(self, close_ticks: int | None = None, open_ticks: int | None = None) -> dict:
        st = await self.get_state()
        gripper = st.motors[-1]
        tgt = close_ticks if close_ticks is not None else gripper.range_min
        applied = await self.send_joint_targets({gripper.id: tgt}, max_step=1200)
        await asyncio.sleep(1.8)
        st2 = await self.get_state()
        g2 = next(m for m in st2.motors if m.id == gripper.id)
        gap = abs(g2.position - gripper.range_min)
        return {
            "ok": True,
            "holding": gap > GRIPPER_SLACK_STEPS,
            "gripper_id": g2.id,
            "position": g2.position,
            "target": applied.get(gripper.id, tgt),
            "current_ma": g2.current_ma,
        }

    async def abort(self) -> dict:
        # Hold Position uses live PresentPosition when fresh (prevents jumping
        # by unseen drift); if telemetry is stale, drop torque safely instead
        # of refusing an emergency stop.
        try:
            self._require_fresh()
        except RuntimeError as exc:
            await self.set_torque(False)
            return {"aborted": True, "mode": "torque_off_stale", "reason": str(exc)}
        st = await self.get_state()
        hold = {m.id: m.position.to_bytes(2, "little") for m in st.motors}
        cmd = self._sync_write(st.bus_serial, RAM_GOAL_POSITION, hold)
        await self._send_commands(self.client, [cmd])
        return {"aborted": True, "mode": "hold", "positions": st.positions()}


def _load_hardware_fallback() -> dict[str, str]:
    try:
        from .calib_store import CalibrationStore

        return CalibrationStore().hardware_config()
    except Exception:
        return {"bus_serial": "", "camera_top": "", "camera_wrist": ""}


def create_bridge() -> StationBridge:
    host = os.environ.get("STATION_HOST", "").strip()
    port = int(os.environ.get("STATION_PORT", "8888"))
    if os.environ.get("DESK_MOCK") == "1" or not host:
        return MockStationBridge()
    hw = _load_hardware_fallback()
    bus = os.environ.get("STATION_BUS_SERIAL", "").strip() or hw.get("bus_serial") or None
    cam_top = os.environ.get("CAMERA_TOP", "").strip() or hw.get("camera_top") or ""
    cam_wrist = os.environ.get("CAMERA_WRIST", "").strip() or hw.get("camera_wrist") or ""
    return LiveStationBridge(host=host, port=port, target_bus=bus, cam_top=cam_top, cam_wrist=cam_wrist)
