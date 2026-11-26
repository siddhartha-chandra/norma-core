"""Desk object sightings history store (DESK-CORE issue #26).

Primary backend: NormFS append-only queue (`desk_core/sightings` by default) over
the active NormaCore Station TCP client.
Fallback backend: local SQLite `sightings` table when NormFS is disabled or
unavailable (disconnected, mock bridge, or enqueue failure).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("desk_core.sightings")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NORMFS_QUEUE = "desk_core/sightings"
_FALSE_VALUES = {"0", "false", "no", "off"}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    time REAL NOT NULL,
    frame_ref TEXT NOT NULL,
    z REAL,
    camera TEXT,
    confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_sightings_label_time ON sightings (label, time DESC);
CREATE INDEX IF NOT EXISTS idx_sightings_time ON sightings (time DESC);
"""


def normfs_sightings_enabled() -> bool:
    """Return False when DESK_SIGHTINGS_NORMFS explicitly disables NormFS writes."""
    raw = os.environ.get("DESK_SIGHTINGS_NORMFS", "1").strip().lower()
    return raw not in _FALSE_VALUES


def default_sightings_queue() -> str:
    return os.environ.get("DESK_SIGHTINGS_QUEUE", DEFAULT_NORMFS_QUEUE).strip() or DEFAULT_NORMFS_QUEUE


def default_sightings_db_path() -> Path:
    explicit = os.environ.get("DESK_SIGHTINGS_DB", "").strip()
    if explicit:
        return Path(explicit)
    station_data = os.environ.get("STATION_DATA_DIR", "").strip()
    if station_data:
        return Path(station_data) / "sightings.db"
    return REPO_ROOT / "station_data" / "sightings.db"


@dataclass(frozen=True)
class Sighting:
    """One timestamped desk object observation."""

    label: str
    x: float
    y: float
    time: float
    frame_ref: str
    z: float | None = None
    camera: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "time": self.time,
            "frame_ref": self.frame_ref,
        }
        if self.z is not None:
            payload["z"] = self.z
        if self.camera is not None:
            payload["camera"] = self.camera
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sighting:
        label = str(data["label"]).strip()
        if not label:
            raise ValueError("Sighting label must be non-empty")
        x = float(data["x"])
        y = float(data["y"])
        ts = float(data["time"])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(ts)):
            raise ValueError("Sighting x, y, and time must be finite numbers")
        frame_ref = str(data.get("frame_ref", ""))
        z_raw = data.get("z")
        z = float(z_raw) if z_raw is not None else None
        if z is not None and not math.isfinite(z):
            raise ValueError("Sighting z must be finite")
        cam_raw = data.get("camera")
        camera = str(cam_raw) if cam_raw is not None else None
        conf_raw = data.get("confidence")
        confidence = float(conf_raw) if conf_raw is not None else None
        if confidence is not None and not math.isfinite(confidence):
            raise ValueError("Sighting confidence must be finite")
        return cls(
            label=label,
            x=x,
            y=y,
            time=ts,
            frame_ref=frame_ref,
            z=z,
            camera=camera,
            confidence=confidence,
        )

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview) -> Sighting:
        data = json.loads(bytes(raw).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Sighting payload must be a JSON object")
        return cls.from_dict(data)


@dataclass(frozen=True)
class SightingWriteResult:
    """Result returned after persisting a sighting."""

    sighting: Sighting
    backend: str  # "normfs" | "sqlite"
    record_id: str
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "backend": self.backend,
            "record_id": self.record_id,
            "label": self.sighting.label,
            "x": self.sighting.x,
            "y": self.sighting.y,
            "time": self.sighting.time,
            "frame_ref": self.sighting.frame_ref,
        }
        if self.fallback_reason is not None:
            out["fallback_reason"] = self.fallback_reason
        return out


class SightingsStore:
    """Append-only desk sighting store: NormFS primary, local SQLite fallback."""

    def __init__(
        self,
        *,
        normfs_client: Any | None = None,
        client_provider: Callable[[], Any | None] | None = None,
        queue_id: str | None = None,
        db_path: str | os.PathLike | None = None,
        normfs_enabled: bool | None = None,
        timeout_s: float = 5.0,
    ):
        self._normfs_client = normfs_client
        self._client_provider = client_provider
        self._queue_id = queue_id
        self._db_path = Path(db_path) if db_path is not None else None
        self._normfs_enabled = normfs_enabled
        self.timeout_s = float(timeout_s)
        self._sqlite_initialized: set[Path] = set()

    @property
    def queue_id(self) -> str:
        return self._queue_id if self._queue_id is not None else default_sightings_queue()

    @property
    def db_path(self) -> Path:
        return self._db_path if self._db_path is not None else default_sightings_db_path()

    @property
    def normfs_enabled(self) -> bool:
        if self._normfs_enabled is not None:
            return self._normfs_enabled
        return normfs_sightings_enabled()

    def _resolve_client(self) -> Any | None:
        client = self._normfs_client
        if client is None and self._client_provider is not None:
            client = self._client_provider()
        if client is None:
            return None
        if not getattr(client, "connected", True) or not getattr(client, "setup_done", True):
            return None
        if not hasattr(client, "enqueue"):
            return None
        return client

    def _open_sqlite(self, *, create: bool) -> sqlite3.Connection | None:
        path = self.db_path
        if not create and not path.exists():
            return None
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        if create and path not in self._sqlite_initialized:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            self._sqlite_initialized.add(path)
        return conn

    def _write_sqlite(self, sighting: Sighting) -> str:
        conn = self._open_sqlite(create=True)
        assert conn is not None
        try:
            cur = conn.execute(
                """
                INSERT INTO sightings (label, x, y, time, frame_ref, z, camera, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sighting.label,
                    sighting.x,
                    sighting.y,
                    sighting.time,
                    sighting.frame_ref,
                    sighting.z,
                    sighting.camera,
                    sighting.confidence,
                ),
            )
            conn.commit()
            return str(cur.lastrowid)
        finally:
            conn.close()

    async def record(
        self,
        label: str,
        x: float,
        y: float,
        *,
        time_s: float | None = None,
        frame_ref: str = "",
        z: float | None = None,
        camera: str | None = None,
        confidence: float | None = None,
    ) -> SightingWriteResult:
        """Persist one sighting to NormFS, falling back to SQLite if unavailable."""
        ts = float(time.time() if time_s is None else time_s)
        sighting = Sighting.from_dict(
            {
                "label": label,
                "x": x,
                "y": y,
                "time": ts,
                "frame_ref": frame_ref,
                "z": z,
                "camera": camera,
                "confidence": confidence,
            }
        )

        if not self.normfs_enabled:
            row_id = await asyncio.to_thread(self._write_sqlite, sighting)
            return SightingWriteResult(
                sighting=sighting,
                backend="sqlite",
                record_id=row_id,
                fallback_reason="normfs_disabled",
            )

        client = self._resolve_client()
        if client is None:
            row_id = await asyncio.to_thread(self._write_sqlite, sighting)
            return SightingWriteResult(
                sighting=sighting,
                backend="sqlite",
                record_id=row_id,
                fallback_reason="normfs_unavailable",
            )

        try:
            entry_id = await asyncio.wait_for(
                client.enqueue(self.queue_id, sighting.to_bytes()),
                timeout=self.timeout_s,
            )
            rec_id = (
                bytes(entry_id).hex()
                if isinstance(entry_id, (bytes, bytearray, memoryview))
                else str(entry_id)
            )
            return SightingWriteResult(
                sighting=sighting,
                backend="normfs",
                record_id=rec_id,
            )
        except Exception as exc:
            reason = f"{exc.__class__.__name__}: {exc}"
            log.warning("NormFS sighting write failed (%s); falling back to SQLite %s", reason, self.db_path)
            row_id = await asyncio.to_thread(self._write_sqlite, sighting)
            return SightingWriteResult(
                sighting=sighting,
                backend="sqlite",
                record_id=row_id,
                fallback_reason=reason,
            )

    async def record_localization(
        self,
        detection: dict[str, Any],
        *,
        frame_ref: str,
        time_s: float | None = None,
    ) -> SightingWriteResult | None:
        """Record a sighting from a `localize_3d` / `localize_object_3d` result dict."""
        if not detection.get("found"):
            return None
        label = str(detection.get("label") or detection.get("query") or "").strip()
        world = detection.get("world") or {}
        if not label or "x" not in world or "y" not in world:
            return None
        x = float(world["x"])
        y = float(world["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        z_raw = world.get("z")
        z = float(z_raw) if z_raw is not None and math.isfinite(float(z_raw)) else None
        conf_raw = detection.get("confidence")
        confidence = float(conf_raw) if conf_raw is not None and math.isfinite(float(conf_raw)) else None
        camera = detection.get("camera")
        return await self.record(
            label=label,
            x=x,
            y=y,
            time_s=time_s,
            frame_ref=frame_ref,
            z=z,
            camera=str(camera) if camera is not None else None,
            confidence=confidence,
        )

    async def _drain_queue_read(
        self, qr: Any, deadline: float
    ) -> tuple[list[Sighting], bytes | None, int]:
        entries: list[Sighting] = []
        last_id: bytes | None = None
        raw_count = 0
        while time.monotonic() < deadline:
            rem = max(0.05, deadline - time.monotonic())
            try:
                entry = await asyncio.wait_for(qr.data.get(), timeout=rem)
            except asyncio.TimeoutError:
                break
            if entry is None:
                break
            raw_count += 1
            raw_id = getattr(getattr(entry, "ID", None), "ID", None)
            if isinstance(raw_id, (bytes, bytearray, memoryview)):
                last_id = bytes(raw_id)
            try:
                entries.append(Sighting.from_bytes(entry.Data))
            except Exception as exc:
                log.debug("Skipping malformed NormFS sighting entry: %s", exc)
        return entries, last_id, raw_count

    async def read_normfs(
        self,
        limit: int = 1000,
        *,
        label: str | None = None,
        before_time: float | None = None,
    ) -> list[Sighting]:
        """Read sightings stored in the NormFS queue, newest first."""
        if limit <= 0 or not self.normfs_enabled:
            return []
        client = self._resolve_client()
        if client is None:
            return []

        deadline = time.monotonic() + self.timeout_s
        filtered = label is not None or before_time is not None

        # When filtering by label/time, paginate from offset 0 so sightings older than
        # a fixed tail window are never silently missed.
        if filtered and hasattr(client, "read_from_offset"):
            page_size = 1000
            buf_size = 256
            next_idx = 0
            matched: list[Sighting] = []
            want_label = label.strip().casefold() if label is not None else None
            cutoff = float(before_time) if before_time is not None else None
            while time.monotonic() < deadline:
                offset = next_idx.to_bytes(8, "little") if next_idx > 0 else b"\x00"
                qr = client.read_from_offset(self.queue_id, offset=offset, limit=page_size, step=1, buf_size=buf_size)
                page, last_id, raw_count = await self._drain_queue_read(qr, deadline)
                for s in page:
                    if want_label is not None and s.label.casefold() != want_label:
                        continue
                    if cutoff is not None and s.time > cutoff:
                        continue
                    matched.append(s)
                if raw_count < page_size or last_id is None:
                    break
                next_idx = int.from_bytes(last_id, "little") + 1
            return _filter_and_sort(matched, limit=limit, label=label, before_time=before_time)

        buf_size = min(max(limit, 16), 256)
        if hasattr(client, "read_from_tail"):
            offset = (limit - 1).to_bytes(8, "little") if limit > 1 else b"\x00"
            qr = client.read_from_tail(self.queue_id, offset=offset, limit=limit, step=1, buf_size=buf_size)
        elif hasattr(client, "read_from_offset"):
            qr = client.read_from_offset(self.queue_id, offset=b"\x00", limit=limit, step=1, buf_size=buf_size)
        else:
            return []

        entries, _, _ = await self._drain_queue_read(qr, deadline)
        return _filter_and_sort(entries, limit=limit, label=label, before_time=before_time)

    def read_sqlite(
        self,
        limit: int = 1000,
        *,
        label: str | None = None,
        before_time: float | None = None,
    ) -> list[Sighting]:
        """Read sightings stored in the SQLite fallback table, newest first."""
        if limit <= 0:
            return []
        conn = self._open_sqlite(create=False)
        if conn is None:
            return []
        try:
            clauses: list[str] = []
            params: list[Any] = []
            if label is not None:
                clauses.append("LOWER(label) = LOWER(?)")
                params.append(label.strip())
            if before_time is not None:
                clauses.append("time <= ?")
                params.append(float(before_time))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(int(limit))
            rows = conn.execute(
                f"""
                SELECT label, x, y, time, frame_ref, z, camera, confidence
                FROM sightings
                {where}
                ORDER BY time DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                Sighting(
                    label=row["label"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    time=float(row["time"]),
                    frame_ref=row["frame_ref"],
                    z=float(row["z"]) if row["z"] is not None else None,
                    camera=row["camera"],
                    confidence=float(row["confidence"]) if row["confidence"] is not None else None,
                )
                for row in rows
            ]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    async def list_sightings(
        self,
        limit: int = 1000,
        *,
        label: str | None = None,
        before_time: float | None = None,
    ) -> list[Sighting]:
        """Read sightings across NormFS (when active) and SQLite fallback, newest first."""
        combined: list[Sighting] = []
        if self.normfs_enabled:
            try:
                combined.extend(await self.read_normfs(limit=limit, label=label, before_time=before_time))
            except Exception as exc:
                log.debug("NormFS sighting read skipped: %s", exc)
        combined.extend(await asyncio.to_thread(self.read_sqlite, limit, label=label, before_time=before_time))
        seen: set[tuple[str, float, float, float, str]] = set()
        deduped: list[Sighting] = []
        for item in combined:
            key = (item.label, item.x, item.y, item.time, item.frame_ref)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return _filter_and_sort(deduped, limit=limit, label=label, before_time=before_time)


def _filter_and_sort(
    sightings: list[Sighting],
    *,
    limit: int,
    label: str | None,
    before_time: float | None,
) -> list[Sighting]:
    out = sightings
    if label is not None:
        want = label.strip().lower()
        out = [s for s in out if s.label.lower() == want]
    if before_time is not None:
        cutoff = float(before_time)
        out = [s for s in out if s.time <= cutoff]
    out.sort(key=lambda s: s.time, reverse=True)
    return out[:limit]
