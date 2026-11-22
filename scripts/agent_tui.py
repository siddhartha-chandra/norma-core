#!/usr/bin/env python3
"""DESK-CORE Voice & Text Robot Agent TUI with Secure Vision-Only LLM Plugins.

Features:
  - Text OR Voice commands (`Ctrl+V` / Speak button) routed to the robot arm over MCP
  - Vision-only model dropdown (Gemini, Claude, OpenAI — only image-capable models)
  - Secure API key manager (`Ctrl+K`) stored in ~/.config/desk-core/keys.json (`0o600`)
  - Arm Motion safety interlock + always-on E-STOP (`Ctrl+E` / `F12`)

Usage:
  uv run python scripts/agent_tui.py --mock
  uv run python scripts/agent_tui.py --live
  uv run python scripts/agent_tui.py --test
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image, ImageDraw
from textual import events, on
from textual.app import App, ComposeResult
from textual_image.widget import Image as TerminalImage
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from desk_core.llm_plugins import (
    SecureKeyStore,
    record_wav_sync,
    run_agent_command,
    speak_text_sync,
    transcribe_wav_sync,
)

MOTION_TOOLS = {
    "move_joints",
    "move_joint",
    "go_home",
    "move_to_pose",
    "pick_at",
    "place_at",
    "pick_pixel",
    "execute_sequence",
}


def _server_params(live: bool) -> StdioServerParameters:
    env = dict(os.environ)
    if live:
        env.pop("DESK_MOCK", None)
        env.pop("DESK_NO_DOTENV", None)
    else:
        env["DESK_MOCK"] = "1"
        env["DESK_MOCK_SYNTHETIC"] = "1"
        env["DESK_NO_DOTENV"] = "1"
        env["STATION_HOST"] = ""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "desk_core.mcp_server"],
        env=env,
        cwd=str(REPO_ROOT),
    )


class KeyConfigModal(ModalScreen[bool]):
    """Secure modal for adding vision-capable LLM API keys (Gemini, Claude, OpenAI)."""

    CSS = """
    KeyConfigModal {
        align: center middle;
    }
    #key-dialog {
        width: 74;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    .key-row {
        height: 3;
        align: left middle;
        margin-bottom: 1;
    }
    .key-row Input {
        width: 1fr;
        margin-right: 1;
    }
    #key-actions {
        height: 3;
        align: right middle;
    }
    #key-actions Button {
        margin-left: 1;
    }
    """

    def __init__(self, keystore: SecureKeyStore) -> None:
        super().__init__()
        self.keystore = keystore
        self._changed = False

    def compose(self) -> ComposeResult:
        st = self.keystore.status()
        with Vertical(id="key-dialog"):
            yield Label("[b]Secure Vision LLM Plugins (0600 local keystore)[/b]")
            yield Label(f"[dim]Only image-capable providers allowed. Stored at: {self.keystore.path}[/dim]")
            yield Label(f"Google Gemini ({st['gemini']['masked'] or 'not set'}):", id="lbl-gemini")
            with Horizontal(classes="key-row"):
                yield Input(placeholder="Paste GEMINI_API_KEY (leave blank to keep)", password=True, id="in-gemini")
                yield Button("Unset", variant="error", id="btn-unset-gemini")
            yield Label(f"Anthropic Claude ({st['claude']['masked'] or 'not set'}):", id="lbl-claude")
            with Horizontal(classes="key-row"):
                yield Input(placeholder="Paste ANTHROPIC_API_KEY (leave blank to keep)", password=True, id="in-claude")
                yield Button("Unset", variant="error", id="btn-unset-claude")
            yield Label(f"OpenAI ({st['openai']['masked'] or 'not set'}):", id="lbl-openai")
            with Horizontal(classes="key-row"):
                yield Input(placeholder="Paste OPENAI_API_KEY (leave blank to keep)", password=True, id="in-openai")
                yield Button("Unset", variant="error", id="btn-unset-openai")
            yield Label(f"Qwen VL ({st['qwen']['masked'] or 'not set'}):", id="lbl-qwen")
            with Horizontal(classes="key-row"):
                yield Input(placeholder="Paste QWEN_API_KEY / DASHSCOPE_API_KEY", password=True, id="in-qwen")
                yield Button("Unset", variant="error", id="btn-unset-qwen")
            yield Label(f"DeepSeek VL ({st['deepseek']['masked'] or 'not set'}):", id="lbl-deepseek")
            with Horizontal(classes="key-row"):
                yield Input(placeholder="Paste DEEPSEEK_API_KEY (leave blank to keep)", password=True, id="in-deepseek")
                yield Button("Unset", variant="error", id="btn-unset-deepseek")
            with Horizontal(id="key-actions"):
                yield Button("Save Keys", variant="primary", id="btn-save-keys")
                yield Button("Close", id="btn-cancel-keys")

    @on(Button.Pressed, "#btn-unset-gemini")
    def _unset_gemini(self) -> None:
        self._unset("gemini", "Google Gemini", "#lbl-gemini", "#in-gemini")

    @on(Button.Pressed, "#btn-unset-claude")
    def _unset_claude(self) -> None:
        self._unset("claude", "Anthropic Claude", "#lbl-claude", "#in-claude")

    @on(Button.Pressed, "#btn-unset-openai")
    def _unset_openai(self) -> None:
        self._unset("openai", "OpenAI", "#lbl-openai", "#in-openai")

    @on(Button.Pressed, "#btn-unset-qwen")
    def _unset_qwen(self) -> None:
        self._unset("qwen", "Qwen VL", "#lbl-qwen", "#in-qwen")

    @on(Button.Pressed, "#btn-unset-deepseek")
    def _unset_deepseek(self) -> None:
        self._unset("deepseek", "DeepSeek VL", "#lbl-deepseek", "#in-deepseek")

    def _unset(self, prov: str, title: str, lbl_id: str, in_id: str) -> None:
        self.keystore.unset_key(prov)
        self._changed = True
        self.query_one(in_id, Input).value = ""
        self.query_one(lbl_id, Label).update(f"{title} (unset):")

    @on(Button.Pressed, "#btn-save-keys")
    def _save(self) -> None:
        for prov, wid in [
            ("gemini", "#in-gemini"),
            ("claude", "#in-claude"),
            ("openai", "#in-openai"),
            ("qwen", "#in-qwen"),
            ("deepseek", "#in-deepseek"),
        ]:
            val = self.query_one(wid, Input).value.strip()
            if val:
                self.keystore.set_key(prov, val)
                self._changed = True
        self.dismiss(self._changed)

    @on(Button.Pressed, "#btn-cancel-keys")
    def _cancel(self) -> None:
        self.dismiss(self._changed)


class InteractivePlaneBox(Vertical):
    """Clickable & draggable orthographic image projection for 3D object placement."""

    IMG_W = 320
    IMG_H = 190
    COLOR_RGB = {
        "blue": (55, 135, 235),
        "red": (220, 65, 60),
        "green": (55, 185, 100),
        "yellow": (235, 190, 55),
        "purple": (160, 85, 215),
        "pink": (230, 115, 175),
        "brown": (140, 88, 55),
        "orange": (235, 135, 45),
    }
    OBJ_HEIGHT_M = {
        "cylinder": 0.09,
        "mug": 0.09,
        "pen": 0.012,
        "wallet": 0.025,
        "airpods": 0.025,
        "wired_earphones": 0.015,
        "mobile_phone": 0.010,
        "sphere": 0.048,
        "cube": 0.045,
    }

    def __init__(
        self,
        plane_id: str,
        title: str,
        h_label: str,
        v_label: str,
        h_range: tuple[float, float],
        v_range: tuple[float, float],
        on_pick: Any,
        id: str | None = None,
    ) -> None:
        super().__init__(classes="plane-box", id=id)
        self.plane_id = plane_id
        self.title = title
        self.h_label = h_label
        self.v_label = v_label
        self.h_range = h_range
        self.v_range = v_range
        self.on_pick = on_pick
        self._cached_key: str | None = None
        self._cached_base: Image.Image | None = None
        self._dragging: bool = False
        self._last_emit_ts: float = 0.0

    def compose(self) -> ComposeResult:
        yield TerminalImage(classes="plane-img")

    def _to_px(self, val_h: float, val_v: float) -> tuple[float, float]:
        u = (val_h - self.h_range[0]) / max(1e-6, self.h_range[1] - self.h_range[0]) * (self.IMG_W - 1)
        v = (self.v_range[1] - val_v) / max(1e-6, self.v_range[1] - self.v_range[0]) * (self.IMG_H - 1)
        return u, v

    def _scale_h(self, meters: float) -> float:
        return meters / max(1e-6, self.h_range[1] - self.h_range[0]) * (self.IMG_W - 1)

    def _scale_v(self, meters: float) -> float:
        return meters / max(1e-6, self.v_range[1] - self.v_range[0]) * (self.IMG_H - 1)

    def _draw_obj(
        self,
        d: ImageDraw.ImageDraw,
        name: str,
        obj: dict[str, Any],
        is_target: bool = False,
    ) -> None:
        x, y, z = float(obj.get("x", 0.25)), float(obj.get("y", 0.0)), float(obj.get("z", 0.0))
        r_m = float(obj.get("radius_m", 0.025))
        shape = obj.get("shape", "cube")
        h_m = self.OBJ_HEIGHT_M.get("mug" if name == "mug" else shape, r_m * 1.8)
        rgb = self.COLOR_RGB.get(obj.get("color", "blue"), (70, 140, 220))
        outline = (255, 230, 90) if is_target else (235, 240, 248)
        ow = 2 if is_target else 1

        if self.plane_id == "xy":
            u, v = self._to_px(x, y)
            rx = max(5.0, self._scale_h(r_m))
            ry = max(5.0, self._scale_v(r_m))
            if shape in ("cylinder", "sphere") or name == "mug":
                d.ellipse([u - rx, v - ry, u + rx, v + ry], fill=rgb, outline=outline, width=ow)
                if name == "mug" or shape == "cylinder":
                    d.ellipse([u - rx * 0.55, v - ry * 0.55, u + rx * 0.55, v + ry * 0.55], fill=(70, 42, 26))
            elif shape == "pen":
                d.rounded_rectangle([u - rx * 1.6, v - ry * 0.4, u + rx * 1.6, v + ry * 0.4], radius=3, fill=rgb, outline=outline, width=ow)
            elif shape == "mobile_phone":
                d.rounded_rectangle([u - rx * 0.85, v - ry * 1.45, u + rx * 0.85, v + ry * 1.45], radius=4, fill=(32, 38, 48), outline=outline, width=ow)
            elif shape == "wallet":
                d.rounded_rectangle([u - rx * 1.25, v - ry * 0.9, u + rx * 1.25, v + ry * 0.9], radius=4, fill=rgb, outline=outline, width=ow)
            else:
                d.rounded_rectangle([u - rx, v - ry, u + rx, v + ry], radius=4, fill=rgb, outline=outline, width=ow)
            d.text((u - rx, max(2, v - ry - 12)), ("* " + name) if is_target else name, fill=outline)
        else:
            coord_h = x if self.plane_id == "xz" else y
            u, v_bot = self._to_px(coord_h, z)
            _, v_top = self._to_px(coord_h, z + h_m)
            rx = max(5.0, self._scale_h(r_m))
            if shape == "sphere":
                ry = max(5.0, 0.5 * abs(v_bot - v_top))
                yc = 0.5 * (v_bot + v_top)
                d.ellipse([u - rx, yc - ry, u + rx, yc + ry], fill=rgb, outline=outline, width=ow)
            else:
                d.rectangle([u - rx, min(v_top, v_bot - 4), u + rx, v_bot], fill=rgb, outline=outline, width=ow)
            d.text((u - rx, max(2, min(v_top, v_bot - 4) - 12)), ("* " + name) if is_target else name, fill=outline)

    def _build_base(
        self,
        target_name: str,
        scene_objects: dict[str, dict[str, Any]],
    ) -> Image.Image:
        img = Image.new("RGB", (self.IMG_W, self.IMG_H), (18, 24, 36))
        d = ImageDraw.Draw(img)

        step = 0.05
        h_cur = math.ceil(self.h_range[0] / step) * step
        while h_cur <= self.h_range[1] + 1e-6:
            gx, _ = self._to_px(h_cur, self.v_range[0])
            d.line([(gx, 0), (gx, self.IMG_H)], fill=(34, 45, 64), width=1)
            h_cur += step
        v_cur = math.ceil(self.v_range[0] / step) * step
        while v_cur <= self.v_range[1] + 1e-6:
            _, gy = self._to_px(self.h_range[0], v_cur)
            col = (85, 105, 135) if abs(v_cur) < 1e-5 and self.plane_id != "xy" else (34, 45, 64)
            d.line([(0, gy), (self.IMG_W, gy)], fill=col, width=1)
            v_cur += step

        def _depth_key(kv: tuple[str, dict[str, Any]]) -> float:
            o = kv[1]
            if self.plane_id == "xy":
                return float(o.get("z", 0.0))
            if self.plane_id == "xz":
                return -float(o.get("y", 0.0))
            return float(o.get("x", 0.0))

        for name, obj in sorted(scene_objects.items(), key=_depth_key):
            if name == target_name:
                continue
            self._draw_obj(d, name, obj, is_target=False)
        d.text((6, 4), f"{self.title} ({self.h_label}, {self.v_label})", fill=(195, 210, 235))
        return img

    def update_plot(
        self,
        target_name: str,
        target_obj: dict[str, Any],
        scene_objects: dict[str, dict[str, Any]],
    ) -> None:
        cache_key = f"{target_name}:{len(scene_objects)}"
        if self._cached_base is None or self._cached_key != cache_key:
            self._cached_base = self._build_base(target_name, scene_objects)
            self._cached_key = cache_key

        img = self._cached_base.copy()
        d = ImageDraw.Draw(img)
        self._draw_obj(d, target_name, target_obj, is_target=True)
        th = float(target_obj["x"] if self.plane_id != "yz" else target_obj["y"])
        tv = float(target_obj["y"] if self.plane_id == "xy" else target_obj["z"])
        cu, cv = self._to_px(th, tv)
        d.line([(cu - 12, cv), (cu + 12, cv)], fill=(255, 230, 90), width=1)
        d.line([(cu, cv - 12), (cu, cv + 12)], fill=(255, 230, 90), width=1)
        self.query_one(TerminalImage).image = img

    def _emit_from_mouse(self, event: events.MouseEvent) -> None:
        w = max(1, self.size.width - 1)
        h = max(1, self.size.height - 1)
        u_frac = max(0.0, min(1.0, event.x / w))
        v_frac = max(0.0, min(1.0, event.y / h))
        h_val = self.h_range[0] + u_frac * (self.h_range[1] - self.h_range[0])
        v_val = self.v_range[1] - v_frac * (self.v_range[1] - self.v_range[0])
        self.on_pick(self.plane_id, round(h_val, 3), round(v_val, 3))

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 1:
            self._dragging = True
            self.capture_mouse()
            self._last_emit_ts = time.monotonic()
            self._emit_from_mouse(event)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging or event.button == 1:
            now = time.monotonic()
            if now - self._last_emit_ts >= 0.045:
                self._last_emit_ts = now
                self._emit_from_mouse(event)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()
            self._emit_from_mouse(event)


class PlacementModal(ModalScreen[dict[str, Any] | None]):
    """3-plane (X-Y, X-Z, Y-Z) interactive 3D placement routine modal."""

    CSS = """
    PlacementModal { align: center middle; }
    #place-dialog {
        width: 110;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #place-top-row { height: auto; align: left middle; margin-bottom: 1; }
    #place-obj-select { width: 32; margin-right: 1; }
    #place-planes { height: 14; margin-bottom: 1; }
    .plane-box { width: 1fr; height: 14; border: solid $primary; padding: 0 1; margin-right: 1; }
    .plane-img { width: 1fr; height: 12; }
    .coord-row { height: 3; align: left middle; margin-bottom: 0; }
    .coord-row Label { width: 10; }
    .coord-row Input { width: 12; margin-right: 1; }
    .coord-row Button { margin-right: 1; min-width: 6; }
    #place-actions { height: 3; align: right middle; margin-top: 1; }
    #place-actions Button { margin-left: 1; }
    """

    BINDINGS = [Binding("escape", "close_modal", "Close")]

    def __init__(self, scene_objects: dict[str, dict[str, Any]], presets: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.scene_objects = dict(scene_objects)
        self.presets = dict(presets)
        first = next(iter(self.scene_objects), "pen")
        init_pos = self.scene_objects.get(first) or self.presets.get(first) or {"x": 0.25, "y": 0.0, "z": 0.0}
        self.tx = float(init_pos.get("x", 0.25))
        self.ty = float(init_pos.get("y", 0.0))
        self.tz = float(init_pos.get("z", 0.0))

    def compose(self) -> ComposeResult:
        opts: list[tuple[str, str]] = []
        seen: set[str] = set()
        for name in self.scene_objects:
            opts.append((f"{name} (in scene)", name))
            seen.add(name)
        for name in self.presets:
            if name not in seen:
                opts.append((f"{name} (preset)", name))
                seen.add(name)
        default_obj = opts[0][1] if opts else "pen"

        with Vertical(id="place-dialog"):
            yield Label("[b]3D Placement Routine (Click or Drag in X-Y, X-Z, Y-Z Projected Views)[/b]")
            with Horizontal(id="place-top-row"):
                yield Label("Object: ")
                yield Select(opts, value=default_obj, allow_blank=False, id="place-obj-select")
                yield Label("", id="place-summary")
            with Horizontal(id="place-planes"):
                yield InteractivePlaneBox("xy", "X-Y Top", "X", "Y", (0.05, 0.50), (-0.25, 0.25), self._on_plane_pick, id="plane-xy")
                yield InteractivePlaneBox("xz", "X-Z Side", "X", "Z", (0.05, 0.50), (0.00, 0.25), self._on_plane_pick, id="plane-xz")
                yield InteractivePlaneBox("yz", "Y-Z Front", "Y", "Z", (-0.25, 0.25), (0.00, 0.25), self._on_plane_pick, id="plane-yz")
            with Horizontal(classes="coord-row"):
                yield Label("X (m):")
                yield Input(f"{self.tx:.3f}", id="in-px")
                yield Button("-0.03", id="btn-xm")
                yield Button("+0.03", id="btn-xp")
                yield Label("Y (m):")
                yield Input(f"{self.ty:.3f}", id="in-py")
                yield Button("-0.03", id="btn-ym")
                yield Button("+0.03", id="btn-yp")
            with Horizontal(classes="coord-row"):
                yield Label("Z (m):")
                yield Input(f"{self.tz:.3f}", id="in-pz")
                yield Button("-0.02", id="btn-zm")
                yield Button("+0.02", id="btn-zp")
            with Horizontal(id="place-actions"):
                yield Button("Move Arm Here (Pick -> Place)", variant="primary", id="btn-place-arm")
                yield Button("Set / Add at (X,Y,Z)", variant="success", id="btn-place-set")
                yield Button("Close", id="btn-place-close")

    def on_mount(self) -> None:
        self._redraw_planes()

    def _on_plane_pick(self, plane_id: str, h_val: float, v_val: float) -> None:
        if plane_id == "xy":
            self.tx, self.ty = h_val, v_val
        elif plane_id == "xz":
            self.tx, self.tz = h_val, v_val
        elif plane_id == "yz":
            self.ty, self.tz = h_val, v_val
        self._sync_inputs()

    def _redraw_planes(self) -> None:
        name = str(self.query_one("#place-obj-select", Select).value or "pen")
        base_spec = self.scene_objects.get(name) or self.presets.get(name) or {"color": "blue", "shape": "cube", "radius_m": 0.025}
        target_obj = {**base_spec, "x": self.tx, "y": self.ty, "z": self.tz}
        self.query_one("#place-summary", Label).update(f"Target: ({self.tx:.3f}, {self.ty:.3f}, {self.tz:.3f}) m")
        self.query_one("#plane-xy", InteractivePlaneBox).update_plot(name, target_obj, self.scene_objects)
        self.query_one("#plane-xz", InteractivePlaneBox).update_plot(name, target_obj, self.scene_objects)
        self.query_one("#plane-yz", InteractivePlaneBox).update_plot(name, target_obj, self.scene_objects)

    def _sync_inputs(self) -> None:
        self.query_one("#in-px", Input).value = f"{self.tx:.3f}"
        self.query_one("#in-py", Input).value = f"{self.ty:.3f}"
        self.query_one("#in-pz", Input).value = f"{self.tz:.3f}"
        self._redraw_planes()

    @on(Select.Changed, "#place-obj-select")
    def _on_obj_change(self, event: Select.Changed) -> None:
        name = str(event.value)
        pos = self.scene_objects.get(name) or self.presets.get(name)
        if pos:
            self.tx = float(pos.get("x", 0.25))
            self.ty = float(pos.get("y", 0.0))
            self.tz = float(pos.get("z", 0.0))
            self._sync_inputs()

    @on(Input.Submitted, "#in-px")
    @on(Input.Submitted, "#in-py")
    @on(Input.Submitted, "#in-pz")
    def _on_coord_submit(self) -> None:
        try:
            self.tx = max(0.05, min(0.50, float(self.query_one("#in-px", Input).value)))
            self.ty = max(-0.25, min(0.25, float(self.query_one("#in-py", Input).value)))
            self.tz = max(0.0, min(0.30, float(self.query_one("#in-pz", Input).value)))
        except ValueError:
            pass
        self._sync_inputs()

    @on(Button.Pressed, "#btn-xm")
    def _xm(self) -> None:
        self.tx = round(max(0.05, self.tx - 0.03), 3)
        self._sync_inputs()

    @on(Button.Pressed, "#btn-xp")
    def _xp(self) -> None:
        self.tx = round(min(0.50, self.tx + 0.03), 3)
        self._sync_inputs()

    @on(Button.Pressed, "#btn-ym")
    def _ym(self) -> None:
        self.ty = round(max(-0.25, self.ty - 0.03), 3)
        self._sync_inputs()

    @on(Button.Pressed, "#btn-yp")
    def _yp(self) -> None:
        self.ty = round(min(0.25, self.ty + 0.03), 3)
        self._sync_inputs()

    @on(Button.Pressed, "#btn-zm")
    def _zm(self) -> None:
        self.tz = round(max(0.0, self.tz - 0.02), 3)
        self._sync_inputs()

    @on(Button.Pressed, "#btn-zp")
    def _zp(self) -> None:
        self.tz = round(min(0.30, self.tz + 0.02), 3)
        self._sync_inputs()

    def _selected_payload(self, action: str) -> dict[str, Any]:
        self._on_coord_submit()
        name = str(self.query_one("#place-obj-select", Select).value or "pen")
        return {"action": action, "name": name, "x": self.tx, "y": self.ty, "z": self.tz}

    @on(Button.Pressed, "#btn-place-arm")
    def _place_arm(self) -> None:
        self.dismiss(self._selected_payload("move_arm"))

    @on(Button.Pressed, "#btn-place-set")
    def _place_set(self) -> None:
        self.dismiss(self._selected_payload("set_direct"))

    @on(Button.Pressed, "#btn-place-close")
    def action_close_modal(self) -> None:
        self.dismiss(None)


class RobotAgentTUI(App):
    TITLE = "DESK-CORE Voice & Vision Robot Agent"
    CSS = """
    Screen { layout: vertical; padding: 0 1; }
    #model-bar { height: auto; align: left middle; margin-bottom: 0; }
    #model-select { width: 1fr; margin-right: 1; }
    #voice-select { width: 22; margin-right: 1; }
    #output-mode { width: 18; }
    #action-bar { height: auto; align: left middle; margin-bottom: 0; }
    #action-bar Button { margin-right: 1; }
    #populate-bar { height: auto; align: left middle; margin-bottom: 0; }
    #populate-select { width: 32; margin-right: 1; }
    #populate-bar Button { margin-right: 1; }
    #key-status { height: auto; margin-bottom: 0; color: $text-muted; }
    #guide-panel { height: auto; border: solid $accent; background: $panel; padding: 0 1; margin-bottom: 1; }
    #camera-panel { height: 18; margin-bottom: 1; display: none; }
    .cam-box { width: 1fr; height: 18; border: solid $success; padding: 0 1; margin-right: 1; }
    .cam-img { width: 1fr; height: 15; }
    #chat-log { height: 1fr; border: solid $primary; padding: 0 1; margin-bottom: 1; }
    #input-bar { height: auto; align: left middle; }
    #cmd-input { width: 1fr; margin-right: 1; }
    #input-bar Button { margin-right: 1; }
    """

    POPULATE_PRESETS: dict[str, dict[str, Any]] = {
        "pen": {"x": 0.21, "y": -0.02, "z": 0.0, "color": "red", "shape": "pen", "radius_m": 0.022},
        "wallet": {"x": 0.34, "y": 0.09, "z": 0.0, "color": "brown", "shape": "wallet", "radius_m": 0.032},
        "airpods": {"x": 0.24, "y": 0.11, "z": 0.0, "color": "blue", "shape": "airpods", "radius_m": 0.024},
        "wired_earphones": {"x": 0.33, "y": -0.09, "z": 0.0, "color": "blue", "shape": "wired_earphones", "radius_m": 0.028},
        "mobile_phone": {"x": 0.19, "y": -0.11, "z": 0.0, "color": "blue", "shape": "mobile_phone", "radius_m": 0.034},
        "purple_sphere": {"x": 0.25, "y": -0.04, "z": 0.0, "color": "purple", "shape": "sphere", "radius_m": 0.024},
    }

    BINDINGS = [
        Binding("f1", "show_guide", "Guide"),
        Binding("ctrl+p", "open_placement", "Placement"),
        Binding("ctrl+k", "open_keys", "API Keys"),
        Binding("ctrl+v", "voice_command", "Voice"),
        Binding("ctrl+e", "trigger_estop", "E-STOP"),
        Binding("f12", "trigger_estop", "E-STOP"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, live: bool = False, keystore: SecureKeyStore | None = None) -> None:
        super().__init__()
        self.live = live
        self.keystore = keystore or SecureKeyStore()
        self.session: ClientSession | None = None
        self.tools_summary: str = ""
        self._mcp_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def compose(self) -> ComposeResult:
        mode = "LIVE" if self.live else "MOCK"
        options = self.keystore.available_dropdown_options()
        default_val = options[0][1] if options else "gemini:gemini-4-hydrogen"
        yield Header(show_clock=True)
        with Horizontal(id="model-bar"):
            yield Select(options, value=default_val, allow_blank=False, id="model-select")
            yield Select(
                [
                    ("Female (Ava)", "female"),
                    ("Male (Andrew)", "male"),
                    ("Female UK (Sonia)", "female_uk"),
                    ("Male UK (Ryan)", "male_uk"),
                ],
                value="female",
                allow_blank=False,
                id="voice-select",
            )
            yield Select(
                [("Audio + Text", "audio_text"), ("Audio only", "audio_only"), ("Text only", "text_only")],
                value="audio_text",
                allow_blank=False,
                id="output-mode",
            )
        with Horizontal(id="action-bar"):
            yield Button("Guide (F1)", id="btn-guide")
            yield Button("Placement (Ctrl+P)", variant="primary", id="btn-placement")
            yield Button("API Keys (Ctrl+K)", id="btn-keys")
            yield Button("E-STOP (Ctrl+E)", variant="error", id="btn-estop")
            yield Checkbox("Arm Motion", value=(not self.live), id="arm-motion")
            yield Checkbox("Show Camera Views", value=False, id="show-cams")
        if not self.live:
            with Horizontal(id="populate-bar"):
                yield Select(
                    [
                        ("Pen", "pen"),
                        ("Wallet", "wallet"),
                        ("AirPods", "airpods"),
                        ("Wired Earphones", "wired_earphones"),
                        ("Mobile Phone", "mobile_phone"),
                        ("Purple Sphere", "purple_sphere"),
                    ],
                    value="pen",
                    allow_blank=False,
                    id="populate-select",
                )
                yield Button("Add Object", variant="success", id="btn-add-obj")
                yield Button("Reset Scene", id="btn-reset-scene")
        yield Static(f"Mode: {mode} | Loading vision plugins...", id="key-status")
        yield Static(self._guide_markup(), id="guide-panel")
        with Horizontal(id="camera-panel"):
            with Vertical(classes="cam-box"):
                yield Label("[b]TOP VIEW[/b]")
                yield TerminalImage(classes="cam-img", id="cam-top-view")
            with Vertical(classes="cam-box"):
                yield Label("[b]GRIPPER VIEW (wrist)[/b]")
                yield TerminalImage(classes="cam-img", id="cam-wrist-view")
        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        with Horizontal(id="input-bar"):
            yield Input(placeholder="Type a robot command (e.g. 'find the keys on the desk')...", id="cmd-input")
            yield Button("Send", variant="primary", id="btn-send")
            yield Button("Voice (Ctrl+V)", variant="warning", id="btn-voice")
        yield Footer()

    def _refresh_status(self) -> None:
        mode = "LIVE" if self.live else "MOCK"
        st = self.keystore.status()
        badges = [
            f"{p.upper()}: {'OK ' + info['masked'] if info['configured'] else 'unset'}"
            for p, info in st.items()
        ]
        self.query_one("#key-status", Static).update(f"Mode: {mode} | " + " | ".join(badges))

        sel = self.query_one("#model-select", Select)
        cur = sel.value
        opts = self.keystore.available_dropdown_options()
        sel.set_options(opts)
        if any(v == cur for _, v in opts):
            sel.value = cur

    def _guide_markup(self) -> str:
        mode_hint = (
            "[cyan]MOCK 3D SCENE ACTIVE[/cyan] — synthetic top-view & gripper-view ('wrist') with known 3D object poses."
            if not self.live
            else "[green]LIVE HARDWARE MODE[/green] — connected to Station cameras & ST3215 arm."
        )
        return (
            "[bold]=== DESK-CORE Voice & Vision Agent Guide (F1 to toggle) ===[/bold]\n"
            f"{mode_hint}\n"
            "[b]Controls:[/b] Type + [b]Enter[/b] | [b]Ctrl+V[/b] Voice (4s) | [b]Ctrl+K[/b] API Keys | [b]Show Camera Views[/b] | [b]Ctrl+E / F12[/b] E-STOP\n"
            "[b]Try:[/b] [i]\"What objects do you see from both cameras?\"[/i] | [i]\"Show 3D positions of all objects\"[/i] | [i]\"Move red_cube closer to mug\"[/i]"
        )

    async def on_mount(self) -> None:
        self._refresh_status()
        self._cam_timer = self.set_interval(0.9, self._poll_cameras)
        self._mcp_task = asyncio.create_task(self._mcp_runner())

    def action_show_guide(self) -> None:
        panel = self.query_one("#guide-panel", Static)
        panel.display = not panel.display

    @on(Button.Pressed, "#btn-guide")
    def _on_guide_btn(self) -> None:
        self.action_show_guide()

    async def _refresh_camera_views(self) -> None:
        if self.session is None or not self.query_one("#show-cams", Checkbox).value:
            return
        if not hasattr(self, "_last_cam_bytes"):
            self._last_cam_bytes: dict[str, bytes] = {}
        try:
            r_top, r_wrist = await asyncio.gather(
                self.session.call_tool("get_frame", {"camera": "top", "denoise": False}),
                self.session.call_tool("get_frame", {"camera": "wrist", "denoise": False}),
            )
            for cam, wid, res in [("top", "#cam-top-view", r_top), ("wrist", "#cam-wrist-view", r_wrist)]:
                for block in res.content:
                    if getattr(block, "type", None) == "image":
                        jpg = base64.b64decode(block.data)
                        if self._last_cam_bytes.get(cam) != jpg:
                            self._last_cam_bytes[cam] = jpg
                            self.query_one(wid, TerminalImage).image = Image.open(io.BytesIO(jpg))
                        break
        except Exception:
            pass

    @on(Button.Pressed, "#btn-placement")
    async def _on_placement_btn(self) -> None:
        await self.action_open_placement()

    async def action_open_placement(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        if self.session is None:
            log.write("[yellow]MCP not connected yet.[/yellow]")
            return
        scene_objs: dict[str, dict[str, Any]] = {}
        try:
            st = await self.session.call_tool("mock_scene_state", {})
            texts = [b.text for b in st.content if getattr(b, "type", None) == "text"]
            if texts:
                parsed = json.loads(texts[0])
                if parsed.get("ok"):
                    scene_objs = parsed.get("objects", {})
        except Exception:
            pass

        async def _apply_placement(res: dict[str, Any] | None) -> None:
            if not res:
                return
            name, x, y, z = res["name"], res["x"], res["y"], res["z"]
            if res["action"] == "move_arm":
                cur = scene_objs.get(name)
                if cur is None:
                    spec = self.POPULATE_PRESETS.get(name, {"color": "blue", "shape": "cube", "radius_m": 0.025})
                    await self.session.call_tool("mock_add_object", {"name": name, "x": x, "y": y, "z": z, **spec})
                    log.write(f"[green]Placed '{name}' at ({x:.3f}, {y:.3f}, {z:.3f})[/green]")
                else:
                    pk = await self._call_mcp_tool("pick_at", {"x": cur["x"], "y": cur["y"], "z": cur["z"]})
                    pl = await self._call_mcp_tool("place_at", {"x": x, "y": y, "z": z})
                    log.write(f"[green]Moved '{name}' via arm -> ({x:.3f}, {y:.3f}, {z:.3f})[/green] (pick={pk.get('ok')}, place={pl.get('ok')})")
            else:
                spec = scene_objs.get(name) or self.POPULATE_PRESETS.get(name) or {"color": "blue", "shape": "cube", "radius_m": 0.025}
                await self.session.call_tool(
                    "mock_add_object",
                    {
                        "name": name,
                        "x": x,
                        "y": y,
                        "z": z,
                        "color": spec.get("color", "blue"),
                        "shape": spec.get("shape", "cube"),
                        "radius_m": float(spec.get("radius_m", 0.025)),
                    },
                )
                log.write(f"[green]Placed '{name}' at ({x:.3f}, {y:.3f}, {z:.3f})[/green]")
            await self._refresh_camera_views()

        self.push_screen(
            PlacementModal(scene_objs, self.POPULATE_PRESETS),
            lambda res: asyncio.create_task(_apply_placement(res)),
        )

    async def _poll_cameras(self) -> None:
        await self._refresh_camera_views()

    @on(Checkbox.Changed, "#show-cams")
    async def _on_toggle_cams(self, event: Checkbox.Changed) -> None:
        panel = self.query_one("#camera-panel", Horizontal)
        panel.display = bool(event.value)
        if event.value:
            await self._refresh_camera_views()

    @on(Button.Pressed, "#btn-add-obj")
    async def _on_add_obj(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        if self.session is None:
            log.write("[yellow]MCP not connected yet.[/yellow]")
            return
        name = str(self.query_one("#populate-select", Select).value or "pen")
        spec = self.POPULATE_PRESETS[name]
        res = await self.session.call_tool("mock_add_object", {"name": name, **spec})
        log.write(f"[green]Added 3D object '{name}' at ({spec['x']}, {spec['y']}, {spec['z']})[/green]")
        await self._refresh_camera_views()

    @on(Button.Pressed, "#btn-reset-scene")
    async def _on_reset_scene(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        if self.session is None:
            return
        await self.session.call_tool("mock_reset_scene", {})
        log.write("[cyan]Reset synthetic 3D scene to defaults.[/cyan]")
        await self._refresh_camera_views()

    async def _mcp_runner(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        err_fh = open(os.devnull, "w", encoding="utf-8")
        try:
            async with stdio_client(_server_params(self.live), errlog=err_fh) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    listed = await session.list_tools()
                    self.tools_summary = "\n".join(
                        f"- {t.name}: {(t.description or '').strip().splitlines()[0]}"
                        for t in listed.tools
                    )
                    log.write(f"[green]Connected to desk-core MCP ({len(listed.tools)} tools ready).[/green]")
                    await self._stop_event.wait()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.write(f"[bold red]MCP error:[/bold red] {exc}")
        finally:
            self.session = None
            err_fh.close()

    async def on_unmount(self) -> None:
        self._stop_event.set()
        if self._mcp_task is not None:
            try:
                await asyncio.wait_for(self._mcp_task, timeout=3.0)
            except Exception:
                self._mcp_task.cancel()

    def action_open_keys(self) -> None:
        self.push_screen(KeyConfigModal(self.keystore), lambda changed: self._refresh_status() if changed else None)

    @on(Button.Pressed, "#btn-keys")
    def _on_keys_btn(self) -> None:
        self.action_open_keys()

    @on(Button.Pressed, "#btn-estop")
    async def _on_estop_btn(self) -> None:
        await self.action_trigger_estop()

    async def action_trigger_estop(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        if self.session is None:
            log.write("[red]Not connected.[/red]")
            return
        res = await self.session.call_tool("estop", {})
        log.write(f"[bold red]E-STOP ENGAGED:[/bold red] {getattr(res, 'structuredContent', res.content)}")

    async def _fetch_images(self) -> list[tuple[str, bytes]]:
        if self.session is None:
            return []
        out: list[tuple[str, bytes]] = []
        for cam in ("top", "wrist"):
            res = await self.session.call_tool("get_frame", {"camera": cam, "denoise": True})
            for block in res.content:
                if getattr(block, "type", None) == "image":
                    out.append((cam, base64.b64decode(block.data)))
        return out

    async def _call_mcp_tool(self, name: str, args: dict[str, Any]) -> Any:
        log = self.query_one("#chat-log", RichLog)
        if self.session is None:
            return {"error": "MCP not connected"}
        armed = self.query_one("#arm-motion", Checkbox).value
        if name in MOTION_TOOLS and not armed:
            msg = f"Blocked `{name}`: check Arm Motion first."
            log.write(f"[yellow]{msg}[/yellow]")
            return {"ok": False, "error": msg}

        log.write(f"[cyan]→ {name}[/cyan]({json.dumps(args)})")
        res = await self.session.call_tool(name, args)
        await self._refresh_camera_views()
        if getattr(res, "structuredContent", None) is not None:
            return res.structuredContent
        texts = [b.text for b in res.content if getattr(b, "type", None) == "text"]
        return "\n".join(texts)

    @on(Input.Submitted, "#cmd-input")
    @on(Button.Pressed, "#btn-send")
    async def _on_send(self) -> None:
        inp = self.query_one("#cmd-input", Input)
        cmd = inp.value.strip()
        if not cmd:
            return
        inp.value = ""
        await self._execute_command(cmd)

    @on(Button.Pressed, "#btn-voice")
    async def _on_voice_btn(self) -> None:
        await self.action_voice_command()

    async def action_voice_command(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[yellow]Listening... speak your command (auto-stops on pause).[/yellow]")
        try:
            wav = await asyncio.to_thread(record_wav_sync, 6.0)
            log.write("[dim]Transcribing...[/dim]")
            transcript = await asyncio.to_thread(transcribe_wav_sync, wav, self.keystore)
            if not transcript:
                log.write("[yellow]No speech detected.[/yellow]")
                return
            log.write(f"[b]Voice:[/b] {transcript}")
            await self._execute_command(transcript)
        except Exception as exc:
            log.write(f"[red]Voice error:[/red] {exc}")

    async def _execute_command(self, cmd: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(f"\n[b]You:[/b] {cmd}")
        val = str(self.query_one("#model-select", Select).value or "gemini:gemini-4-hydrogen")
        provider, _, model_id = val.partition(":")
        try:
            out = await run_agent_command(
                command_text=cmd,
                provider=provider,
                model_id=model_id,
                keystore=self.keystore,
                fetch_images_fn=self._fetch_images,
                tools_summary=self.tools_summary,
                call_tool_fn=self._call_mcp_tool,
            )
            mode = str(self.query_one("#output-mode", Select).value or "audio_text")
            if mode in ("audio_text", "text_only"):
                log.write(f"[bold green]{provider}:{model_id}:[/bold green] {out['reply']}")
            else:
                log.write(f"[dim]{provider}:{model_id} (audio-only response)[/dim]")
            if mode in ("audio_text", "audio_only"):
                voice = str(self.query_one("#voice-select", Select).value or "female")
                await asyncio.to_thread(speak_text_sync, out["reply"], voice)
        except Exception as exc:
            log.write(f"[bold red]Agent error:[/bold red] {exc}")


async def _run_headless_test() -> int:
    import tempfile
    from unittest.mock import patch
    from desk_core.llm_plugins import AgentStepResponse, AgentToolCall, validate_vision_plugin

    # 1. Verify strict vision-only provider/model enforcement
    try:
        validate_vision_plugin("text-only-llm", "foo")
        raise AssertionError("Non-vision provider should be rejected")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as td:
        ks = SecureKeyStore(Path(td) / "keys.json")
        ks.set_key("gemini", "test-gem-key-1234")
        ks.set_key("claude", "test-claude-key-5678")
        ks.unset_key("claude")
        assert not ks.get_key("claude"), "Expected claude key to be unset"
        opts = {val for _lbl, val in ks.available_dropdown_options()}
        for expected in (
            "gemini:gemini-4-hydrogen",
            "claude:claude-fable-5-2",
            "openai:gpt-6.2-astra",
            "qwen:qwen4-max",
            "deepseek:deepseek-flash",
        ):
            assert expected in opts, f"Missing expected model {expected} in dropdown"
        mode_perm = (Path(td) / "keys.json").stat().st_mode & 0o777
        assert mode_perm == 0o600, f"Expected 0600 permissions, got {oct(mode_perm)}"

        app = RobotAgentTUI(live=False, keystore=ks)
        async with app.run_test() as pilot:
            for _ in range(30):
                if app.session is not None:
                    break
                await pilot.pause(0.1)
            assert app.session is not None, "MCP session failed to initialize"

            fake_step = AgentStepResponse(
                thought="Locate keys first.",
                tool_calls=[AgentToolCall(tool="get_calib_status", args={})],
                reply="Calibration status checked.",
                done=True,
            )
            with patch("desk_core.llm_plugins.call_vision_model_step", return_value=fake_step):
                app.query_one("#output-mode", Select).value = "audio_only"
                await app._execute_command("check calibration")
                app.query_one("#output-mode", Select).value = "audio_text"
                await app._execute_command("check calibration again")
            await pilot.pause(0.1)

    print("VOICE & VISION AGENT TUI PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DESK-CORE Voice & Text Robot Agent TUI")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Connect to live Station")
    mode.add_argument("--mock", action="store_true", help="Run offline mock robot")
    mode.add_argument("--test", action="store_true", help="Run headless self-test")
    args = ap.parse_args()

    if args.test:
        return asyncio.run(_run_headless_test())

    RobotAgentTUI(live=bool(args.live)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
