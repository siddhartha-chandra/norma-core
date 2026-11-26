#!/usr/bin/env python3
"""Interactive Textual TUI to test every DESK-CORE MCP tool over real stdio MCP.

Usage:
  uv run --offline --managed-python python scripts/mcp_tui.py --mock   # offline mock rig
  uv run --offline --managed-python python scripts/mcp_tui.py --live   # live rig (.env STATION_HOST)
  uv run --offline --managed-python python scripts/mcp_tui.py --test   # headless TUI self-test
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from textual import on
from textual.app import App, ComposeResult
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
    ListItem,
    ListView,
    RichLog,
    Select,
    Static,
    TextArea,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SAFETY_GROUPS: dict[str, tuple[str, str]] = {
    "observe": ("READ", "green"),
    "get_state": ("READ", "green"),
    "get_joint_angles": ("READ", "green"),
    "get_frame": ("READ", "green"),
    "get_calib_status": ("READ", "green"),
    "reload_config": ("READ", "green"),
    "localize_3d": ("VISION", "cyan"),
    "robo_jev_arm_watch": ("SAFETY", "yellow"),
    "robo_jev_check": ("SAFETY", "yellow"),
    "robo_jev_guard": ("SAFETY", "yellow"),
    "speak_status": ("AUDIO", "cyan"),
    "pixel_to_world": ("GEOM", "cyan"),
    "get_metric_depth": ("GEOM", "cyan"),
    "solve_ik": ("PLAN", "cyan"),
    "update_scene_obstacle": ("PLAN", "cyan"),
    "estop": ("SAFETY", "bold red"),
    "set_torque": ("SAFETY", "yellow"),
    "grasp": ("GRIP", "yellow"),
    "release": ("GRIP", "yellow"),
    "move_joints": ("MOTION", "red"),
    "move_joint": ("MOTION", "red"),
    "go_home": ("MOTION", "red"),
    "move_to_pose": ("MOTION", "red"),
    "pick_at": ("MOTION", "red"),
    "place_at": ("MOTION", "red"),
    "pick_pixel": ("MOTION", "red"),
    "sweep_aside": ("MOTION", "red"),
    "execute_sequence": ("MOTION", "red"),
}

DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    "observe": {"views": ["top", "wrist"], "denoise": True},
    "get_state": {},
    "get_joint_angles": {"include_gripper": False},
    "get_frame": {"camera": "top", "denoise": True, "as_matrix": False, "max_dim": 0},
    "get_calib_status": {},
    "reload_config": {},
    "localize_3d": {"query": "keys", "camera": "top", "use_gemini": True, "timeout_s": 6.0},
    "robo_jev_arm_watch": {"x": 640.0, "y": 360.0, "query": "keys", "camera": "top"},
    "robo_jev_check": {"camera": "top", "abort_on_hazard": True},
    "robo_jev_guard": {"query": "keys", "camera": "top", "obstacle_wait_timeout_s": 5.0, "use_gemini": False},
    "speak_status": {"text": "Desk core standing by."},
    "pixel_to_world": {"x": 640.0, "y": 360.0, "camera": "top"},
    "get_metric_depth": {"x": 640.0, "y": 360.0, "camera": "top"},
    "solve_ik": {"x": 0.25, "y": 0.0, "z": 0.10, "qx": 0.0, "qy": 0.564687767, "qz": 0.0, "qw": 0.825335614},
    "update_scene_obstacle": {
        "name": "mug",
        "x": 0.25,
        "y": 0.0,
        "z": 0.05,
        "shape": "cylinder",
        "radius_m": 0.045,
        "height_m": 0.10,
        "remove": False,
    },
    "estop": {},
    "set_torque": {"enable": True},
    "grasp": {},
    "release": {},
    "move_joints": {"targets": {"1": 2048}, "max_step": 150, "wait": True},
    "move_joint": {"joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "wait": True},
    "go_home": {},
    "move_to_pose": {"x": 0.25, "y": 0.0, "z": 0.12, "qx": 0.0, "qy": 0.564687767, "qz": 0.0, "qw": 0.825335614},
    "pick_at": {"x": 0.25, "y": 0.0, "z": 0.05, "qx": 0.0, "qy": 0.707106781, "qz": 0.0, "qw": 0.707106781, "approach_dz": 0.06},
    "place_at": {"x": 0.25, "y": 0.12, "z": 0.05, "qx": 0.0, "qy": 0.707106781, "qz": 0.0, "qw": 0.707106781, "retract_dz": 0.06},
    "pick_pixel": {"x": 640.0, "y": 360.0, "camera": "top", "qx": 0.0, "qy": 0.707106781, "qz": 0.0, "qw": 0.707106781},
    "execute_sequence": {
        "steps": [
            {"tool": "get_calib_status", "args": {}, "save_as": "cal"},
            {"tool": "localize_3d", "args": {"query": "keys", "camera": "top"}, "save_as": "obj"},
        ],
        "stop_on_false": True,
    },
}


LIVE_SAFETY_NOTES: dict[str, str] = {
    "observe": "Safe (read-only): no motor motion. Run first to verify link & camera feeds.",
    "get_state": "Safe (read-only): reads live ST3215 ticks/currents. Check M1 position before move_joints.",
    "get_joint_angles": "Safe (read-only): reads current [j1..j7] rad. Copy these before testing move_joint.",
    "get_frame": "Safe (read-only): fetches 1 camera frame and saves JPEG/npy locally.",
    "get_calib_status": "Safe (read-only): inspects calibration JSON, missing fields & IK backend.",
    "reload_config": "Safe (read-only): reloads calibration JSON & IK module without moving.",
    "localize_3d": "Safe (vision only): runs Gemini 4 Hydrogen or OpenCV fallback; no motor motion.",
    "robo_jev_arm_watch": "Safe (vision only): arms <150 ms shift tracking on pixel (x, y).",
    "robo_jev_check": "Safe (reflex check): checks target_shifted/path_clear; calls estop() on hazard.",
    "robo_jev_guard": "Safe (pause/recover): waits for obstacle to clear; recalibrates shift if in safe bounds.",
    "speak_status": "Safe (audio only): speaks status cue via portable speaker TTS.",
    "pixel_to_world": "Safe (math only): projects pixel to 3D (nominal height until #4/#6 extrinsics land).",
    "get_metric_depth": "Safe (math only): estimates depth (nominal height until #4/#6 extrinsics land).",
    "solve_ik": "Safe (math only): solves [j1..j7] without moving (fallback IK until #5/#7 land).",
    "update_scene_obstacle": "Safe (memory only): registers/removes collision obstacle; no motor motion.",
    "estop": "Safe (emergency hold): immediately freezes all motors at current ticks.",
    "set_torque": "Caution: enable=false drops holding torque — support the arm by hand first!",
    "grasp": "Low risk (jaws only): arm stays still; keep fingers/cables clear of gripper jaws.",
    "release": "Low risk (jaws only): opens gripper jaws; arm stays still.",
    "move_joints": "Moves arm: check get_state first, then nudge 1 joint (e.g. '1') +-100..200 ticks (max_step<=200).",
    "move_joint": "Moves arm: start from current get_joint_angles and perturb 1 joint by +-0.1 rad (not [0]*7!).",
    "go_home": "Moves arm: moves all 7 arm joints toward midpoints — clear workspace around base first.",
    "move_to_pose": "Wait for #4/#6 extrinsics + #5/#7 IK before live run (currently fallback IK).",
    "pick_at": "Wait for #4/#6 extrinsics + #5/#7 IK before live run (full descent + grasp).",
    "place_at": "Wait for #4/#6 extrinsics + #5/#7 IK before live run (full descent + release).",
    "pick_pixel": "Wait for #4/#6 extrinsics + #5/#7 IK before live run (vision-to-grasp).",
    "execute_sequence": "Safe with default read-only steps; keep workspace clear if adding motion steps.",
}


def _server_params(live: bool) -> StdioServerParameters:
    env = dict(os.environ)
    if live:
        env.pop("DESK_MOCK", None)
        env.pop("DESK_NO_DOTENV", None)
    else:
        env["DESK_MOCK"] = "1"
        env["DESK_NO_DOTENV"] = "1"
        env["STATION_HOST"] = ""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "desk_core.mcp_server"],
        env=env,
        cwd=str(REPO_ROOT),
    )


class ToolItem(ListItem):
    def __init__(self, tool_name: str, badge: str, color: str) -> None:
        super().__init__(Label(f"[{color}][{badge:6s}][/{color}] {tool_name}"))
        self.tool_name = tool_name


KEY_PROVIDERS = ["gemini", "anthropic", "openai", "qwen", "deepseek"]


def _keystore_path() -> Path:
    env_p = os.environ.get("DESK_KEYS_PATH", "").strip()
    return Path(env_p) if env_p else (Path.home() / ".config" / "desk-core" / "keys.json")


def _read_keystore() -> dict[str, str]:
    p = _keystore_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if str(v).strip()}
    except Exception:
        return {}


def _write_keystore(data: dict[str, str]) -> Path:
    p = _keystore_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    payload = json.dumps(data, indent=2) + "\n"
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return p


class KeysModal(ModalScreen[None]):
    CSS = """
    KeysModal {
        align: center middle;
    }
    #keys-dialog {
        width: 64;
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    #keys-btns {
        height: 3;
        margin-top: 1;
    }
    #keys-btns Button {
        margin-right: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="keys-dialog"):
            yield Label("[b]Secure Vision API Key Store (0600)[/b]")
            yield Static("", id="keys-status")
            yield Select([(p.title(), p) for p in KEY_PROVIDERS], value="gemini", id="key-provider")
            yield Input(placeholder="Paste API key (masked)...", password=True, id="key-input")
            with Horizontal(id="keys-btns"):
                yield Button("Save (0600)", id="btn-key-save", variant="primary")
                yield Button("Unset", id="btn-key-unset", variant="warning")
                yield Button("Close", id="btn-key-close")

    def on_mount(self) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        store = _read_keystore()
        parts = [f"{p}:{'SET' if p in store else 'unset'}" for p in KEY_PROVIDERS]
        self.query_one("#keys-status", Static).update("Stored: " + " | ".join(parts))

    @on(Button.Pressed, "#btn-key-save")
    def on_save(self) -> None:
        prov = str(self.query_one("#key-provider", Select).value or "gemini")
        val = self.query_one("#key-input", Input).value.strip()
        if not val:
            return
        store = _read_keystore()
        store[prov] = val
        _write_keystore(store)
        self.query_one("#key-input", Input).value = ""
        self._refresh_status()

    @on(Button.Pressed, "#btn-key-unset")
    def on_unset(self) -> None:
        prov = str(self.query_one("#key-provider", Select).value or "gemini")
        store = _read_keystore()
        store.pop(prov, None)
        _write_keystore(store)
        self._refresh_status()

    @on(Button.Pressed, "#btn-key-close")
    def on_close(self) -> None:
        self.dismiss(None)


class DeskCoreTUI(App):
    TITLE = "DESK-CORE MCP Tester"
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    #sidebar {
        width: 34;
        border: solid $accent;
        padding: 0 1;
    }
    #workspace {
        width: 1fr;
        padding: 0 1;
    }
    #tool-info {
        height: auto;
        max-height: 8;
        border: solid $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    #args-editor {
        height: 10;
        border: solid $secondary;
        margin-bottom: 1;
    }
    #controls {
        height: 3;
        align: left middle;
        margin-bottom: 1;
    }
    #controls Button {
        margin-right: 1;
    }
    #arm-motion {
        margin-left: 1;
    }
    #output-log {
        height: 1fr;
        border: solid $surface-lighten-2;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+r", "run_selected", "Run Tool"),
        Binding("ctrl+k", "open_keys", "API Keys"),
        Binding("ctrl+e", "trigger_estop", "E-STOP"),
        Binding("f12", "trigger_estop", "E-STOP"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, live: bool = False) -> None:
        super().__init__()
        self.live = live
        self._mcp_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self.session: ClientSession | None = None
        self.tools_by_name: dict[str, Any] = {}
        self.selected_tool: str = "observe"

    def compose(self) -> ComposeResult:
        mode = "LIVE" if self.live else "MOCK"
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label(f"[b]MCP Tools ({mode})[/b]")
                yield ListView(id="tool-list")
            with Vertical(id="workspace"):
                yield Static("Connecting to desk-core MCP server...", id="tool-info")
                yield Label("Arguments (JSON):")
                yield TextArea("{}", language="json", id="args-editor")
                with Horizontal(id="controls"):
                    yield Button("Run Tool (Ctrl+R)", id="btn-run", variant="primary")
                    yield Button("API Keys (Ctrl+K)", id="btn-keys", variant="default")
                    yield Button("E-STOP (Ctrl+E)", id="btn-estop", variant="error")
                    yield Checkbox("Arm Motion", value=(not self.live), id="arm-motion")
                yield RichLog(id="output-log", highlight=True, markup=True, wrap=True)
        yield Footer()

    async def on_mount(self) -> None:
        log = self.query_one("#output-log", RichLog)
        mode = "LIVE" if self.live else "MOCK"
        log.write(f"[dim]Starting desk-core MCP server ({mode} mode)...[/dim]")
        self._mcp_task = asyncio.create_task(self._mcp_runner())

    async def _mcp_runner(self) -> None:
        log = self.query_one("#output-log", RichLog)
        devnull = open(os.devnull, "w", encoding="utf-8")
        try:
            async with stdio_client(_server_params(self.live), errlog=devnull) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    listed = await session.list_tools()
                    self.tools_by_name = {t.name: t for t in listed.tools}

                    order = list(SAFETY_GROUPS.keys())
                    names = sorted(
                        self.tools_by_name.keys(),
                        key=lambda n: (order.index(n) if n in order else 999, n),
                    )
                    lv = self.query_one("#tool-list", ListView)
                    for name in names:
                        badge, color = SAFETY_GROUPS.get(name, ("TOOL", "white"))
                        await lv.append(ToolItem(name, badge, color))

                    log.write(f"[green]Connected! {len(names)} MCP tools ready.[/green]")
                    if names:
                        lv.index = 0
                        self._select_tool(names[0])

                    await self._stop_event.wait()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.write(f"[bold red]MCP connect failed:[/bold red] {exc}")
        finally:
            self.session = None
            devnull.close()

    async def on_unmount(self) -> None:
        self._stop_event.set()
        if self._mcp_task is not None:
            try:
                await asyncio.wait_for(self._mcp_task, timeout=3.0)
            except Exception:
                self._mcp_task.cancel()

    def _select_tool(self, name: str) -> None:
        self.selected_tool = name
        tool = self.tools_by_name.get(name)
        badge, color = SAFETY_GROUPS.get(name, ("TOOL", "white"))
        desc = (tool.description or "").strip().splitlines()[0] if tool else ""
        schema = (getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}) if tool else {}
        props = list((schema.get("properties") or {}).keys())
        safety_note = LIVE_SAFETY_NOTES.get(name, "Check parameters and clear workspace before running.")
        info = self.query_one("#tool-info", Static)
        info.update(
            f"[b]{name}[/b] [{color}][{badge}][/{color}]\n"
            f"{desc}\n"
            f"[yellow]Live Tip:[/yellow] {safety_note}\n"
            f"[dim]Params: {', '.join(props) if props else '(none)'}[/dim]"
        )
        default_json = json.dumps(DEFAULT_ARGS.get(name, {}), indent=2)
        self.query_one("#args-editor", TextArea).load_text(default_json)

    @on(ListView.Selected, "#tool-list")
    def _on_list_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ToolItem):
            self._select_tool(event.item.tool_name)

    @on(ListView.Highlighted, "#tool-list")
    def _on_list_highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, ToolItem):
            self._select_tool(event.item.tool_name)

    @on(Button.Pressed, "#btn-run")
    async def _on_run_btn(self) -> None:
        await self.action_run_selected()

    @on(Button.Pressed, "#btn-estop")
    async def _on_estop_btn(self) -> None:
        await self.action_trigger_estop()

    @on(Button.Pressed, "#btn-keys")
    def _on_keys_btn(self) -> None:
        self.action_open_keys()

    def action_open_keys(self) -> None:
        self.push_screen(KeysModal())

    async def action_trigger_estop(self) -> None:
        await self._call_tool("estop", {})

    async def action_run_selected(self) -> None:
        name = self.selected_tool
        badge, _ = SAFETY_GROUPS.get(name, ("TOOL", "white"))
        armed = self.query_one("#arm-motion", Checkbox).value
        log = self.query_one("#output-log", RichLog)

        raw = self.query_one("#args-editor", TextArea).text.strip() or "{}"
        try:
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError("Arguments must be a JSON object")
        except Exception as exc:
            log.write(f"[bold red]Invalid JSON arguments:[/bold red] {exc}")
            return

        requires_arm = badge == "MOTION" and (name != "sweep_aside" or args.get("execute") is True)
        if requires_arm and not armed:
            log.write(
                f"[bold yellow]Blocked {name}:[/bold yellow] check [b]Arm Motion[/b] first "
                "(safety interlock for physical arm movement)."
            )
            return

        await self._call_tool(name, args)

    async def _call_tool(self, name: str, args: dict[str, Any]) -> None:
        log = self.query_one("#output-log", RichLog)
        if self.session is None:
            log.write("[red]Not connected to MCP server.[/red]")
            return

        log.write(f"\n[bold cyan]>>> {name}[/bold cyan]({json.dumps(args)})")
        t0 = time.perf_counter()
        try:
            res = await self.session.call_tool(name, args)
        except Exception as exc:
            log.write(f"[bold red]Call error ({(time.perf_counter() - t0) * 1000:.0f} ms):[/bold red] {exc}")
            return

        dt_ms = (time.perf_counter() - t0) * 1000.0
        img_idx = 0
        out_dir = REPO_ROOT / "runs" / "tui_frames"

        if getattr(res, "structuredContent", None) is not None:
            sc = res.structuredContent
            payload = sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
            log.write(f"[green]{dt_ms:.0f} ms[/green] {json.dumps(payload, indent=2)}")
            return

        for block in res.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                txt = block.text.strip()
                try:
                    parsed = json.dumps(json.loads(txt), indent=2)
                    log.write(f"[green]{dt_ms:.0f} ms[/green] {parsed}")
                except ValueError:
                    log.write(f"[green]{dt_ms:.0f} ms[/green] {txt}")
            elif btype == "image":
                out_dir.mkdir(parents=True, exist_ok=True)
                ext = "jpg" if "jpeg" in (getattr(block, "mimeType", "") or "") else "png"
                img_path = out_dir / f"{name}_{img_idx}.{ext}"
                img_path.write_bytes(base64.b64decode(block.data))
                log.write(f"[magenta]Saved image[{img_idx}]:[/magenta] {img_path}")
                img_idx += 1


async def _run_headless_test() -> int:
    app = DeskCoreTUI(live=False)
    async with app.run_test() as pilot:
        # Wait for MCP stdio server initialization
        for _ in range(30):
            if app.session is not None and app.tools_by_name:
                break
            await pilot.pause(0.1)
        assert app.session is not None, "MCP session failed to initialize in TUI"
        assert len(app.tools_by_name) >= 22, f"Expected >=22 tools, got {len(app.tools_by_name)}"

        # Test a safe READ tool and a VISION tool
        app._select_tool("get_calib_status")
        await app.action_run_selected()
        app._select_tool("localize_3d")
        await app.action_run_selected()
        await pilot.pause(0.2)
    print(f"TUI SELF-TEST PASS ({len(app.tools_by_name)} tools loaded & executed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive Textual TUI for DESK-CORE MCP tools")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Connect to live Station (.env STATION_HOST)")
    mode.add_argument("--mock", action="store_true", help="Run offline mock robot")
    mode.add_argument("--test", action="store_true", help="Run headless TUI self-test and exit")
    args = ap.parse_args()

    if args.test:
        return asyncio.run(_run_headless_test())

    DeskCoreTUI(live=bool(args.live)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
