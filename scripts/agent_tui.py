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
import json
import os
import sys
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
        width: 68;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #key-dialog Input {
        margin-bottom: 1;
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

    def compose(self) -> ComposeResult:
        st = self.keystore.status()
        with Vertical(id="key-dialog"):
            yield Label("[b]Secure Vision LLM Plugins (0600 local keystore)[/b]")
            yield Label(f"[dim]Only image-capable providers allowed. Stored at: {self.keystore.path}[/dim]")
            yield Label(f"Google Gemini ({st['gemini']['masked'] or 'not set'}):")
            yield Input(placeholder="Paste GEMINI_API_KEY (leave blank to keep)", password=True, id="in-gemini")
            yield Label(f"Anthropic Claude ({st['claude']['masked'] or 'not set'}):")
            yield Input(placeholder="Paste ANTHROPIC_API_KEY (leave blank to keep)", password=True, id="in-claude")
            yield Label(f"OpenAI ({st['openai']['masked'] or 'not set'}):")
            yield Input(placeholder="Paste OPENAI_API_KEY (leave blank to keep)", password=True, id="in-openai")
            with Horizontal(id="key-actions"):
                yield Button("Save Keys", variant="primary", id="btn-save-keys")
                yield Button("Cancel", id="btn-cancel-keys")

    @on(Button.Pressed, "#btn-save-keys")
    def _save(self) -> None:
        for prov, wid in [("gemini", "#in-gemini"), ("claude", "#in-claude"), ("openai", "#in-openai")]:
            val = self.query_one(wid, Input).value.strip()
            if val:
                self.keystore.set_key(prov, val)
        self.dismiss(True)

    @on(Button.Pressed, "#btn-cancel-keys")
    def _cancel(self) -> None:
        self.dismiss(False)


class RobotAgentTUI(App):
    TITLE = "DESK-CORE Voice & Vision Robot Agent"
    CSS = """
    Screen { layout: vertical; padding: 0 1; }
    #top-bar { height: 3; align: left middle; margin-bottom: 1; }
    #model-select { width: 48; margin-right: 1; }
    #top-bar Button { margin-right: 1; }
    #key-status { height: auto; margin-bottom: 1; color: $text-muted; }
    #chat-log { height: 1fr; border: solid $primary; padding: 0 1; margin-bottom: 1; }
    #input-bar { height: 3; align: left middle; }
    #cmd-input { width: 1fr; margin-right: 1; }
    #input-bar Button { margin-right: 1; }
    """

    BINDINGS = [
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
        with Horizontal(id="top-bar"):
            yield Select(options, value=default_val, allow_blank=False, id="model-select")
            yield Button("API Keys (Ctrl+K)", id="btn-keys")
            yield Button("E-STOP (Ctrl+E)", variant="error", id="btn-estop")
            yield Checkbox("Arm Motion", value=(not self.live), id="arm-motion")
        yield Static(f"Mode: {mode} | Loading vision plugins...", id="key-status")
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

    async def on_mount(self) -> None:
        self._refresh_status()
        self._mcp_task = asyncio.create_task(self._mcp_runner())

    async def _mcp_runner(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        try:
            async with stdio_client(_server_params(self.live)) as (read, write):
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
        res = await self.session.call_tool("get_frame", {"camera": "top", "denoise": True})
        for block in res.content:
            if getattr(block, "type", None) == "image":
                out.append(("top", base64.b64decode(block.data)))
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
        log.write("[yellow]Listening (4s)... speak your command.[/yellow]")
        try:
            wav = await asyncio.to_thread(record_wav_sync, 4.0)
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
            log.write(f"[bold green]{provider}:{model_id}:[/bold green] {out['reply']}")
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
                await app._execute_command("check calibration")
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
