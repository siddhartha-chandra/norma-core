"""MCP desk memory verification against the internal Station simulation."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]


def test_mcp_desk_memory_uses_persisted_station_mock_sightings(tmp_path):
    env = {
        **os.environ,
        "DESK_MOCK": "1",
        "DESK_MOCK_SYNTHETIC": "1",
        "DESK_MOCK_HEADING_OUT": "0",
        "DESK_NO_DOTENV": "1",
        "DESK_RUNLOG": "0",
        "DESK_SIGHTINGS_NORMFS": "1",
        "DESK_SIGHTINGS_DB": str(tmp_path / "sightings.db"),
    }
    env.pop("DESK_MOCK_HIVE_SEED", None)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "desk_core.mcp_server"], cwd=str(REPO), env=env,
    )

    async def call(session, tool, **args):
        result = await session.call_tool(tool, args)
        assert not result.is_error, result
        return result.structured_content

    async def run():
        query = {"query": "keys", "camera": "top", "provider": "opencv", "denoise": False}
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                first = await call(session, "localize_3d", **query)
                moved = await call(session, "mock_add_object", name="keys", x=0.36, y=-0.04, color="blue")
                assert moved["ok"]
                second = await call(session, "localize_3d", **query)
                # The newest frame is an unrelated red object.
                other = await call(session, "localize_3d", **{**query, "query": "red cube"})
                assert other["found"] and other["label"] != first["label"]
                for observed in (first, second):
                    assert observed["found"] and observed["sighting"]["backend"] == "sqlite"
                    print("Station mock recorded:", json.dumps(observed["sighting"], sort_keys=True))
                assert first["label"] == second["label"]
                assert first["world"] != second["world"]

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                assert {"last_seen", "where_was"} <= tools
                label = first["label"]
                start, end = first["sighting"]["time"], second["sighting"]["time"]
                assert start < end < other["sighting"]["time"]

                for tool, args, expected in (
                    ("last_seen", {"label": label}, second),
                    ("where_was", {"label": label, "time": (start + end) / 2}, first),
                    ("where_was", {"label": label, "time": start}, first),
                ):
                    result = await call(session, tool, **args)
                    print(f"{tool}({json.dumps(args)}) -> {json.dumps(result, sort_keys=True)}")
                    assert result["found"] is True
                    for key in ("label", "x", "y", "time", "frame_ref"):
                        assert result[key] == expected["sighting"][key]
                    assert result["z"] == expected["world"]["z"]

                assert await call(session, "where_was", label=label, time=start - 1) == {"found": False, "label": label}
                assert await call(session, "last_seen", label="unseen") == {"found": False, "label": "unseen"}

    asyncio.run(run())
