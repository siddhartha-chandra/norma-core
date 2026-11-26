import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import replace

import pytest

from desk_core.hive import Layout, Obstacle, TILE_THICKNESS_M, Tile, plan_hive
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from conftest import REPO


def test_piles_group_by_color_then_insect():
    layout = Layout.load()
    plan = plan_hive([
        Tile("b2", "black", "ant", 0.30, 0.025),
        Tile("c1", "cream", "bee", 0.20, -0.025),
        Tile("b1", "black", "Ant ", 0.20, 0.025),
        Tile("c2", "cream", "ant", 0.30, -0.025),
    ], layout)
    assert [(p.color, p.insect, p.tiles) for p in plan.piles] == [
        ("cream", "ant", ("c2",)), ("cream", "bee", ("c1",)), ("black", "ant", ("b1", "b2")),
    ]
    cream_ant, cream_bee, black_ant = plan.piles
    assert cream_ant.y == cream_bee.y == layout.cream_y
    assert black_ant.y == layout.black_y
    assert cream_bee.x - cream_ant.x == pytest.approx(layout.pile_spacing)
    assert [m.destination[2] for m in plan.moves if m.name.startswith("b")] == [0, TILE_THICKNESS_M]


def test_empty_single_color_and_unreadable():
    layout = Layout.load()
    assert plan_hive([], layout).moves == plan_hive([], layout).piles == ()
    tile = Tile("already stacked", "cream", "ant", layout.x_center, layout.cream_y)
    hidden = Tile("unreadable", "cream", None, 0.20, 0.0)
    plan = plan_hive([tile, hidden], layout)
    assert not plan.moves
    assert plan.piles[0].tiles == (tile.name,)
    assert plan.skipped == (hidden.name,)


def test_moves_obstruction_before_picking_the_bottom_tile():
    layout = Layout.load()
    cream = Tile("bottom", "cream", "ant", 0.28, 0.0)
    black = Tile("top", "black", "bee", cream.x, cream.y, TILE_THICKNESS_M)
    plan = plan_hive([cream, black], layout)
    assert [move.name for move in plan.moves] == ["top", "bottom"]


def test_stages_a_tile_when_two_piles_are_blocking_each_other():
    layout = Layout.load()
    cream = Tile("cream", "cream", "ant", layout.x_center, layout.black_y)
    black = Tile("black", "black", "ant", layout.x_center, layout.cream_y)
    plan = plan_hive([cream, black], layout)
    first, second, third = plan.moves
    assert first.name == third.name != second.name
    assert first.destination == third.source
    assert second.destination == first.source
    assert third.destination == second.source


def test_refuses_collisions_and_out_of_workspace_piles():
    layout = Layout.load()
    tile = Tile("ant", "cream", "ant", 0.20, 0)
    with pytest.raises(ValueError, match="out-of-workspace"):
        plan_hive([replace(tile, x=layout.x_min)], layout)
    with pytest.raises(ValueError, match="not enough workspace"):
        plan_hive([tile], replace(layout, cream_y=layout.y_min))
    with pytest.raises(ValueError, match="blocked by a tabletop obstacle"):
        plan_hive([tile], layout, (Obstacle(layout.x_center, layout.cream_y, 0.01),))
    with pytest.raises(ValueError, match="overlapping Hive tiles"):
        plan_hive([tile, replace(tile, name="bee", insect="bee")], layout)
    with pytest.raises(ValueError, match="at most 12"):
        plan_hive([tile] * 13, layout)


def test_hive_sort_refuses_to_carry_a_non_hive_object(mock_bridge, monkeypatch):
    from desk_core import mcp_server

    monkeypatch.setattr(mcp_server, "_bridge", mock_bridge)
    mock_bridge.seed_mock_hive(42)
    tile = mock_bridge.get_mock_scene_state()["objects"]["cream-ant-1"]
    mock_bridge.add_mock_object("rider", tile["x"], tile["y"], TILE_THICKNESS_M, radius_m=0.005)
    before = mock_bridge.get_mock_scene_state()
    result = asyncio.run(mcp_server.hive_sort(execute=True))
    assert result == {"ok": False, "error": "Hive tile 'cream-ant-1' is blocked by 'rider'"}
    assert mock_bridge.get_mock_scene_state() == before


@pytest.mark.parametrize("seed", [42, 73])
def test_hive_sorts_a_randomized_station_mock_scene(seed):
    async def run():
        env = {**os.environ, "DESK_MOCK": "1", "DESK_NO_DOTENV": "1", "DESK_RUNLOG": "0",
               "DESK_MOCK_SYNTHETIC": "1", "DESK_MOCK_HIVE_SEED": str(seed)}
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "desk_core.mcp_server"], cwd=str(REPO), env=env,
        )
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()

                async def call(tool, **args):
                    response = await client.call_tool(tool, args)
                    assert not response.is_error, response
                    return response.structured_content

                before = await call("mock_scene_state")
                preview = await call("hive_sort")
                assert preview["ok"] and not preview["executed"] and preview["moves"]
                assert await call("mock_scene_state") == before
                command = await call("plan_command", command="sort Hive tiles")
                result = await call("execute_sequence", steps=command["steps"])
                assert result["ok"], result
                sorted_scene = await call("mock_scene_state")
                assert sorted_scene["held_object"] is None
                assert sorted_scene["objects"].keys() == before["objects"].keys()
                piles = defaultdict(list)
                for obj in sorted_scene["objects"].values():
                    piles[obj["color"], obj["insect"]].append((obj["x"], obj["y"], obj["z"]))
                assert len(piles) == 6
                layout = Layout.load()
                pile_centers = set()
                for (color, _), layers in piles.items():
                    (x0, y0, z0), (x1, y1, z1) = sorted(layers, key=lambda layer: layer[2])
                    assert (x0, y0) == (x1, y1)
                    assert y0 == (layout.cream_y if color == "cream" else layout.black_y)
                    assert (z0, z1) == (0.0, TILE_THICKNESS_M)
                    pile_centers.add((x0, y0))
                assert len(pile_centers) == len(piles)
                assert not (await call("hive_sort"))["moves"]

    asyncio.run(run())
