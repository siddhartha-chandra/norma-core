"""Pure planning for two rows of Hive tile piles."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

COLORS = ("cream", "black")
TILE_RADIUS_M = 0.022
TILE_THICKNESS_M = 0.012
MAX_TILES = 12
POSITION_TOL_M = 0.0006
PILE_CLEARANCE_M = 0.006
HIVE_SHAPE = "hive_tile"


@dataclass(frozen=True)
class Tile:
    name: str
    color: str
    insect: str | None
    x: float
    y: float
    z: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return self.color, (self.insect or "").strip().casefold()


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius_m: float


@dataclass(frozen=True)
class Layout:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_center: float
    pile_spacing: float
    cream_y: float
    black_y: float

    @classmethod
    def load(cls) -> Layout:
        path = Path(__file__).resolve().parent.parent / "calib" / "hive_layout.json"
        return cls(**json.loads(path.read_text()))

    def fits(self, x: float, y: float) -> bool:
        return (
            self.x_min + TILE_RADIUS_M <= x <= self.x_max - TILE_RADIUS_M
            and self.y_min + TILE_RADIUS_M <= y <= self.y_max - TILE_RADIUS_M
        )


@dataclass(frozen=True)
class Move:
    name: str
    source: tuple[float, float, float]
    destination: tuple[float, float, float]


@dataclass(frozen=True)
class Pile:
    color: str
    insect: str
    x: float
    y: float
    tiles: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    moves: tuple[Move, ...]
    piles: tuple[Pile, ...]
    skipped: tuple[str, ...]


def _distance(obj: Tile | Obstacle, x: float, y: float) -> float:
    return math.hypot(obj.x - x, obj.y - y)


def _at(obj: Tile, x: float, y: float) -> bool:
    return _distance(obj, x, y) <= POSITION_TOL_M


def plan_hive(tiles: list[Tile], layout: Layout, obstacles: tuple[Obstacle, ...] = ()) -> Plan:
    """Build all moves before execution. Z is the height of the tile base above the table."""
    if len(tiles) > MAX_TILES:
        raise ValueError(f"at most {MAX_TILES} Hive tiles can be sorted")
    if len({t.name for t in tiles}) != len(tiles):
        raise ValueError("Hive tile names must be unique")
    clearance = 2 * TILE_RADIUS_M + PILE_CLEARANCE_M
    if layout.pile_spacing < clearance or abs(layout.cream_y - layout.black_y) < clearance:
        raise ValueError("Hive pile spacing does not leave clearance between tiles")

    positions = {t.name: t for t in sorted(tiles, key=lambda t: (t.z, t.name))}
    for name, tile in positions.items():
        if not name or tile.color not in COLORS or not layout.fits(tile.x, tile.y) or not math.isfinite(tile.z) or tile.z < 0:
            raise ValueError(f"invalid or out-of-workspace Hive tile: {name!r}")
        neighbors = [other for other in tiles if other.name != name and _distance(other, tile.x, tile.y) < 2 * TILE_RADIUS_M]
        if any(not _at(other, tile.x, tile.y) or abs(other.z - tile.z) < TILE_THICKNESS_M - POSITION_TOL_M for other in neighbors):
            raise ValueError(f"overlapping Hive tiles at {name!r}")
        if tile.z > POSITION_TOL_M and not any(abs(other.z + TILE_THICKNESS_M - tile.z) <= POSITION_TOL_M for other in neighbors):
            raise ValueError(f"unsupported Hive tile: {name!r}")

    locations: dict[tuple[str, str], tuple[float, float]] = {}
    for color, y in zip(COLORS, (layout.cream_y, layout.black_y)):
        keys = sorted({t.key for t in tiles if t.color == color and t.key[1]})
        for i, key in enumerate(keys):
            x = round(layout.x_center + (i - (len(keys) - 1) / 2) * layout.pile_spacing, 4)
            if not layout.fits(x, y):
                raise ValueError(f"not enough workspace for {color} Hive piles")
            if any(_distance(o, x, y) < o.radius_m + TILE_RADIUS_M + PILE_CLEARANCE_M for o in obstacles):
                raise ValueError(f"{color} {key[1]} pile is blocked by a tabletop obstacle")
            locations[key] = (x, y)

    # Count only the contiguous, correct layers from the bottom of each pile.
    stacks: dict[tuple[str, str], list[str]] = {key: [] for key in locations}
    for tile in positions.values():
        if tile.key in locations and _at(tile, *locations[tile.key]):
            stack = stacks[tile.key]
            if abs(tile.z - len(stack) * TILE_THICKNESS_M) <= POSITION_TOL_M:
                stack.append(tile.name)
    settled = {name for stack in stacks.values() for name in stack}
    pending = [t.name for t in positions.values() if t.key[1] and t.name not in settled]
    skipped = tuple(t.name for t in positions.values() if not t.key[1])
    moves: list[Move] = []

    def move(tile: Tile, x: float, y: float, z: float) -> None:
        moves.append(Move(tile.name, (tile.x, tile.y, tile.z), (x, y, z)))
        positions[tile.name] = replace(tile, x=x, y=y, z=z)

    staged: set[str] = set()
    while pending:
        exposed = [
            positions[name] for name in pending
            if not any(_at(t, positions[name].x, positions[name].y) and t.z > positions[name].z + POSITION_TOL_M
                       for t in positions.values())
        ]
        for tile in exposed:
            x, y = locations[tile.key]
            stack = stacks[tile.key]
            if any(_distance(t, x, y) < clearance for t in positions.values() if t.name != tile.name and t.name not in stack):
                continue
            move(tile, x, y, round(len(stack) * TILE_THICKNESS_M, 4))
            stack.append(tile.name)
            pending.remove(tile.name)
            break
        else:
            # Temporarily clear a pile slot involved in a dependency cycle.
            blocker = next((t for t in exposed if t.name not in staged and
                            any(_distance(t, *spot) < clearance for spot in locations.values())), None)
            if blocker is None:
                raise ValueError("Hive piles are blocked by unreadable or inaccessible tiles")
            staging_spots = (
                (round(layout.x_min + TILE_RADIUS_M + i * clearance, 4),
                 round(layout.y_min + TILE_RADIUS_M + j * clearance, 4))
                for j in range(math.ceil((layout.y_max - layout.y_min) / clearance))
                for i in range(math.ceil((layout.x_max - layout.x_min) / clearance))
            )
            for x, y in staging_spots:
                if (layout.fits(x, y) and all(_distance(t, x, y) >= clearance for t in positions.values())
                        and all(math.hypot(px - x, py - y) >= clearance for px, py in locations.values())
                        and all(_distance(o, x, y) >= o.radius_m + TILE_RADIUS_M + PILE_CLEARANCE_M for o in obstacles)):
                    move(blocker, x, y, 0.0)
                    staged.add(blocker.name)
                    break
            else:
                raise ValueError("no clear workspace to stage Hive tiles")

    piles = tuple(Pile(*key, *spot, tuple(stacks[key])) for key, spot in locations.items())
    return Plan(tuple(moves), piles, skipped)
