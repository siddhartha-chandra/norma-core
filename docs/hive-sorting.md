# Sort Hive tiles in the Station mock

From the repository root, start the Agent TUI with a seeded scene:

```sh
DESK_MOCK_HIVE_SEED=42 uv run python scripts/agent_tui.py --mock
```

Enable **Arm Motion** and enter **Sort Hive tiles**. This command does not
require a Jev or vision-model key. Change the seed for a different arrangement
of 12 tiles. The top camera shows cream and black rows, each with separate
ant, bee, and beetle piles.

The MCP tool `hive_sort` defaults to a dry run: it returns the planned moves,
pile memberships (bottom first), and skipped tiles without moving anything.
To arrange the mock scene, call `hive_sort` with `execute: true`. For a
custom mock scene, add objects with `mock_add_object` using
`shape: "hive_tile"`, a color of `"cream"` or `"black"`, and an
`insect` label. Keep non-Hive objects clear of the two sorting rows.
Unreadable tiles (`insect: null`) are left in place and reported.

The mock uses its object inventory instead of camera classification and its
Cartesian gripper and stacking model instead of joint IK. Live execution is
disabled: tile recognition and pickup/placement have not been calibrated or
validated against physical tiles. To preview the layout from measured tiles
without Station, pass `tiles` to `hive_sort` with `name`, `color`, `insect`,
`x`, `y`, and optional `z` (tile base height above the table; default 0).
Coordinates are in meters and use the workspace in `calib/hive_layout.json`.
