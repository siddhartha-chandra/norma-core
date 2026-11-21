# DESK-CORE Claude Code Instructions

Follow `AGENTS.md` and use the `.claude/skills/desk-operator/SKILL.md` workflow whenever controlling the arm via the `desk-core` MCP server.

- To force Mock mode during tests: `DESK_MOCK=1`
- To switch to the live rig: set `STATION_HOST`, `STATION_BUS_SERIAL`, `CAMERA_TOP`, `CAMERA_WRIST` in `.env` and run `uv run --offline --managed-python python scripts/mcp_smoke.py --live`.
