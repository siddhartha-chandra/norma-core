# DESK-CORE Operator Guide (Claude Code & OpenAI Codex)

This repo runs the **DESK-CORE** MCP server (`desk_core.mcp_server`) connected to NormaCore Station (`MockStationBridge` when `STATION_HOST` is unset, `LiveStationBridge` when set in `.env`).

## Essential Verification Commands
```bash
uv run --offline --managed-python python scripts/test_reconnect.py
uv run --offline --managed-python python scripts/mcp_smoke.py
uv run --offline --managed-python python scripts/test_primitives.py
```

## Robot Hardware Rules (Never Skip)
1. Live state comes from `st3215/rx`; range clamping and creep step limits (`ROBOT_MAX_STEP_TICKS`) are enforced server-side in `desk_core/station_bridge.py`.
2. Always start a session with `observe()` (verify `link.rx_age_ms` and both `top` + `wrist` cameras).
3. Register tabletop obstacles (mug, tray) with `update_scene_obstacle()` before moving.
4. Use the Pink IK / pose pipeline (`solve_ik -> move_joint`, `move_to_pose`, `pick_at`, `place_at`, `execute_sequence`) rather than blind raw tick guesses.
5. If anything shifts or stalls, call `estop()` immediately and re-`observe()`.

## Teammate Integration Plug-Ins
- **Suhail (`Pink` IK):** Implement `desk_core/ik_pink.py` with `solve_ik(x, y, z, alpha, beta, gamma, q_curr_rad=..., obstacles=...) -> list[float]` (or set `PINK_IK_URL`), then call `reload_config()`.
- **Abhimanyu & Scot (Calibration + Depth):** Write `calib/rig_calib.json` (intrinsics `K_top`, extrinsics `T_world_cam_top`, `z_table`, `H_table`, `joint_signs`, `joint_zero_ticks`) or `desk_core/geom_depth.py:get_metric_depth(x, y, camera)`, then call `reload_config()`.
