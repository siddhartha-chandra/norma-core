---
name: desk-operator
description: Safe embodied control loop for DESK-CORE on NormaCore Station via MCP primitives and composite pipelines.
---

# DESK-CORE Robot Operator Skill

## Available Primitives & Composites
- **Observe & Angles:** `observe()`, `get_frame(camera="top")`, `get_joint_angles()`, `get_calib_status()`
- **3D Estimation:** `get_metric_depth(x, y, camera="top")`, `pixel_to_world(x, y)`
- **Kinematics & Actuation:** `solve_ik(x, y, z, alpha, beta, gamma) -> [j_1..j_7]`, `move_joint(joints) -> bool`
- **Composites:** `move_to_pose(...)`, `pick_at(...)`, `place_at(...)`, `pick_pixel(...)`, `execute_sequence(steps)`
- **Gripper & Safety:** `grasp()`, `release()`, `go_home()`, `estop()`

## Standard Task Procedure
1. **Inspect:** Call `observe(views=["top", "wrist"])` and `get_calib_status()`.
2. **Map Obstacles:** Identify mug / tray obstacles in the top frame and register them with `update_scene_obstacle()`.
3. **Execute Safely (Prefer Composites):**
   - For simple target pickup from image coordinates `(x, y)`: run `pick_pixel(x, y, yaw=...)` or chain `get_metric_depth -> pixel_to_world -> solve_ik -> move_joint -> grasp` via `execute_sequence(...)`.
   - For carrying to tray: call `place_at(tray_x, tray_y, tray_z)` then `go_home()`.
4. **Recover Cleanly:** If `move_joint` returns `False` or `grasp` shows `holding=False`, re-open jaws (`release()`), re-`observe()`, and retry safely; call `estop()` on any obstacle hazard.
