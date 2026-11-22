#!/usr/bin/env python3
"""End-to-end stdio verification of the 5 robotics primitives, composites & plugins."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{' — ' + detail if detail else ''}")
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def _parse(res):
    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    for b in res.content:
        if getattr(b, "type", None) == "text":
            t = b.text.strip()
            try:
                p = json.loads(t)
                return p.get("result", p) if isinstance(p, dict) and set(p) == {"result"} else p
            except ValueError:
                return t
    return None


async def main() -> int:
    print("desk-core Primitives & Composite MCP Test (8-motor / 7-DoF mock)")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Verify hot-reloadable CalibrationStore override in temp path
        calib_path = Path(tmpdir) / "rig_calib.json"
        calib_path.write_text(json.dumps({
            "arm_type": "elrobot-8dof-test",
            "num_arm_joints": 7,
            "hardware": {
                "bus_serial": "CAL-BUS-99",
                "camera_top": "cal-top-cam",
                "camera_wrist": "cal-wrist-cam",
            },
            "K_top": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            "dist_top": [-0.25, 0.05, 0.0, 0.0, 0.0],
            "T_world_cam_top": [
                [1.0, 0.0, 0.0, 0.25],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.70],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "K_wrist": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            "dist_wrist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "T_ee_cam_wrist": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -0.5, 0.8660254, -0.04],
                [0.0, -0.8660254, -0.5, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "z_table": 0.05,
        }))

        env = dict(os.environ)
        env["DESK_MOCK"] = "1"
        env["DESK_NO_DOTENV"] = "1"
        env["STATION_HOST"] = ""
        env.pop("STATION_BUS_SERIAL", None)
        env.pop("CAMERA_TOP", None)
        env.pop("CAMERA_WRIST", None)
        env["ROBOT_NUM_MOTORS"] = "8"
        env["DESK_CALIB_PATH"] = str(calib_path)

        cmd = "uv" if shutil.which("uv") else sys.executable
        args = ["run", "--offline", "--managed-python", "python", "-m", "desk_core.mcp_server"] if cmd == "uv" else ["-m", "desk_core.mcp_server"]
        params = StdioServerParameters(command=cmd, args=args, env=env, cwd=str(REPO_ROOT))

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = {t.name for t in (await session.list_tools()).tools}
                for req in [
                    "move_joint", "get_frame", "solve_ik", "get_joint_angles", "get_metric_depth",
                    "move_to_pose", "pick_at", "place_at", "pick_pixel", "execute_sequence",
                    "pixel_to_world", "localize_3d", "update_scene_obstacle", "get_calib_status", "reload_config",
                ]:
                    check(f"tool exposed: {req}", req in tools)

                prompts = {p.name for p in (await session.list_prompts()).prompts}
                check("prompt skills exposed", {"desk_operator_skill", "pick_and_place_skill"} <= prompts, str(sorted(prompts)))

                # 1. get_joint_angles() -> [j1..j7]
                q0 = _parse(await session.call_tool("get_joint_angles", {}))
                check("get_joint_angles returns 7 rad floats", isinstance(q0, list) and len(q0) == 7, str(q0))

                from desk_core.calib_store import CalibrationStore
                from desk_core.station_bridge import MotorState
                conversion_calib = CalibrationStore()
                measured_ranges = {
                    1: (1146, 2950),
                    2: (1002, 3094),
                    3: (907, 3189),
                    4: (900, 3196),
                    5: (166, 3930),
                    6: (1061, 3035),
                    7: (124, 3972),
                }
                conversion_motors = [
                    MotorState(mid, 2048, 0, *measured_ranges[mid])
                    for mid in range(1, 8)
                ]
                q_roundtrip = [0.0, 0.25, -0.5, 1.0, -1.0, 0.75, -0.25]
                ticks_roundtrip = conversion_calib.joint_angles_to_ticks(q_roundtrip, conversion_motors)
                q_converted = conversion_calib.ticks_to_joint_angles(
                    [
                        MotorState(mid, ticks_roundtrip[mid], 0, 0, 4095)
                        for mid in range(1, 8)
                    ]
                )
                check(
                    "radian/URDF joint angles round-trip through ST3215 ticks",
                    all(abs(a - b) < 0.002 for a, b in zip(q_roundtrip, q_converted)),
                    f"{ticks_roundtrip} -> {q_converted}",
                )

                # 2. solve_ik(x, y, z, qx, qy, qz, qw) -> [j1..j7]
                # Target 1 cm above the current flange pose, so it is reachable within the motor ranges.
                from scipy.spatial.transform import Rotation

                from desk_core.ik_tracik import get_solver

                T_now = get_solver().fk(q0)
                ik_x, ik_y, ik_z = (T_now[:3, 3] + [0.0, 0.0, 0.01]).tolist()
                ik_qx, ik_qy, ik_qz, ik_qw = Rotation.from_matrix(T_now[:3, :3]).as_quat().tolist()
                q_ik = _parse(await session.call_tool("solve_ik", {
                    "x": ik_x, "y": ik_y, "z": ik_z,
                    "qx": ik_qx, "qy": ik_qy, "qz": ik_qz, "qw": ik_qw,
                }))
                check("solve_ik returns 7 rad floats", isinstance(q_ik, list) and len(q_ik) == 7 and all(isinstance(v, (int, float)) for v in q_ik), str(q_ik))

                # 3. move_joint(joints=[...]) -> True, and move_joint(out-of-range) -> False
                ok_move = _parse(await session.call_tool("move_joint", {"joints": q_ik}))
                check("move_joint(valid) returns True", ok_move is True, repr(ok_move))
                bad_move = _parse(await session.call_tool("move_joint", {"joints": [9.9] * 7}))
                check("move_joint(out_of_range) returns False", bad_move is False, repr(bad_move))

                # Verify joint angles updated cleanly
                q1 = _parse(await session.call_tool("get_joint_angles", {}))
                check("angles changed after move_joint", any(abs(a - b) > 1e-3 for a, b in zip(q0, q1)), f"{q0[:2]} -> {q1[:2]}")

                # 4. get_frame() -> numpy RGB matrix saved + optional inline matrix + Image block
                res_frame = await session.call_tool("get_frame", {"camera": "top", "as_matrix": True, "max_dim": 16})
                meta = _parse(res_frame)
                arr = np.load(meta["npy_path"])
                check("get_frame saves valid RGB uint8 ndarray", arr.shape in ((480, 640, 3), (720, 1280, 3)) and arr.dtype == np.uint8, str(arr.shape))
                check("get_frame inline matrix matches requested max_dim", "matrix" in meta and len(meta["matrix"]) in (9, 12), str(meta.get("matrix_shape")))

                # 5. get_metric_depth(x, y) + undistortion + wrist FK check
                depth_z = _parse(await session.call_tool("get_metric_depth", {"x": 320.0, "y": 240.0, "camera": "top"}))
                check("get_metric_depth uses calibrated ray-plane z", isinstance(depth_z, float) and abs(depth_z - 0.65) < 1e-3, f"z={depth_z}")

                corner_no_dist_x = 0.25 + ((560.0 - 320.0) / 600.0) * 0.65
                w_corner = _parse(await session.call_tool("pixel_to_world", {"x": 560.0, "y": 240.0, "camera": "top"}))
                check("pixel_to_world applies cv2 undistortion on corners", abs(w_corner["x"] - corner_no_dist_x) > 0.01, f"x={w_corner['x']} vs pinhole={corner_no_dist_x:.4f}")

                w_wrist_before = _parse(await session.call_tool("pixel_to_world", {"x": 320.0, "y": 240.0, "camera": "wrist", "q_rad": [0.0] * 7}))
                w_wrist_moved = _parse(await session.call_tool("pixel_to_world", {"x": 320.0, "y": 240.0, "camera": "wrist"}))
                check("wrist FK shifts pixel_to_world with live joint pose", w_wrist_before != w_wrist_moved, f"{w_wrist_before} -> {w_wrist_moved}")

                cal_st = _parse(await session.call_tool("get_calib_status", {}))
                hw = cal_st["calibration"]["hardware"]
                check("hardware fallback read from rig_calib.json", hw == {"bus_serial": "CAL-BUS-99", "camera_top": "cal-top-cam", "camera_wrist": "cal-wrist-cam"}, str(hw))

                # 6. Verify {wrist, scene} camera_calibration.json schema normalization + missing_required
                from desk_core.calib_store import CalibrationStore
                cam_only = Path(tmpdir) / "camera_calibration.json"
                cam_only.write_text(json.dumps({
                    "wrist": {"camera_matrix": [[457.0, 0.0, 650.0], [0.0, 456.0, 368.0], [0.0, 0.0, 1.0]], "distortion_coefficients": [0.08, -0.09, 0.0, 0.0, 0.02]},
                    "scene": {"camera_matrix": [[458.7, 0.0, 652.0], [0.0, 457.8, 375.3], [0.0, 0.0, 1.0]], "distortion_coefficients": [0.086, -0.115, 0.0, 0.0, 0.039]},
                }))
                cs_cam = CalibrationStore(cam_only)
                st_cam = cs_cam.status()
                check("camera_calibration.json maps scene->top & reports missing_required", st_cam["has_K_top"] and st_cam["has_K_wrist"] and "T_world_cam_top (or H_table)" in st_cam["missing_required"], str(st_cam["missing_required"]))

                # 6. Composite execute_sequence ($var pipeline) + pick_pixel
                seq_res = _parse(await session.call_tool("execute_sequence", {
                    "steps": [
                        {"tool": "update_scene_obstacle", "args": {"name": "mug", "x": 0.25, "y": 0.0}},
                        {"tool": "solve_ik", "args": {"x": ik_x, "y": ik_y, "z": ik_z, "qx": ik_qx, "qy": ik_qy, "qz": ik_qz, "qw": ik_qw}, "save_as": "q_pick"},
                        {"tool": "move_joint", "args": {"joints": "$q_pick"}},
                        {"tool": "grasp", "args": {}},
                        {"tool": "release", "args": {}},
                    ]
                }))
                check("execute_sequence composite pipeline passes", seq_res.get("ok") is True and seq_res.get("steps") == 5, str(seq_res.get("failed_step")))

                pix_res = _parse(await session.call_tool("pick_pixel", {"x": 180.0, "y": 310.0, "camera": "top"}))
                check("pick_pixel composite passes", pix_res.get("ok") is True and "world" in pix_res, str(pix_res.get("world")))

                # 7. localize_3d MCP tool (OpenCV fallback when GEMINI_API_KEY unset)
                loc_mcp = _parse(await session.call_tool("localize_3d", {"query": "keys", "camera": "top", "use_gemini": False}))
                check(
                    "localize_3d MCP tool returns 6D pose + box_2d",
                    loc_mcp.get("backend") == "opencv-fallback" and "world" in loc_mcp and "pose_6d" in loc_mcp and len(loc_mcp.get("box_2d", [])) == 4,
                    str(loc_mcp.get("world")),
                )

        # 8. Direct vision_3d unit tests: orange servo horn masking + Gemini response schema
        import cv2
        from unittest.mock import patch
        from desk_core.calib_store import CalibrationStore
        from desk_core.vision_3d import localize_object_3d

        cs_vis = CalibrationStore(calib_path)
        canvas = np.full((480, 640, 3), 230, dtype=np.uint8)
        # Draw big orange servo horn on left (should be masked!)
        cv2.rectangle(canvas, (60, 120), (200, 340), (20, 120, 250), -1)  # BGR orange
        # Draw smaller blue key fob on right (should be detected!)
        cv2.rectangle(canvas, (440, 200), (540, 280), (210, 70, 20), -1)  # BGR blue
        ok_enc, buf = cv2.imencode(".jpg", canvas)
        assert ok_enc
        scene_jpeg = buf.tobytes()

        res_cv = await localize_object_3d(scene_jpeg, "keys", cs_vis, use_gemini=False)
        check(
            "opencv fallback masks orange servo horn & finds target",
            res_cv["found"] and res_cv["backend"] == "opencv-fallback" and 440 <= res_cv["pixel"][0] <= 540,
            f"pixel={res_cv['pixel']} world={res_cv['world']}",
        )

        class _FakeInteraction:
            output_text = json.dumps({
                "found": True,
                "label": "silicone key fob",
                "point": [500, 765],
                "box_2d": [416, 687, 583, 843],
                "yaw_deg": 15.0,
                "confidence": 0.96,
            })

        with patch("desk_core.vision_3d._call_gemini_interactions_sync", return_value={
            "found": True,
            "label": "silicone key fob",
            "point": [500, 765],
            "box_2d": [416, 687, 583, 843],
            "yaw_deg": 15.0,
            "confidence": 0.96,
        }):
            res_gem = await localize_object_3d(scene_jpeg, "keys", cs_vis, use_gemini=True)
            check(
                "gemini-4-hydrogen path maps [y,x] 0-1000 to metric 6D pose",
                res_gem["found"] and res_gem["backend"] == "gemini-4-hydrogen" and abs(res_gem["pixel"][0] - 489.6) < 1.0
                and all(key in res_gem["pose_6d"] for key in ("qx", "qy", "qz", "qw")),
                f"pixel={res_gem['pixel']} pose_6d={res_gem['pose_6d']}",
            )

    print("\nPRIMITIVES & COMPOSITE MCP PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
