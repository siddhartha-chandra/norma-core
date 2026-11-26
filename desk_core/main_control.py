"""Numbered object-to-X workflow shared by the MCP tools and the CLI."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .ik import _mesh_vertices, _rpy_rotation
from .motion import MotionLimits, calibrated_range, lowest_point_m, stream_joint_path
from .object_measurement import fit_cuboid, segment
from .pose_estimation import estimate_grasp_pose as grasp_pose_from_stereo
from .pose_estimation import estimate_top_surface, pixels_on_plane
from .station_bridge import DEFAULT_MAX_STEP, GRIPPER_SLACK_STEPS, REPO_ROOT
from .stereo_sweep import aim_wrist as plan_wrist_aim
from .stereo_sweep import capture_wrist_stereo as run_wrist_sweep
from .stereo_sweep import dense_cloud, joint_path, project, registered_obstacle_path_problem, write_ply
from .tolerant_grasp import plan_tolerant_grasp as plan_tolerant_top_grasp
from .urdf_fk import load_urdf
from .wrist_stereo import StereoResult, WristPair
from .wrist_stereo import estimate_wrist_depth as triangulate_wrist


TABLE_CLEARANCE_M = 0.005
JAW_SAG_MARGIN_M = 0.003
FAST_LIMITS = MotionLimits(speed_cm_s=3.0, rot_speed_deg_s=30.0, joint_speed_deg_s=45.0, min_duration_s=1.0,
                           lag_limit_ticks=120, max_step_ticks=300)
SETTLE_TICKS = 25
HOVER_M = 0.06


## added by Abhi
@dataclass
class WristMeasurement:
    pair: WristPair
    roi: tuple[int, int, int, int]
    q_rad_a: list[float] | None = None
    stereo: StereoResult | None = None
    mask: np.ndarray | None = None
    surface_z_m: float | None = None
    footprint_source: str | None = None


def _require(result: dict) -> dict:
    if not result.get("ok"):
        raise ValueError(result.get("error") or result.get("reason") or str(result.get("problems") or "Stage failed"))
    return result


def _tool_parameters(tool: dict) -> dict:
    if not isinstance(tool, dict):
        raise ValueError("Provide calibrated grasp_tool parameters for the actual follower.")
    required = {"tool_quaternion_xyzw", "tcp_offset_m", "reference_jaw_yaw_deg", "jaw_opening_mm"}
    if missing := required - tool.keys():
        raise ValueError(f"Missing grasp tool calibration: {', '.join(sorted(missing))}")
    q = np.asarray(tool["tool_quaternion_xyzw"], dtype=float)
    offset = np.asarray(tool["tcp_offset_m"], dtype=float)
    yaw = float(tool["reference_jaw_yaw_deg"])
    opening = float(tool["jaw_opening_mm"])
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
        raise ValueError("tool_quaternion_xyzw must be a normalized flange quaternion.")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("tcp_offset_m must be a finite tool-local XYZ vector.")
    if not np.isfinite([yaw, opening]).all() or opening <= 0:
        raise ValueError("Provide a finite reference jaw heading and positive measured jaw opening.")
    if Rotation.from_quat(q).apply([0, 1, 0])[2] > -0.98:
        raise ValueError("This workflow requires a verified top-down flange orientation (local +Y down).")
    return dict(tool_quaternion_xyzw=q, tcp_offset_m=offset,
                reference_jaw_yaw_rad=float(np.deg2rad(yaw)), jaw_opening_mm=opening)


## added by Abhi
### 6. Conservative jaw mesh bounds for table clearance.
@lru_cache(maxsize=4)
def _jaw_bounds(urdf: Path) -> np.ndarray:
    robot = load_urdf(urdf)
    xml = ET.parse(urdf).getroot()
    boxes = []
    for link in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1"):
        node = xml.find(f"./link[@name='{link}']/collision")
        origin = node.find("origin")
        mesh = node.find("./geometry/mesh")
        scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
        stl = _mesh_vertices(urdf.parent / mesh.attrib["filename"]) * scale
        bounds = np.vstack((stl.min(axis=0), stl.max(axis=0)))
        corners = bounds[np.array(list(np.ndindex(2, 2, 2))), np.arange(3)]
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        corners = corners @ _rpy_rotation(rpy).T + xyz
        joint = robot.joints_by_child[link]
        for opening in (joint.lower, joint.upper):
            transform = robot.fk([opening], target_link=link, base_link="Gripper_Base_v1_1", joint_names=[joint.name])
            boxes.append(corners @ transform[:3, :3].T + transform[:3, 3])
    return np.vstack(boxes)


## added by Abhi
class MainControl:
    """Runs each stage with the same bridge, calibration and registered obstacles."""

    def __init__(self, bridge, calib, tools, obstacles):
        self.bridge = bridge
        self.calib = calib
        self.tools = tools
        self.obstacles = obstacles
        self.captures: dict[str, WristMeasurement] = {}
        self.pick_plan: dict | None = None

    ### 1. Observe and validate the intended follower.
    async def observe(self) -> dict:
        observation = await self.tools["observe"](views=["top", "wrist"], denoise=False)
        if len(observation) != 3 or any(isinstance(frame, str) for frame in observation[1:]):
            raise ValueError("Both scene and wrist cameras must be available.")
        summary = json.loads(observation[0])
        link = summary["link"]
        if link.get("bridge") == "live":
            rx_age = link.get("rx_age_ms")
            if not link.get("socket_up") or link.get("resyncing") or rx_age is None or not 0 <= rx_age <= 350:
                raise ValueError("Station RX is stale or reconnecting.")
            hardware = self.calib.hardware_config()
            ids = [hardware[f"camera_{view}"] for view in ("top", "wrist")]
            matches = [[key for key in summary["cameras"] if camera_id and camera_id in key] for camera_id in ids]
            if any(len(keys) != 1 for keys in matches) or matches[0] == matches[1]:
                raise ValueError("Could not identify separate calibrated top and wrist cameras.")
        return summary

    async def _motor_state(self):
        state = await self.bridge.get_state()
        if [m.id for m in state.motors] != list(range(1, 9)) or any(m.error_status for m in state.motors):
            raise ValueError("Expected eight healthy motors on the ElRobot follower.")
        link = self.bridge.link_status()
        if link.get("bridge") == "live":
            bus = self.bridge.target_bus or self.calib.hardware_config()["bus_serial"]
            if not bus or state.bus_serial != bus or link.get("resyncing") or not link.get("socket_up"):
                raise ValueError("Follower bus is missing or changed.")
            now = time.monotonic()
            if any(not 0 <= now - self.bridge.rx_received_at.get((bus, m.id), -np.inf) <= 0.35 for m in state.motors):
                raise ValueError("Expected fresh RX for all eight follower motors.")
        if any(not lo <= m.position <= hi for m in state.motors for lo, hi in [calibrated_range(m)]):
            raise ValueError("The follower is outside a calibrated motor range.")
        return replace(state, motors=[replace(m) for m in state.motors])

    ### 2. Detect the requested object and extract a wrist segmentation.
    async def detect_requested_object(self, query: str, provider: str = "auto") -> dict:
        from .vision_3d import COLOR_HINTS

        if not query.strip():
            raise ValueError("Supply the name of the requested object.")
        if provider == "opencv" and not set(re.findall(r"[a-z]+", query.lower())) & COLOR_HINTS.keys():
            raise ValueError("OpenCV requires an explicit target color. Use a vision provider to detect an object by name.")
        detection = await self.tools["localize_3d"](query=query, camera="top", denoise=False, provider=provider)
        if detection.get("backend") == "opencv-fallback" and provider != "opencv":
            raise ValueError("Scene detection fell back to OpenCV. Confirm the object or select OpenCV for a known color target.")
        if not detection.get("found"):
            return {"ok": False, "found": False, "message": f"I couldn't see {query} in the scene."}
        world = detection["world"]
        target = [float(world[axis]) for axis in "xyz"]
        if not np.isfinite(target).all() or not all(np.isfinite(detection["pixel"])):
            raise ValueError("Scene localization did not return a usable target.")
        return {"ok": True, "found": True, "target_base_m": target, "detection": detection}

    async def _confirm_wrist_target(self, capture_id, query, provider):
        """Detect the same object on the exact reference frame used for stereo."""
        from .vision_3d import localize_object_3d

        record = self.captures[capture_id]
        pair = record.pair
        ok, jpeg = cv2.imencode(".jpg", pair.frame_a)
        if not ok:
            raise ValueError("Unable to encode the reference wrist capture.")
        detection = await localize_object_3d(
            jpeg.tobytes(), query, self.calib, camera="wrist", q_rad=record.q_rad_a, timeout_s=6.0, provider=provider,
        )
        if not detection.get("found") or (detection.get("backend") == "opencv-fallback" and provider != "opencv"):
            raise ValueError(f"Couldn't confirm {query} in the reference wrist frame.")
        yl, xl, yh, xh = detection["box_2d"]
        height, width = pair.frame_a.shape[:2]
        x0, x1 = int(xl * width / 1000), int(np.ceil(xh * width / 1000))
        y0, y1 = int(yl * height / 1000), int(np.ceil(yh * height / 1000))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Wrist detection returned an empty box.")
        x, y, w, h = record.roi
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        # The scene camera target is about 2 cm off, so allow the wrist box centre outside the projected ROI.
        pad = 80
        if not (x - pad <= cx <= x + w + pad and y - pad <= cy <= y + h + pad):
            raise ValueError("Wrist detection disagrees with the scene target projection.")
        # Leave a background band for GrabCut around the detected box.
        x0, x1 = max(0, x0 - 5), min(width, x1 + 5)
        y0, y1 = max(0, y0 - 5), min(height, y1 + 5)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Wrist detection returned an empty box.")
        return x0, y0, x1 - x0, y1 - y0

    ### 3. Aim the wrist with the measured coarse scene target.
    async def aim_wrist(self, target_base_m, execute=False) -> dict:
        return await plan_wrist_aim(self.bridge, self.calib, target_base_m, execute=execute,
                                    obstacles=tuple(self.obstacles.values()))

    ### 4. Capture an identified pair and estimate sparse wrist depth.
    async def capture_wrist_stereo(self, target_base_m, execute=False) -> dict:
        capture = await run_wrist_sweep(self.bridge, self.calib, target_base_m, execute=execute,
                                       obstacles=tuple(self.obstacles.values()))
        if not capture.get("ok") or not execute:
            return capture
        pair = capture.pop("pair")
        capture_id = uuid4().hex
        if len(self.captures) >= 4:
            self.captures.pop(next(iter(self.captures)))
        self.captures[capture_id] = WristMeasurement(pair, tuple(capture["roi"]), capture["joint_angles_a_rad"])
        return {**capture, "capture_id": capture_id}

    async def estimate_wrist_depth(self, capture_id, roi=None, build_cloud=False) -> dict:
        record = self.captures[capture_id]
        pair = record.pair
        roi = record.roi if roi is None else tuple(roi)
        stereo = await asyncio.to_thread(
            triangulate_wrist, pair.frame_a, pair.frame_b, pair.camera_matrix, pair.distortion,
            pair.T_base_cam_a, pair.T_base_cam_b, roi,
        )
        x, y, w, h = roi
        rows, cols = pair.frame_a.shape[:2]
        left, top = max(0, x - 24), max(0, y - 24)
        right, bottom = min(cols, x + w + 24), min(rows, y + h + 24)
        mask = np.zeros((rows, cols), dtype=bool)
        mask[top:bottom, left:right] = await asyncio.to_thread(
            segment, pair.frame_a[top:bottom, left:right], (x - left, y - top, w, h), 3,
        )
        surface = estimate_top_surface(stereo, roi, table_z_m=pair.table_z_m)
        support = np.rint(surface.pixels_a).astype(np.int32)
        support[:, 0] = np.clip(support[:, 0], x, x + w - 1)
        support[:, 1] = np.clip(support[:, 1], y, y + h - 1)
        source = "grabcut"
        if mask[support[:, 1], support[:, 0]].mean() < 0.65 or mask.sum() < 0.20 * w * h:
            hull = cv2.convexHull(support)
            if (len(support) < 12 or np.any(np.ptp(support, axis=0) < 0.55 * np.array([w, h]))
                    or cv2.contourArea(hull) < 0.30 * w * h):
                raise ValueError("The wrist silhouette and depth matches do not cover enough of the target box.")
            # Pad the sampled hull to account for textured features stopping short of the visible boundary.
            region = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(region, hull - np.array([x, y], dtype=np.int32), 1)
            region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            mask.fill(False)
            mask[y:y + h, x:x + w] = region.astype(bool)
            source = "stereo_hull"
        xy = np.rint(stereo.pixels_a).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
        keep = mask[xy[:, 1], xy[:, 0]]
        filtered = StereoResult(
            stereo.points_base_m[keep], stereo.depth_a_m[keep], stereo.pixels_a[keep], stereo.pixels_b[keep],
            stereo.reprojection_error_px[keep], stereo.parallax_deg[keep], stereo.match_indices[keep], stereo.baseline_m,
        )
        surface = estimate_top_surface(filtered, roi, table_z_m=pair.table_z_m)
        record.roi, record.mask, record.stereo, record.surface_z_m = roi, mask, filtered, surface.z_m
        record.footprint_source = source
        result = {"ok": True, "capture_id": capture_id, "roi": list(roi), "baseline_m": stereo.baseline_m,
                  "surface_z_m": surface.z_m, "match_count": len(filtered.points_base_m),
                  "median_depth_m": filtered.median_depth_m, "footprint_source": source}
        if build_cloud:
            result["cloud"] = await asyncio.to_thread(self._cloud, capture_id)
        return result

    def _cloud(self, capture_id: str) -> dict:
        pair = self.captures[capture_id].pair
        relative = np.linalg.solve(pair.T_base_cam_b, pair.T_base_cam_a)
        try:
            points, colors, info = dense_cloud(
                pair.frame_a, pair.frame_b, pair.camera_matrix, pair.distortion, pair.T_base_cam_a, relative,
            )
            if not len(points):
                return {"ok": False, **info}
            directory = REPO_ROOT / "runs" / "pick_place" / capture_id
            directory.mkdir(parents=True, exist_ok=True)
            ply = directory / "cloud.ply"
            write_ply(ply, points, colors)
            stereo = self.captures[capture_id].stereo
            (directory / "stereo_result.json").write_text(json.dumps({
                "table_z_m": pair.table_z_m, "T_base_cam_a": pair.T_base_cam_a.tolist(),
                "T_base_cam_b": pair.T_base_cam_b.tolist(),
                "points_base_m": stereo.points_base_m.tolist() if stereo is not None else [],
            }))
            result = {"ok": True, "ply": str(ply), **info}
            try:
                # Open3D can fail at native graphics initialization. Keep it out of the MCP process.
                render = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "scripts" / "plot_stereo_cloud.py"),
                     str(directory), "--color", "rgb"], capture_output=True, text=True, timeout=30,
                )
                preview = directory / "cloud_rgb.png"
                if render.returncode == 0 and preview.is_file():
                    result["open3d_preview"] = str(preview)
                else:
                    result["open3d_error"] = render.stderr.strip()[-600:] or f"Renderer exited with {render.returncode}"
            except (OSError, subprocess.TimeoutExpired) as exc:
                result["open3d_error"] = str(exc)
            return result
        except (cv2.error, OSError, ValueError) as exc:
            return {"ok": False, "reason": str(exc)}

    ### 5. Measure the segmented footprint and compare the two jaw directions.
    def measure_graspability(self, capture_id, jaw_opening_mm, clearance_mm=2.0) -> dict:
        if not np.isfinite([jaw_opening_mm, clearance_mm]).all() or jaw_opening_mm <= 0 or clearance_mm < 0:
            raise ValueError("Provide a positive measured jaw opening and nonnegative clearance in millimeters.")
        record = self.captures[capture_id]
        pair = record.pair
        if record.mask is None or record.surface_z_m is None or pair.table_z_m is None:
            raise ValueError("Estimate wrist depth and supply the calibrated table height before checking jaw fit.")
        contours, _ = cv2.findContours(record.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(outline) < 3:
            raise ValueError("The target silhouette has no measurable footprint.")
        points = pixels_on_plane(outline, pair.camera_matrix, pair.distortion, pair.T_base_cam_a, record.surface_z_m)
        corners, dims = fit_cuboid(points, pair.table_z_m, 0, 100)
        fits = {name: dims[key] + 2 * clearance_mm <= jaw_opening_mm
                for name, key in (("across", "width_mm"), ("along", "length_mm"))}
        return {"ok": True, "capture_id": capture_id, **dims, "fits": fits,
                "corners_base_m": corners.tolist(), "jaw_opening_mm": jaw_opening_mm,
                "footprint_source": record.footprint_source}

    ### 6. Estimate the grasp pose and preflight the IK waypoints.
    def estimate_grasp_pose(self, capture_id, tool, jaw_offset_deg=90.0, grasp_drop_m=None) -> dict:
        params = _tool_parameters(tool)
        record = self.captures[capture_id]
        pair = record.pair
        opening = params.pop("jaw_opening_mm")
        measurement = self.measure_graspability(capture_id, opening)
        angle = np.deg2rad(jaw_offset_deg)
        span = abs(measurement["length_mm"] * np.cos(angle)) + abs(measurement["width_mm"] * np.sin(angle))
        if not np.isfinite(span) or span + 4 > opening + 1e-9:
            raise ValueError("The measured jaws cannot fit the target at the requested heading.")
        height = measurement["height_mm"] / 1000
        default_drop = height / 2
        drop = default_drop if grasp_drop_m is None else float(grasp_drop_m)
        if not 0 <= drop <= height - TABLE_CLEARANCE_M:
            raise ValueError("The requested grasp depth leaves less than 5 mm at the TCP.")
        yaw = np.deg2rad(measurement["yaw_deg"])
        centre = np.array(measurement["centre_base_mm"]) / 1000
        centre[2] = record.surface_z_m
        axis = np.array([np.cos(yaw), np.sin(yaw), 0]) * measurement["length_mm"] / 4000
        axis_px = [project(pair.T_base_cam_a, pair.camera_matrix, pair.distortion, centre + sign * axis)
                   for sign in (-1, 1)]
        y, x = np.unravel_index(np.argmax(cv2.distanceTransform(record.mask.astype(np.uint8), cv2.DIST_L2, 5)),
                                record.mask.shape)
        pose = grasp_pose_from_stereo(
            record.stereo, pair.camera_matrix, pair.distortion, pair.T_base_cam_a, record.roi,
            **params, axis_pixels=axis_px, grasp_pixel=(float(x), float(y)),
            jaw_offset_rad=float(angle), grasp_drop_m=drop, table_z_m=pair.table_z_m,
        )
        return {"ok": True, "capture_id": capture_id, "ik_target": pose.ik_target,
                "grasp_point_base_m": pose.grasp_point_base_m.tolist(), "object_yaw_rad": pose.object_yaw_rad,
                "object_rpy_rad": [0.0, 0.0, pose.object_yaw_rad],
                "height_mm": measurement["height_mm"], "jaw_yaw_rad": pose.jaw_yaw_rad}

    ### 6. Plan a tolerant top-down grasp from the measured 3D pose and planar yaw.
    async def plan_tolerant_grasp(self, capture_id, tool) -> dict:
        self.pick_plan = None
        params = _tool_parameters(tool)
        measurement = self.measure_graspability(capture_id, params["jaw_opening_mm"])
        state = await self._motor_state()
        plan = await asyncio.to_thread(
            plan_tolerant_top_grasp, self.calib, state.motors, np.array(measurement["centre_base_mm"][:2]) / 1000,
            float(np.deg2rad(measurement["yaw_deg"])), measurement["width_mm"], measurement["length_mm"],
            measurement["height_mm"], tcp_in_flange=params["tcp_offset_m"], opening_mm=params["jaw_opening_mm"],
            jaw_points_in_tool=_jaw_bounds(self.calib.urdf_path()),
            jaw_floor_clearance_m=TABLE_CLEARANCE_M + JAW_SAG_MARGIN_M,
        )
        summary = {k: plan[k] for k in ("ik_solutions", "seeds", "jaw_tolerance_mm", "heading_tolerance_deg",
                                        "rejected", "elapsed_s") if k in plan}
        if not plan["ok"]:
            return {"ok": False, "capture_id": capture_id, "reason": plan["reason"], "measurement": measurement,
                    **summary}
        best = plan["options"][0]
        T = np.asarray(best["flange_pose"])
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()
        ik_target = dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), map(float, (*T[:3, 3], *quat))))
        self.pick_plan = {"capture_id": capture_id, "q_hover": best["q_hover"], "q_contact": best["q_grasp"]}
        return {"ok": True, "capture_id": capture_id, "measurement": measurement, "ik_target": ik_target,
                "grasp_point_base_m": best["tip_base_m"], "q_hover": best["q_hover"], "q_contact": best["q_grasp"],
                "jaw_yaw_rad": float(np.deg2rad(best["jaw_yaw_deg"])), "jaw_offset_mm": best["jaw_offset_mm"],
                "long_offset_mm": best["long_offset_mm"], "hover_m": best["hover_m"],
                "jaw_lowest_above_table_m": best["jaw_lowest_above_table_m"], **summary}

    def _checked_path(self, q0, q1, carried=None, lift_from_table=False):
        a, b = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
        if a.shape != (7,) or b.shape != (7,) or not np.isfinite([a, b]).all():
            raise ValueError("A motion waypoint must contain seven finite joint angles.")
        path = joint_path(a, b, step_deg=1.0)
        table_z = self.calib.z_table()
        floor = table_z + TABLE_CLEARANCE_M
        object_floor = table_z - 0.001 if lift_from_table else floor
        jaw = _jaw_bounds(self.calib.urdf_path())
        for q in path:
            pose = self.calib.fk_ee_pose(q)
            jaw_z = (jaw @ pose[2, :3] + pose[2, 3]).min()
            if min(jaw_z, lowest_point_m(self.calib, q)) < floor:
                raise ValueError("A planned waypoint violates the 5 mm table clearance.")
            if carried is not None:
                bottom_z = (carried @ pose[2, :3] + pose[2, 3]).min()
                if bottom_z < object_floor:
                    raise ValueError("A planned waypoint moves the carried object into the table clearance.")
                if bottom_z >= floor:
                    object_floor = floor
        if carried is not None and bottom_z < floor:
            raise ValueError("The lift did not clear the table by 5 mm.")
        obstacle_points = jaw if carried is None else np.vstack((jaw, carried))
        if problem := registered_obstacle_path_problem(
            self.calib, path, tuple(self.obstacles.values()), points_in_flange=obstacle_points,
        ):
            raise ValueError(problem)
        return path

    def _validate_goal(self, goal, state):
        motors = [replace(m, range_min=calibrated_range(m)[0], range_max=calibrated_range(m)[1]) for m in state.motors]
        ok, reason = self.calib.validate_joint_limits(goal, motors)
        if not ok:
            raise ValueError(reason)
        if any(not lo - 1e-6 <= q <= hi + 1e-6 for q, (lo, hi) in zip(goal, self.calib.bounds_rad(motors))):
            raise ValueError("The IK solution exceeds the calibrated motor range.")

    async def _move_checked(self, goal, operator_limits, carried=None, lift_from_table=False):
        state = await self._motor_state()
        self._validate_goal(goal, state)
        start = self.calib.ticks_to_rad(state.motors)
        path = await asyncio.to_thread(self._checked_path, start, goal, carried, lift_from_table)
        xyz = np.array([self.calib.fk_ee_pose(q)[:3, 3] for q in path])
        length_cm = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) * 100
        travel = float(np.abs(path[-1] - path[0]).max())
        duration = max(operator_limits.min_duration_s, 1.875 * length_cm / operator_limits.speed_cm_s,
                       1.875 * np.degrees(travel) / operator_limits.joint_speed_deg_s)
        actual_start = (await self._motor_state()).positions()
        expected_start = self.calib.rad_to_ticks(start, state.motors)
        if any(abs(actual_start[mid] - tick) > 6 for mid, tick in expected_start.items()):
            raise ValueError("The follower moved while this path was being checked.")
        _require(await stream_joint_path(self.bridge, self.calib, path, duration, operator_limits, gravity_ff=False))
        targets = self.calib.rad_to_ticks(list(goal), state.motors)
        for _ in range(15):
            final = (await self._motor_state()).positions()
            if all(abs(final[mid] - tick) <= SETTLE_TICKS for mid, tick in targets.items()):
                break
            await asyncio.sleep(0.2)
        else:
            off = {mid: final[mid] - tick for mid, tick in targets.items() if abs(final[mid] - tick) > SETTLE_TICKS}
            raise ValueError(f"Follower did not settle at the checked waypoint (ticks off: {off}).")

    ### 7. Pick, then lift and recheck the gripper.
    async def _require_held(self):
        state = await self._motor_state()
        gripper = state.motors[-1]
        close = int(self.calib.data["gripper"]["close_ticks"])
        if hasattr(self.bridge, "_effective_motor_range"):
            lo, hi = self.bridge._effective_motor_range(state.bus_serial, gripper)
        elif 0 <= gripper.range_min < gripper.range_max and (gripper.range_min, gripper.range_max) != (0, 4095):
            lo, hi = gripper.range_min, gripper.range_max
        else:
            lo, hi = 0, 4095
        if 0 <= lo < hi and (lo, hi) != (0, 4095):
            close = max(lo, min(hi, close))
        if gripper.position - close <= GRIPPER_SLACK_STEPS:
            raise ValueError("Object is no longer held.")

    async def _pick(self, hover, contact, limits, carried):
        _require(await self.tools["release"]())
        await self._move_checked(hover, limits)
        await self._move_checked(contact, limits)
        grip = _require(await self.tools["grasp"]())
        if not grip.get("holding"):
            raise ValueError("Gripper did not verify contact.")
        await self.retract_to_hover(limits, carried)
        return grip

    ### 7. Retract the closed gripper straight back to the planned pickup hover before carrying.
    async def retract_to_hover(self, limits=None, carried=None) -> dict:
        if not getattr(self, "pick_plan", None):
            raise ValueError("Plan a tolerant grasp before retracting to its hover.")
        await self._move_checked(self.pick_plan["q_hover"], limits or FAST_LIMITS, carried,
                                 lift_from_table=True)
        await self._require_held()
        return {"ok": True, "q_hover": self.pick_plan["q_hover"]}

    ### 8. Carry to the saved X with its stored joint ticks (no IK) through a checked joint path.
    async def _drop_goal(self):
        drop = json.loads((REPO_ROOT / "calib" / "drop_location.json").read_text())
        if "joint_ticks" not in drop:
            raise ValueError("calib/drop_location.json has no joint_ticks for the drop pose.")
        state = await self._motor_state()
        arm = [replace(m, position=int(drop["joint_ticks"][str(m.id)])) for m in state.motors[:7]]
        return self.calib.ticks_to_rad(arm)

    async def move_to_drop(self, limits=None, carried=None) -> dict:
        goal = await self._drop_goal()
        state = await self._motor_state()
        self._validate_goal(goal, state)
        await self._move_checked(goal, limits or FAST_LIMITS, carried)
        return {"ok": True, "q_drop": list(goal)}

    async def _place(self, limits, carried):
        await self.move_to_drop(limits, carried)
        await self._require_held()
        return _require(await self.tools["release"]())

    ### 9. Return to the saved home through a checked joint path.
    async def _home_goal(self):
        home = json.loads((REPO_ROOT / "calib" / "fixed_positions.json").read_text())["home"]
        state = await self._motor_state()
        arm = [replace(m, position=int(home["joint_ticks"][str(m.id)])) for m in state.motors[:7]]
        ticks = int(home["gripper_ticks"])
        lo, hi = calibrated_range(state.motors[-1])
        if not lo <= ticks <= hi:
            raise ValueError("The saved gripper home is outside its calibrated ticks.")
        return self.calib.ticks_to_rad(arm), ticks

    async def _home_gripper(self, target):
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            grip = (await self._motor_state()).motors[-1]
            if abs(grip.position - target) <= 25:
                return
            applied = await self.bridge.send_joint_targets({grip.id: target}, max_step=DEFAULT_MAX_STEP)
            if applied.get(grip.id, grip.position) == grip.position or not await self.bridge.wait_settle(
                applied, timeout_s=2.0,
            ):
                break
        raise ValueError("The gripper did not reach its saved home ticks.")

    ## added by Abhi
    ### 10. Run the numbered workflow with failure gates and timings.
    async def master_pick_place(self, query, execute=False, tool=None, obstacles=None, provider="auto",
                                build_cloud=False, return_home=True) -> dict:
        trace, stage, started, moved = [], "observe", time.perf_counter(), False

        def complete(name):
            nonlocal started
            trace.append({"step": name, "duration_ms": round(1000 * (time.perf_counter() - started), 1)})
            started = time.perf_counter()

        try:
            await self.observe()
            complete(stage)

            stage = "detect_requested_object"
            detection = await self.detect_requested_object(query, provider)
            complete(stage)
            if not detection["ok"]:
                return {**detection, "executed": False, "trace": trace}
            target = detection["target_base_m"]

            stage = "validate_setup"
            if execute:
                if self.calib.data.get("calibrated") is not True or "z_table" not in self.calib.data:
                    raise ValueError("Execution requires the measured follower rig and table height.")
                tool = tool if tool is not None else self.calib.data.get("grasp_tool")
                params = _tool_parameters(tool)
                await self._motor_state()
            for obstacle in obstacles or ():
                await self.tools["update_scene_obstacle"](**obstacle)
            complete(stage)

            stage = "aim_wrist"
            moved = execute
            aim = _require(await self.aim_wrist(target, execute))
            complete(stage)
            if not execute:
                return {"ok": True, "executed": False, "found": True, "detection": detection,
                        "aim": aim, "trace": trace, "next_step": "capture_wrist_stereo"}
            stage = "open_gripper"
            opened = _require(await self.tools["release"]())
            if opened.get("released") is False:
                raise ValueError("Gripper did not open before wrist stereo.")
            complete(stage)
            stage = "capture_wrist_stereo"
            capture = _require(await self.capture_wrist_stereo(target, execute=True))
            capture_id = capture["capture_id"]
            complete(stage)

            stage = "estimate_wrist_depth"
            roi = await self._confirm_wrist_target(capture_id, query, provider)
            depth = _require(await self.estimate_wrist_depth(capture_id, roi=roi))
            complete(stage)
            stage = "measure_graspability"
            measurement = self.measure_graspability(capture_id, params["jaw_opening_mm"])
            if not measurement["fits"]["across"]:
                raise ValueError("The segmented target is too wide for the measured jaws.")
            complete(stage)

            stage = "plan_tolerant_grasp"
            pick = _require(await self.plan_tolerant_grasp(capture_id, tool))
            contact = pick["ik_target"]
            tcp_at_pick = np.asarray(pick["grasp_point_base_m"])
            rotation = Rotation.from_quat([contact[key] for key in ("qx", "qy", "qz", "qw")])
            bottom = np.asarray(measurement["corners_base_m"])[:4]
            carried = params["tcp_offset_m"] + rotation.inv().apply(bottom - tcp_at_pick)
            complete(stage)

            stage = "check_motion_paths"
            state = await self._motor_state()
            current = self.calib.ticks_to_rad(state.motors)
            q_hover, q_contact = pick["q_hover"], pick["q_contact"]
            q_drop = await self._drop_goal()
            home, home_gripper = await self._home_goal() if return_home else (None, None)
            for q in [q_hover, q_contact, q_drop] + ([] if home is None else [home]):
                self._validate_goal(q, state)
            limits = FAST_LIMITS
            routes = [(current, q_hover, None), (q_hover, q_contact, None), (q_contact, q_hover, carried, True),
                      (q_hover, q_drop, carried)]
            if home is not None:
                routes.append((q_drop, home, None))
            for route in routes:
                await asyncio.to_thread(self._checked_path, *route)
            complete(stage)

            stage = "pick"
            grip = await self._pick(q_hover, q_contact, limits, carried)
            complete(stage)
            stage = "place"
            release = await self._place(limits, carried)
            complete(stage)
            if home is not None:
                stage = "home"
                await self._move_checked(home, limits)
                await self._home_gripper(home_gripper)
                complete(stage)
            result = {"ok": True, "executed": True, "found": True, "capture_id": capture_id, "depth": depth,
                      "aim": aim, "stereo_capture": capture,
                      "measurement": measurement, "pick": pick, "grasp": grip, "release": release,
                      "home": return_home, "trace": trace}
            if build_cloud:
                stage = "cloud"
                result["cloud"] = await asyncio.to_thread(self._cloud, capture_id)
                complete(stage)
            return result
        except Exception as exc:
            complete(stage)
            recovery = {}
            if moved:
                try:
                    recovery["estop"] = await self.tools["estop"]()
                    await self.observe()
                except Exception as recovery_error:
                    recovery["error"] = str(recovery_error)
            return {"ok": False, "executed": moved, "failed_step": stage, "error": str(exc),
                    "trace": trace, "recovery": recovery}


## added by Abhi
### CLI. Call the same master MCP implementation from the repository root.
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Object name to pick and place at X.")
    parser.add_argument("--execute", action="store_true", help="Run the workflow. Default is an observation and aim dry run.")
    parser.add_argument("--tool", type=Path, help="Measured grasp tool JSON. Defaults to grasp_tool in the active rig.")
    parser.add_argument("--obstacles", type=Path, help="JSON array of tabletop obstacles for this run.")
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--cloud", action="store_true", help="Build a point cloud after placing the object.")
    parser.add_argument("--no-home", action="store_true")
    args = parser.parse_args()
    from .mcp_server import master_pick_place

    tool = json.loads(args.tool.read_text()) if args.tool else None
    obstacles = json.loads(args.obstacles.read_text()) if args.obstacles else None
    result = asyncio.run(master_pick_place(args.query, execute=args.execute, tool=tool, obstacles=obstacles,
                                           provider=args.provider, build_cloud=args.cloud, return_home=not args.no_home))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
