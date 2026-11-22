"""Tolerant top-down grasp planning from a measured 3D object pose and planar orientation.

Wide jaws self-centre the object as they close, so the jaw-tip midpoint may sit off-centre along the closing
axis and the jaw heading may deviate from the ideal one. Bounded least squares over the arm joints solves
dead-zone residuals for these tolerances, then each solution gets a vertical hover and a clear joint path.
"""
from __future__ import annotations

import collections
import math
import time

import numpy as np
from scipy.optimize import least_squares

from .ik_tracik import TracIKSolver
from .motion import allowed_bounds_rad, lowest_point_m
from .stereo_sweep import joint_path, path_clearance


def _dead(e: float, tol: float) -> float:
    return math.copysign(max(0.0, abs(e) - tol), e)


def grasp_tolerances(width_mm: float, length_mm: float, opening_mm: float) -> tuple[float, float]:
    """Jaw-axis offset slack (mm) and heading slack (rad) that still keep the object inside the open jaws.

    Parameters
    ----------
    width_mm, length_mm : float
        Object footprint across and along its long axis.
    opening_mm : float
        Measured maximum jaw opening.

    Returns
    -------
    tuple of float
        Allowed jaw-tip offset along the closing axis (2 mm spare per side) and allowed heading error.
    """
    jaw_tol = max(0.0, (opening_mm - width_mm) / 2 - 2.0)
    yaw_tol = next((math.radians(d) for d in range(30, -1, -1)
                    if width_mm * math.cos(math.radians(d)) + length_mm * math.sin(math.radians(d)) <= opening_mm - 4),
                   0.0)
    return jaw_tol, yaw_tol


def plan_tolerant_grasp(calib, motors, centre_xy_m, yaw_rad, width_mm, length_mm, height_mm, *,
                        tcp_in_flange, opening_mm=54.0, tip_height_mm=(12.0, 22.0), long_tol_mm=10.0,
                        max_tilt_deg=60.0, seeds=40, max_options=3, hover_backs_m=(0.08, 0.06, 0.05, 0.04),
                        jaw_points_in_tool=None, jaw_floor_clearance_m=0.008) -> dict:
    """Plan grasp and hover joint configurations around a measured object.

    Parameters
    ----------
    calib : CalibrationStore
        Active rig calibration (URDF, table height, motor conversion).
    motors : list of MotorState
        Current follower telemetry; the first seven are the arm joints.
    centre_xy_m : array_like
        Object footprint centre in ``base_link`` XY, metres.
    yaw_rad : float
        Object long-axis heading in ``base_link`` XY.
    width_mm, length_mm, height_mm : float
        Measured object box.
    tcp_in_flange : array_like
        Jaw-tip midpoint in the flange frame, metres.
    opening_mm : float, optional
        Measured jaw opening.
    tip_height_mm : tuple of float, optional
        Allowed jaw-tip height band above the table.
    long_tol_mm : float, optional
        Allowed jaw-tip offset along the object's long axis.
    max_tilt_deg : float, optional
        Maximum approach tilt from vertical.
    seeds : int, optional
        Number of IK seeds (the first is the current pose).
    max_options : int, optional
        Stop after this many feasible grasps.
    hover_backs_m : tuple of float, optional
        Vertical hover distances to try, longest first.
    jaw_points_in_tool : array_like, optional
        Jaw collision-box corners in the flange frame (both jaws, open and closed). When given, every corner
        must stay ``jaw_floor_clearance_m`` above the table: the jaw-tip midpoint alone does not bound the
        jaws, which reach about 27 mm sideways and drop below it when the approach is tilted.
    jaw_floor_clearance_m : float, optional
        Minimum jaw height above the table at the grasp.

    Returns
    -------
    dict
        ``ok``, ``options`` (best first, each with ``q_grasp``, ``q_hover``, ``hover_m``, ``flange_pose``, ``tip_base_m``,
        ``jaw_offset_mm``, ``long_offset_mm``, ``jaw_yaw_deg``, ``tilt_deg``, ``move_deg``, ``path_lowest_m``,
        ``object_clearance_m``), the tolerances used, ``rejected`` counts and ``elapsed_s``.
    """
    t0 = time.monotonic()
    tcp = np.asarray(tcp_in_flange, dtype=float)
    centre = np.asarray(centre_xy_m, dtype=float)[:2]
    long_axis = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    jaw_dir = np.array([-long_axis[1], long_axis[0]])
    table_z = calib.z_table()
    z_lo, z_hi = table_z + tip_height_mm[0] / 1000, table_z + tip_height_mm[1] / 1000
    jaw_tol_mm, yaw_tol = grasp_tolerances(width_mm, length_mm, opening_mm)
    if width_mm + 4 > opening_mm:
        return {"ok": False, "reason": "The object is wider than the jaw opening.", "options": []}
    jaws = None if jaw_points_in_tool is None else np.asarray(jaw_points_in_tool, dtype=float).reshape(-1, 3)
    jaw_floor = table_z + jaw_floor_clearance_m

    def tip_of(T):
        return T[:3, :3] @ tcp + T[:3, 3]

    def jaw_lowest(T):
        return float((jaws @ T[2, :3] + T[2, 3]).min()) if jaws is not None else math.inf

    def violations(T):
        tip = tip_of(T)
        x, y = T[:3, 0], T[:3, 1]
        d = tip[:2] - centre
        jaw_h = x[:2] / max(np.linalg.norm(x[:2]), 1e-9)
        heading = math.asin(np.clip(abs(jaw_h[0] * jaw_dir[1] - jaw_h[1] * jaw_dir[0]), 0, 1))
        tilt = math.acos(np.clip(-y[2], -1, 1))
        # Angles are scaled as metres at a 0.1 m lever so all residuals share units.
        return np.array([
            _dead(d @ jaw_dir, jaw_tol_mm / 1000),
            _dead(d @ long_axis, long_tol_mm / 1000),
            max(0.0, z_lo - tip[2]) - max(0.0, tip[2] - z_hi),
            0.1 * max(0.0, abs(math.asin(np.clip(x[2], -1, 1))) - math.radians(10)),
            0.1 * max(0.0, heading - yaw_tol),
            0.1 * max(0.0, tilt - math.radians(max_tilt_deg)),
            max(0.0, jaw_floor - jaw_lowest(T)),
        ])

    def centring(T):
        d = tip_of(T)[:2] - centre
        return np.array([0.5 * (d @ jaw_dir), 0.2 * (d @ long_axis)])

    solver = TracIKSolver(str(calib.urdf_path()), timeout=0.05)
    q0 = np.array(calib.ticks_to_rad(motors[:7]))
    lower, upper = allowed_bounds_rad(calib, motors)
    rng = np.random.default_rng(0)
    T0 = solver.fk(q0)
    yaw0 = math.atan2(T0[1, 0], T0[0, 0])

    def yaw_change(T):
        return abs((math.atan2(T[1, 0], T[0, 0]) - yaw0 + math.pi) % (2 * math.pi) - math.pi)

    # The jaws are symmetric, so the wrist-roll seeds offer both jaw headings and the nearest one wins.
    roll_seeds = [np.clip(q0 + np.eye(7)[6] * r, lower, upper) for r in (math.pi, -math.pi, math.pi / 2, -math.pi / 2)]
    found = []
    for i in range(seeds):
        seed = q0 if i == 0 else roll_seeds[i - 1] if i <= len(roll_seeds) else np.clip(q0 + rng.normal(0, 0.9, 7), lower, upper)
        res = least_squares(
            lambda q: np.concatenate([violations(solver.fk(q)), centring(solver.fk(q)), 0.002 * (q - q0)]),
            np.clip(seed, lower + 1e-6, upper - 1e-6), bounds=(lower, upper), xtol=1e-8, max_nfev=400,
        )
        if np.max(np.abs(violations(solver.fk(res.x)))) < 1e-3:
            found.append(res.x)

    obstacle = dict(axis_xy=centre, z_low=table_z, z_high=table_z + height_mm / 1000, radius=length_mm / 2000)
    start_lowest = lowest_point_m(calib, q0)
    options, rejected = [], collections.Counter()
    for q_grasp in sorted(found, key=lambda q: (round(math.degrees(yaw_change(solver.fk(q))) / 15),
                                                np.max(np.abs(q - q0)))):
        T = solver.fk(q_grasp)
        q_hover, back = None, None
        for back in hover_backs_m:
            hover_T = T.copy()
            hover_T[2, 3] += back
            q_hover = solver._raw_solve(hover_T[:3, 3], hover_T[:3, :3], q_grasp, lower, upper)
            if q_hover is not None:
                break
        if q_hover is None:
            rejected["hover ik"] += 1
            continue
        path = joint_path(q0, q_hover)
        lowest = min(lowest_point_m(calib, q) for q in path)
        clear = path_clearance(calib, path, obstacle)
        if lowest < min(0.05, start_lowest - 0.002) or clear < 0.015:
            rejected["path to hover"] += 1
            continue
        tip = tip_of(T)
        d = tip[:2] - centre
        options.append({
            "q_grasp": q_grasp.tolist(), "q_hover": np.asarray(q_hover).tolist(), "hover_m": back,
            "flange_pose": T.tolist(),
            "tip_base_m": tip.tolist(), "jaw_offset_mm": float(d @ jaw_dir * 1000),
            "long_offset_mm": float(d @ long_axis * 1000),
            "jaw_yaw_deg": float(math.degrees(math.atan2(T[1, 0], T[0, 0])) % 180),
            "tilt_deg": float(math.degrees(math.acos(np.clip(-T[2, 1], -1, 1)))),
            "jaw_lowest_above_table_m": None if jaws is None else float(jaw_lowest(T) - table_z),
            "yaw_change_deg": float(math.degrees(yaw_change(T))),
            "move_deg": float(math.degrees(np.max(np.abs(np.asarray(q_hover) - q0)))),
            "path_lowest_m": float(lowest), "object_clearance_m": float(clear),
        })
        if len(options) >= max_options:
            break
    return {
        "ok": bool(options), "options": options, "ik_solutions": len(found), "seeds": seeds,
        "jaw_tolerance_mm": jaw_tol_mm, "heading_tolerance_deg": math.degrees(yaw_tol),
        "rejected": dict(rejected), "elapsed_s": round(time.monotonic() - t0, 2),
        **({} if options else {"reason": "No tolerant grasp with a reachable hover and clear path."}),
    }
