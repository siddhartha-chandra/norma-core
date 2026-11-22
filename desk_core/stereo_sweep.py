"""Kinematics-aware wrist stereo sweeps: slide planning, image-refined relative pose and dense colored clouds.

The wrist camera is rigidly mounted on the flange, so a straight flange slide with fixed orientation
translates the camera by the same vector. Slides are chosen perpendicular to the optical axis, sized for
a target parallax, kept short enough that the target stays in frame, and ranked by how far the joints
must swing (large swings amplify kinematic model errors, which break the epipolar search).

Relative pose convention: ``rel`` maps camera-A coordinates to camera-B coordinates (x_b = R x_a + t),
i.e. ``rel = inv(T_base_cam_b) @ T_base_cam_a``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import itertools
import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .calib_store import TICKS_PER_RAD, CalibrationStore
from .ik_tracik import TracIKSolver
from .motion import TCP_IN_FLANGE, MotionLimits, allowed_bounds_rad, lowest_point_m, plan_flange_move


@dataclass
class SlideCandidate:
    """One planned camera slide."""

    image_angle_deg: float
    distance_m: float
    direction_base: np.ndarray
    plan: dict = field(repr=False)
    max_joint_deg: float = math.inf

    @property
    def ok(self) -> bool:
        return bool(self.plan.get("ok"))

    def describe(self) -> str:
        status = "ok" if self.ok else "refused: " + "; ".join(self.plan.get("problems", []))[:80]
        return (f"image angle {self.image_angle_deg:+5.0f} deg, {self.distance_m * 100:4.1f} cm, "
                f"max joint swing {self.max_joint_deg:5.1f} deg, {status}")


def camera_pose(calib: CalibrationStore, motors) -> np.ndarray:
    """base_link <- wrist optical frame, from the live arm ticks and the measured hand-eye mount."""
    arm = [m for m in motors if 1 <= m.id <= calib.num_arm_joints(len(motors))]
    return calib.fk_ee_pose(calib.ticks_to_rad(arm)) @ np.asarray(calib.data["T_ee_cam_wrist"], dtype=float)


def pixel_ray(K, dist, pixel) -> np.ndarray:
    """Unit ray (camera frame) through an original-image pixel."""
    ideal = cv2.undistortPoints(np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2), K, dist).ravel()
    ray = np.array([ideal[0], ideal[1], 1.0])
    return ray / np.linalg.norm(ray)


def table_point(T_base_cam, K, dist, pixel, table_z) -> tuple[np.ndarray, float]:
    """Intersect the pixel ray with the table plane; returns the base_link point and its optical depth."""
    ray_cam = pixel_ray(K, dist, pixel)
    ray_base = T_base_cam[:3, :3] @ ray_cam
    if ray_base[2] > -1e-3:
        raise ValueError("the target ray does not hit the table")
    lam = (table_z - T_base_cam[2, 3]) / ray_base[2]
    return T_base_cam[:3, 3] + lam * ray_base, float(lam * ray_cam[2])


def project(T_base_cam, K, dist, point_base) -> np.ndarray:
    """Pixel of a base_link point in the camera."""
    T_cam_base = np.linalg.inv(T_base_cam)
    p = T_cam_base[:3, :3] @ point_base + T_cam_base[:3, 3]
    return cv2.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)[0].ravel()


def max_slide_in_frame(roi, image_size, depth_m, focal_px, angle_deg, margin_px=40) -> float:
    """Longest camera slide (m) along ``angle_deg`` that keeps the ROI inside the second image.

    Moving the camera along +u shifts the scene towards -u by about f * B / Z pixels.
    """
    x, y, w, h = roi
    width, height = image_size
    du, dv = -math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))
    room = []
    for d, low, high, size in ((du, x, x + w, width), (dv, y, y + h, height)):
        if d > 1e-6:
            room.append((size - margin_px - high) / d)
        elif d < -1e-6:
            room.append((low - margin_px) / -d)
    shift_px = max(0.0, min(room)) if room else math.inf
    return shift_px * depth_m / focal_px


def plan_slides(calib: CalibrationStore, motors, T_base_cam, roi, image_size, depth_m, focal_px,
                distance_m: float, angles_deg=(0, 180, 30, -30, 150, -150), min_distance_m=0.01,
                shrink=0.7, limits: MotionLimits | None = None) -> list[SlideCandidate]:
    """Dry-run slides along image directions, shrinking the distance until the planner accepts them.

    Each angle is a direction in the camera's image plane (0 = image +u, i.e. sideways), mapped into
    base_link through the current camera orientation. Per angle the distance starts at ``distance_m``
    (capped so the ROI stays in frame) and shrinks by ``shrink`` while the plan is refused.
    """
    limits = limits or MotionLimits()
    candidates = []
    for angle in angles_deg:
        a = math.radians(angle)
        direction = T_base_cam[:3, :3] @ np.array([math.cos(a), math.sin(a), 0.0])
        distance = min(distance_m, max_slide_in_frame(roi, image_size, depth_m, focal_px, angle))
        while distance >= min_distance_m:
            plan = plan_flange_move(calib, motors, direction * distance, Rotation.identity(), "flange", limits)
            swing = math.inf
            if plan["ok"]:
                swing = float(np.max(np.abs(plan["summary"]["delta_ticks"])) / TICKS_PER_RAD * 180 / math.pi)
            candidates.append(SlideCandidate(angle, distance, direction, plan, swing))
            if plan["ok"]:
                break
            distance *= shrink
    return candidates


def rank_slides(candidates: list[SlideCandidate], max_joint_deg: float, distance_weight=0.5) -> list[SlideCandidate]:
    """Accepted slides under the joint-swing limit, best first: long baseline, small joint swing."""
    usable = [c for c in candidates if c.ok and c.max_joint_deg <= max_joint_deg]
    longest = max((c.distance_m for c in usable), default=1.0)
    return sorted(usable, key=lambda c: c.max_joint_deg / max_joint_deg - distance_weight * c.distance_m / longest)


# --- aligning the object to the image centre ---------------------------------------------------------------------

def object_in_base(calib: CalibrationStore, motors, xyz, frame: str = "base_link") -> np.ndarray:
    """Express a point given in ``frame`` in base_link.

    ``frame`` is ``base_link`` (the world frame), any URDF link (posed at the live joint angles),
    ``wrist_camera`` (OpenCV optical frame at the live pose) or ``top_camera`` (calibrated scene camera).
    """
    p = np.asarray(xyz, dtype=float).reshape(3)
    if frame in ("base_link", calib.world_frame()):
        return p
    if frame == "wrist_camera":
        T = camera_pose(calib, motors)
    elif frame == "top_camera":
        T = calib.T_world_cam("top")
        if T is None:
            raise ValueError("the top camera has no calibrated pose (T_world_cam_top)")
    else:
        arm = [m for m in motors if 1 <= m.id <= calib.num_arm_joints(len(motors))]
        T = calib.fk_ee_pose(calib.ticks_to_rad(arm), target_link=frame)
    return T[:3, :3] @ p + T[:3, 3]


def look_at_pose(target, tilt_deg, standoff_m, azimuth_deg, roll_deg, x_hint, centre_ray, heading=None) -> np.ndarray:
    """Camera pose whose image-centre ray passes through ``target`` from ``standoff_m`` away.

    The camera sits ``tilt_deg`` from straight above the target, on the side given by the horizontal
    ``heading`` (target towards camera; default towards the robot base) rotated by ``azimuth_deg`` about
    the vertical. Its image x axis is horizontal (sideways slides stay level), signed to agree with
    ``x_hint`` (the current camera x axis, avoiding wrist flips), then rolled by ``roll_deg`` about the ray.
    """
    target = np.asarray(target, dtype=float)
    up = np.array([0.0, 0.0, 1.0])
    side = -target * [1.0, 1.0, 0.0] if heading is None else np.array([heading[0], heading[1], 0.0])
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([0, 1.0, 0])
    side = Rotation.from_rotvec(up * math.radians(azimuth_deg)).apply(side)
    tilt = math.radians(tilt_deg)
    centre = target + standoff_m * (math.sin(tilt) * side + math.cos(tilt) * up)
    z = (target - centre) / standoff_m
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = x_hint - (x_hint @ z) * z
    x /= np.linalg.norm(x)
    if x @ x_hint < 0:
        x = -x
    R_look = np.column_stack([x, np.cross(z, x), z]) @ Rotation.from_rotvec([0, 0, math.radians(roll_deg)]).as_matrix()
    # Rotate so the image-centre ray (not the optical axis) points at the target.
    r = np.asarray(centre_ray, dtype=float) / np.linalg.norm(centre_ray)
    axis = np.cross([0.0, 0.0, 1.0], r)
    R_align = np.eye(3) if np.linalg.norm(axis) < 1e-9 else Rotation.from_rotvec(
        axis / np.linalg.norm(axis) * math.asin(min(1.0, np.linalg.norm(axis)))).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_look @ R_align.T
    T[:3, 3] = centre
    return T


@dataclass
class ViewPose:
    """A reachable camera pose that centres the target, with the joint move to get there."""

    q: np.ndarray
    T_base_cam: np.ndarray
    params: dict
    path: np.ndarray = field(repr=False)
    move_deg: float = 0.0
    lowest_m: float = 0.0
    tip_clearance_m: float = 0.0
    slide: SlideCandidate | None = None
    score: float = math.inf

    def describe(self) -> str:
        p = self.params
        slide = (f"best slide {self.slide.distance_m * 100:.1f} cm @ {self.slide.image_angle_deg:+.0f} deg, "
                 f"swing {self.slide.max_joint_deg:.1f} deg") if self.slide else "no slide"
        return (f"standoff {p['standoff_m'] * 100:.0f} cm, tilt {p['tilt_deg']:.0f}, azimuth {p['azimuth_deg']:+.0f}, "
                f"roll {p['roll_deg']:+.0f} | move {self.move_deg:.0f} deg, lowest {self.lowest_m * 100:.1f} cm, "
                f"tip-object {self.tip_clearance_m * 100:.1f} cm | {slide}")


def motors_at(calib: CalibrationStore, motors, q) -> list:
    """Copies of the motor states with arm positions set to joint angles ``q``."""
    ticks = calib.rad_to_ticks(list(q), motors)
    return [replace(m, position=int(ticks[m.id])) if m.id in ticks else m for m in motors]


def cylinder_distance(points, axis_xy, z_low, z_high, radius) -> np.ndarray:
    """Distance from points to an upright solid cylinder (0 inside)."""
    points = np.atleast_2d(points)
    radial = np.maximum(np.linalg.norm(points[:, :2] - np.asarray(axis_xy), axis=1) - radius, 0.0)
    vertical = np.maximum(np.maximum(z_low - points[:, 2], points[:, 2] - z_high), 0.0)
    return np.hypot(radial, vertical)


def path_clearance(calib: CalibrationStore, path, obstacle: dict) -> float:
    """Smallest distance from the jaw tips, flange or wrist camera to the ``obstacle`` cylinder along ``path``."""
    T_ee_cam = np.asarray(calib.data["T_ee_cam_wrist"], dtype=float)
    points = []
    for q in path:
        F = calib.fk_ee_pose(q)
        points += [F[:3, :3] @ TCP_IN_FLANGE + F[:3, 3], F[:3, 3], (F @ T_ee_cam)[:3, 3]]
    return float(cylinder_distance(np.array(points), **obstacle).min())


def joint_path(q0, q1, step_deg=2.0) -> np.ndarray:
    n = max(2, int(math.ceil(np.max(np.abs(np.degrees(q1 - q0))) / step_deg)) + 1)
    return np.array([q0 + s * (q1 - q0) for s in np.linspace(0.0, 1.0, n)])


_VIEW_SOLVERS: dict = {}


def _view_solver(urdf) -> TracIKSolver:
    # Look-at poses can be far from the seed; the shared planner solver's 5 ms budget is tuned for small steps.
    if urdf not in _VIEW_SOLVERS:
        _VIEW_SOLVERS[urdf] = TracIKSolver(urdf, timeout=0.015)
    return _VIEW_SOLVERS[urdf]


def view_grid(T_base_cam, target, standoffs_m=(0.18, 0.22, 0.26), tilts_deg=(0, 15, 30),
              azimuths_deg=(0, -20, 20, -40, 40), roll_offsets_deg=(0, -25, 25),
              standoff_range=(0.16, 0.30)) -> list[dict]:
    """Viewing geometries around the current one: the live standoff, tilt, side and roll first, then variations.

    Rolls are the live camera roll plus ``roll_offsets_deg``, and 0 (image x axis level).
    """
    target = np.asarray(target, dtype=float)
    offset = T_base_cam[:3, 3] - target
    standoff = float(np.clip(np.linalg.norm(offset), *standoff_range))
    tilt = math.degrees(math.acos(np.clip(offset[2] / np.linalg.norm(offset), -1, 1)))
    heading = tuple(offset[:2] / max(np.linalg.norm(offset[:2]), 1e-9))
    z = -offset / np.linalg.norm(offset)
    x_level = np.cross(z, [0.0, 0.0, 1.0])
    live_roll = 0.0
    if np.linalg.norm(x_level) > 1e-6:
        x_level /= np.linalg.norm(x_level)
        x_now = T_base_cam[:3, 0]
        x_level = x_level if x_level @ x_now >= 0 else -x_level
        live_roll = math.degrees(math.atan2(x_now @ np.cross(z, x_level), x_now @ x_level))
    rolls = list(dict.fromkeys(round(live_roll + r) for r in (*roll_offsets_deg, -live_roll)))
    grid, seen = [], set()
    for s, t, a, r in itertools.product((standoff, *standoffs_m), (tilt, *tilts_deg), azimuths_deg, rolls):
        key = (round(s, 3), round(t), 0 if t < 1 else a, r)
        if key not in seen:
            seen.add(key)
            grid.append(dict(standoff_m=s, tilt_deg=t, azimuth_deg=a, roll_deg=r, heading=heading))
    return grid


def plan_view_pose(calib: CalibrationStore, motors, target, K, dist, image_size, *,
                   limits: MotionLimits | None = None, min_tip_clearance_m=0.06, max_move_deg=100.0,
                   max_joint_deg=20.0, slide_parallax_deg=8.0, evaluate_top=8, grid: list[dict] | None = None,
                   fixed: dict | None = None, obstacle: dict | None = None,
                   obstacle_clearance_m=0.04) -> list[ViewPose]:
    """Search look-at poses that centre ``target`` and rank them by how well the stereo slide will work.

    Every candidate (``grid``, default :func:`view_grid` around the live view) must be reachable inside
    the motor ranges (with margin), move no joint more than ``max_move_deg``, keep the arm above the
    floor along the joint-space move, and keep the jaw tips ``min_tip_clearance_m`` from the target.
    With ``obstacle`` (an upright cylinder: ``axis_xy``, ``z_low``, ``z_high``, ``radius``, e.g. a can),
    the jaw tips, flange and camera must also stay ``obstacle_clearance_m`` from it along the move. The
    smallest moves are then scored by their best sideways slide (small joint swing, long baseline
    relative to the nominal one) plus a penalty for the move size. ``fixed`` restricts the search to one
    parameter set (used to re-centre with the same viewing geometry).
    """
    limits = limits or MotionLimits()
    solver = _view_solver(str(calib.urdf_path()))
    arm = [m for m in motors if 1 <= m.id <= calib.num_arm_joints(len(motors))]
    q0 = np.array(calib.ticks_to_rad(arm))
    lower, upper = allowed_bounds_rad(calib, motors)
    T_now = camera_pose(calib, motors)
    T_ee_cam = np.asarray(calib.data["T_ee_cam_wrist"], dtype=float)
    centre_px = (image_size[0] / 2, image_size[1] / 2)
    centre_ray = np.append(cv2.undistortPoints(np.array([[centre_px]], np.float64), K, dist).ravel(), 1.0)
    floor = min(limits.floor_cm / 100, lowest_point_m(calib, q0) - 0.002)

    grid = [fixed] if fixed else (grid or view_grid(T_now, target))
    feasible, solutions = [], [q0]
    for params in grid:
        T_cam = look_at_pose(target, x_hint=T_now[:3, 0], centre_ray=centre_ray, **params)
        T_flange = T_cam @ np.linalg.inv(T_ee_cam)
        best = None
        # Seed from the live pose, then from the closest-to-live solution found so far.
        seeds = [q0] + sorted(solutions[1:], key=lambda s: np.max(np.abs(s - q0)))[:1]
        for seed in seeds:
            # The raw solve skips the failure diagnosis of solve(), which dominates a search with many misses.
            q = solver._raw_solve(T_flange[:3, 3], T_flange[:3, :3], np.clip(seed, lower, upper), lower, upper)
            if q is not None and (best is None or np.max(np.abs(q - q0)) < np.max(np.abs(best - q0))):
                best = q
        if best is None:
            continue
        solutions.append(best)
        move = float(np.max(np.abs(np.degrees(best - q0))))
        flange = solver.fk(best)
        tip = flange[:3, :3] @ TCP_IN_FLANGE + flange[:3, 3]
        clearance = float(np.linalg.norm(tip - target))
        path = joint_path(q0, best)
        if move > max_move_deg or clearance < min_tip_clearance_m:
            continue
        if obstacle is not None and path_clearance(calib, path, obstacle) < obstacle_clearance_m:
            continue
        lowest = min(lowest_point_m(calib, q) for q in path)
        if lowest < floor:
            continue
        feasible.append(ViewPose(best, T_cam, params, path, move, lowest, clearance))

    feasible.sort(key=lambda v: v.move_deg)
    for view in feasible[:evaluate_top]:
        at = motors_at(calib, motors, view.q)
        roi = (int(centre_px[0]) - 80, int(centre_px[1]) - 80, 160, 160)
        depth = view.params["standoff_m"]
        distance = max(0.01, min(0.05, depth * math.tan(math.radians(slide_parallax_deg))))
        ranked = rank_slides(plan_slides(calib, at, view.T_base_cam, roi, image_size, depth, K[0, 0], distance,
                                         limits=limits), max_joint_deg)
        if obstacle is not None:
            ranked = [c for c in ranked if path_clearance(calib, c.plan["path"], obstacle) >= obstacle_clearance_m]
        view.slide = ranked[0] if ranked else None
        if view.slide is not None:
            view.score = (view.slide.max_joint_deg / max_joint_deg - view.slide.distance_m / distance
                          + view.move_deg / max_move_deg)
    return sorted(feasible[:evaluate_top], key=lambda v: v.score)


def _blob_mask(patch, mode):
    if mode == "color":
        # Dark pixels (checkerboard squares, shadows) have noisy saturation, so require some brightness too.
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 85)
        close, open_ = 9, 5
    else:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.int16)
        mask = np.median(gray) - gray > 35
        # Small kernels: a coin next to checkerboard squares must not merge with them.
        close, open_ = 5, 3
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_, open_), np.uint8))


def detect_object(image, hint, mode: str = "color", search_px: int = 160):
    """Find the object blob nearest ``hint`` within ``search_px``.

    ``color`` finds a saturated, bright object (e.g. a Rubik's cube); ``round`` a disc darker than the
    local background (e.g. a coin on paper). Returns ``(centre_xy, bbox_xywh)`` in image pixels, or None.
    """
    if mode not in ("color", "round"):
        return None
    hint = np.asarray(hint, dtype=float)
    h, w = image.shape[:2]
    x0, y0 = int(max(0, hint[0] - search_px)), int(max(0, hint[1] - search_px))
    x1, y1 = int(min(w, hint[0] + search_px)), int(min(h, hint[1] + search_px))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    mask = _blob_mask(image[y0:y1, x0:x1], mode)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, best_score = None, -math.inf
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 150 or area > 0.6 * mask.size:
            continue
        m = cv2.moments(contour)
        centre = np.array([m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0])
        distance = float(np.linalg.norm(centre - hint))
        if distance > search_px:
            continue
        score = math.log(area) - 3 * distance / search_px
        if mode == "round":
            fill = area / (math.pi * cv2.minEnclosingCircle(contour)[1] ** 2)
            if fill < 0.75:
                continue
            score += 3 * fill
        if score > best_score:
            bx, by, bw, bh = cv2.boundingRect(contour)
            best, best_score = (centre, (bx + x0, by + y0, bw, bh)), score
    return best


def _static_mask(gray_a, gray_b, threshold=6) -> np.ndarray:
    """Pixels identical in both views and connected to the bottom edge: the gripper, fixed to the camera."""
    diff = cv2.GaussianBlur(cv2.absdiff(gray_a, gray_b), (9, 9), 0) < threshold
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels = cv2.connectedComponents(diff)
    bottom = set(np.unique(labels[-1])) - {0}
    mask = np.isin(labels, list(bottom)) if bottom else np.zeros_like(diff, dtype=bool)
    return cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)


def _sampson(params, xa, xb):
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    theta, phi = params[3:]
    t = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
    E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
    Ex, Etx = xa @ E.T, xb @ E
    return np.sum(xb * Ex, axis=1) / np.sqrt(Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2)


def _polish_relative_pose(R, t, na, nb, focal_px):
    """Least-squares Sampson refinement of (R, unit t) over normalized inlier correspondences."""
    from scipy.optimize import least_squares

    t = t / np.linalg.norm(t)
    x0 = np.r_[Rotation.from_matrix(R).as_rotvec(), math.acos(np.clip(t[2], -1, 1)), math.atan2(t[1], t[0])]
    xa, xb = np.c_[na, np.ones(len(na))], np.c_[nb, np.ones(len(nb))]
    fit = least_squares(_sampson, x0, args=(xa, xb), loss="soft_l1", f_scale=1.0 / focal_px)
    theta, phi = fit.x[3:]
    return (Rotation.from_rotvec(fit.x[:3]).as_matrix(),
            np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]))


def refine_relative_pose(frame_a, frame_b, K, dist, rel_arm, max_rotation_diff_deg=6.0,
                         max_direction_diff_deg=30.0, static_px=3.0) -> tuple[np.ndarray, dict]:
    """Relative pose from whole-image feature matches, scaled by the arm's baseline length.

    The images fix rotation and translation direction far better than the arm's kinematic model; the
    encoders still provide metric scale. Falls back to ``rel_arm`` when the image estimate is weak or
    disagrees with the arm beyond the given limits.
    """
    gray_a, gray_b = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (frame_a, frame_b))
    sift = cv2.SIFT_create(nfeatures=5000)
    keys_a, desc_a = sift.detectAndCompute(gray_a, None)
    keys_b, desc_b = sift.detectAndCompute(gray_b, None)
    info = {"source": "arm"}
    if desc_a is None or desc_b is None:
        info["reason"] = "no features"
        return rel_arm, info
    pairs = [m for m, n in cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2) if m.distance < 0.75 * n.distance]
    a = np.float64([keys_a[m.queryIdx].pt for m in pairs]).reshape(-1, 2)
    b = np.float64([keys_b[m.trainIdx].pt for m in pairs]).reshape(-1, 2)
    moving = np.linalg.norm(a - b, axis=1) > static_px
    a, b = a[moving], b[moving]
    info["matches"] = int(len(a))
    if len(a) < 30:
        info["reason"] = f"only {len(a)} moving matches"
        return rel_arm, info
    na = cv2.undistortPoints(a.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    nb = cv2.undistortPoints(b.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    E, inliers = cv2.findEssentialMat(na, nb, np.eye(3), cv2.USAC_MAGSAC, 0.9999, 0.5 / K[0, 0])
    if E is None or E.shape != (3, 3):
        info["reason"] = "essential matrix failed"
        return rel_arm, info
    _, R, t, inliers = cv2.recoverPose(E, na, nb, np.eye(3), mask=inliers)
    keep = inliers.ravel() > 0
    R, t = _polish_relative_pose(R, t.ravel(), na[keep], nb[keep], K[0, 0])
    t_arm = rel_arm[:3, 3]
    rot_diff = math.degrees((Rotation.from_matrix(R) * Rotation.from_matrix(rel_arm[:3, :3]).inv()).magnitude())
    dir_diff = math.degrees(math.acos(np.clip(t @ t_arm / np.linalg.norm(t_arm), -1, 1)))
    info.update({"inliers": int(inliers.sum()), "rotation_diff_deg": round(rot_diff, 2),
                 "direction_diff_deg": round(dir_diff, 1),
                 "arm_rotation_deg": round(math.degrees(Rotation.from_matrix(rel_arm[:3, :3]).magnitude()), 2),
                 "image_rotation_deg": round(math.degrees(Rotation.from_matrix(R).magnitude()), 2)})
    if inliers.sum() < 30 or rot_diff > max_rotation_diff_deg or dir_diff > max_direction_diff_deg:
        info["reason"] = "image pose disagrees with the arm or has too few inliers"
        return rel_arm, info
    rel = np.eye(4)
    rel[:3, :3] = R
    rel[:3, 3] = t * np.linalg.norm(t_arm)
    info["source"] = "image"
    return rel, info


def fit_table_plane(points, tol_m=0.003, max_tilt_deg=25.0, iterations=600, seed=0):
    """RANSAC plane with a near-vertical normal (the table); returns (unit normal with +z, offset, inlier mask).

    The plane is ``normal @ p == offset``. Returns None when no plane with enough support is found.
    """
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(len(points), 20000), replace=False)]
    best, best_count = None, 0
    min_nz = math.cos(math.radians(max_tilt_deg))
    for _ in range(iterations):
        p0, p1, p2 = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm * np.sign(normal[2] or 1.0)
        if normal[2] < min_nz:
            continue
        count = int((np.abs(sample @ normal - normal @ p0) < tol_m).sum())
        if count > best_count:
            best, best_count = (normal, float(normal @ p0)), count
    if best is None or best_count < 0.1 * len(sample):
        return None
    inliers = np.abs(points @ best[0] - best[1]) < tol_m
    centroid = points[inliers].mean(axis=0)
    normal = np.linalg.svd(points[inliers] - centroid, full_matrices=False)[2][2]
    normal *= np.sign(normal[2])
    inliers = np.abs((points - centroid) @ normal) < tol_m
    return normal, float(normal @ centroid), inliers


def table_alignment(normal, offset, camera_centre, table_z):
    """Similarity about the camera centre that makes the fitted table horizontal at ``table_z``.

    Rotates the fitted normal onto +z (smallest rotation) and scales so the camera's distance to the plane
    equals its arm-derived height above the table. Returns ``(scale, R_fix, info)``; apply with
    ``c + scale * R_fix @ (p - c)``.
    """
    measured = float(normal @ camera_centre - offset)
    height = float(camera_centre[2] - table_z)
    axis = np.cross(normal, [0.0, 0.0, 1.0])
    angle = math.asin(min(1.0, float(np.linalg.norm(axis))))
    R_fix = np.eye(3) if angle < 1e-9 else Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
    return height / measured, R_fix, {"tilt_corrected_deg": round(math.degrees(angle), 2),
                                      "camera_to_plane_m": round(measured, 4), "arm_height_m": round(height, 4)}


def apply_alignment(points, camera_centre, scale, R_fix):
    return camera_centre + scale * (np.asarray(points) - camera_centre) @ R_fix.T


def align_pose(T, camera_centre, scale, R_fix):
    """Apply the table alignment to a camera pose."""
    out = np.eye(4)
    out[:3, :3] = R_fix @ T[:3, :3]
    out[:3, 3] = apply_alignment(T[:3, 3], camera_centre, scale, R_fix)
    return out


def dense_cloud(frame_a, frame_b, K, dist, T_base_cam_a, rel, depth_range_m=(0.08, 0.8), step=2):
    """Dense colored point cloud from SGBM on the rectified pair, in base_link.

    Returns ``(points_base, colors_rgb01, info)``; the left view of the rectified pair is whichever camera
    is on the left, and the gripper (static in both views) is masked out.
    """
    size = (frame_a.shape[1], frame_a.shape[0])
    R, t = rel[:3, :3], rel[:3, 3]
    centre_b = -R.T @ t
    if centre_b[0] >= 0:
        left, right, R_lr, t_lr, T_base_left = frame_a, frame_b, R, t, T_base_cam_a
    else:
        left, right = frame_b, frame_a
        R_lr, t_lr = R.T, -R.T @ t
        T_base_left = T_base_cam_a @ np.linalg.inv(rel)
    dist = None if dist is None else np.asarray(dist, dtype=np.float64).reshape(1, -1)
    R_lr = np.ascontiguousarray(R_lr)
    t_lr = np.asarray(t_lr, dtype=np.float64).reshape(3, 1)
    # alpha=1 keeps the whole field of view; enlarge the canvas so the rectified focal stays near native.
    P1 = cv2.stereoRectify(K, dist, K, dist, size, R_lr, t_lr, alpha=1, flags=cv2.CALIB_ZERO_DISPARITY)[2]
    grow = float(np.clip(K[0, 0] / P1[0, 0], 1.0, 2.0))
    rect_size = (int(size[0] * grow), int(size[1] * grow))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, dist, K, dist, size, R_lr, t_lr, alpha=1,
                                                newImageSize=rect_size, flags=cv2.CALIB_ZERO_DISPARITY)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        return np.empty((0, 3)), np.empty((0, 3)), {"reason": "slide is mostly vertical in the image"}
    maps = [cv2.initUndistortRectifyMap(K, dist, Rr, Pr, rect_size, cv2.CV_32FC1) for Rr, Pr in ((R1, P1), (R2, P2))]
    rect_l = cv2.remap(left, *maps[0], cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, *maps[1], cv2.INTER_LINEAR)
    static = _static_mask(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
    ones = np.ones(left.shape[:2], np.uint8)
    static_rect = cv2.remap(static.astype(np.uint8), *maps[0], cv2.INTER_NEAREST).astype(bool)
    static_rect |= cv2.remap(ones, *maps[0], cv2.INTER_NEAREST, borderValue=0) == 0

    focal, baseline = P1[0, 0], float(np.linalg.norm(t_lr))
    d_min = int(math.floor(focal * baseline / depth_range_m[1]))
    n_disp = int(math.ceil((focal * baseline / depth_range_m[0] - d_min) / 16) * 16)
    n_disp = max(16, min(n_disp, 16 * 24))
    block = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=d_min, numDisparities=n_disp, blockSize=block,
                                 P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = sgbm.compute(cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    xyz = cv2.reprojectImageTo3D(disparity, Q)
    valid = (disparity > d_min) & ~static_rect & np.isfinite(xyz).all(axis=2)
    valid &= (xyz[:, :, 2] > depth_range_m[0]) & (xyz[:, :, 2] < depth_range_m[1])
    keep = np.zeros_like(valid)
    keep[::step, ::step] = valid[::step, ::step]
    points_rect = xyz[keep]
    colors = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)[keep].astype(np.float64) / 255.0
    points_left = points_rect @ R1          # rectified -> left camera (R1 rotates left camera into rectified)
    points_base = points_left @ T_base_left[:3, :3].T + T_base_left[:3, 3]
    info = {"points": int(len(points_base)), "baseline_m": round(baseline, 4), "disparity_range": [d_min, d_min + n_disp],
            "left_view": "A" if left is frame_a else "B", "valid_fraction": round(float(valid.mean()), 3)}
    return points_base, colors, info


def write_ply(path, points, colors) -> None:
    """Binary little-endian PLY with float xyz and uchar rgb (colors in [0, 1])."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = (np.clip(np.asarray(colors, dtype=np.float64).reshape(-1, 3), 0, 1) * 255).astype(np.uint8)
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as stream:
        stream.write(header.encode())
        stream.write(vertex.tobytes())
