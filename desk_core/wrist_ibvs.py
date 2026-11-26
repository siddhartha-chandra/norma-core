"""Local image feedback for a flat, elongated marker at the calibrated table plane."""
from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .pose_estimation import _wrap, pixels_on_plane
from .stereo_sweep import _blob_mask


PERIOD_S = 0.1
TIMEOUT_S = 1.5
YAW_TOL_RAD = math.radians(2)
MAX_START_ERROR_M = 0.035
MAX_XY_STEP_M = 0.012
MAX_ROLL_STEP_RAD = math.radians(15)


@dataclass(frozen=True)
class Alignment:
    error_mm: float
    yaw_error_rad: float
    position: np.ndarray
    rotation: Rotation


def _marker_moments(frame, center, K, dist, camera_pose, plane_z, optical_xy):
    mask = _blob_mask(frame, "color")
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    h, w = mask.shape
    candidates = []
    for index, contour in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if not 150 <= area <= 0.4 * h * w or x == 0 or y == 0 or x + width == w or y + height == h:
            continue
        m = cv2.moments(contour)
        uv = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
        distance = np.linalg.norm(uv - center)
        if distance <= min(w, h) / 3:
            candidates.append((distance, index))
    if not candidates:
        raise ValueError("No complete colored marker near the wrist optical center.")
    _, index = min(candidates, key=lambda item: item[0])

    def area_moments(contour):
        xy = pixels_on_plane(contour.reshape(-1, 2), K, dist, camera_pose, plane_z)[:, :2] - optical_xy
        m = cv2.moments(xy.astype(np.float32))
        return np.array([m[name] for name in ("m00", "m10", "m01", "m20", "m11", "m02")])

    moments = area_moments(contours[index])
    hole = hierarchy[0, index, 2]
    while hole != -1:
        moments -= area_moments(contours[hole])
        hole = hierarchy[0, hole, 0]
    area, sx, sy, sxx, sxy, syy = moments
    if area <= 0:
        raise ValueError("Marker has no positive area on the table plane.")
    center_xy = np.array([sx, sy]) / area
    covariance = np.array([[sxx, sxy], [sxy, syy]]) / area - np.outer(center_xy, center_xy)
    return optical_xy + center_xy, covariance


def alignment_step(frame, K, dist, camera_pose, flange_pose, plane_z) -> Alignment:
    """Compute a bounded XY correction and roll using a measured flange and optical pose."""
    center = K[:2, 2]
    optical_xy = pixels_on_plane([center], K, dist, camera_pose, plane_z)[0, :2]
    target_xy, covariance = _marker_moments(frame, center, K, dist, camera_pose, plane_z, optical_xy)
    variances, axes = np.linalg.eigh(covariance)
    if variances[-1] <= 2.25 * variances[0]:
        raise ValueError("Marker yaw is ambiguous; use an elongated marker.")
    axis = axes[:, -1]
    rotation = flange_pose[:3, :3]
    if rotation[2, 1] > -0.8:
        raise ValueError("Start wrist alignment at a downward gripper hover pose.")
    error_m = np.linalg.norm(target_xy - optical_xy)
    if error_m > MAX_START_ERROR_M:
        raise ValueError("Marker is outside the final-approach alignment range.")
    # The ElRobot jaws close along flange X and roll about flange Y.
    jaw_xy = rotation[:2, 0]
    target_yaw = math.atan2(axis[1], axis[0]) + math.pi / 2
    yaw_error = _wrap(target_yaw - math.atan2(jaw_xy[1], jaw_xy[0]), math.pi)
    roll = _wrap(math.atan2(-axis @ jaw_xy, -axis @ rotation[:2, 2]), math.pi)
    if abs(yaw_error) <= YAW_TOL_RAD:
        roll = 0.0
    next_rotation = Rotation.from_matrix(rotation) * Rotation.from_rotvec(
        [0.0, np.clip(roll, -MAX_ROLL_STEP_RAD, MAX_ROLL_STEP_RAD), 0.0]
    )

    # Predict the optical-axis shift from roll, including the wrist camera's offset.
    position = flange_pose[:3, 3].copy()
    next_camera = camera_pose.copy()
    delta_rotation = next_rotation.as_matrix() @ rotation.T
    next_camera[:3, :3] = delta_rotation @ camera_pose[:3, :3]
    next_camera[:3, 3] = position + delta_rotation @ (camera_pose[:3, 3] - position)
    next_xy = pixels_on_plane([center], K, dist, next_camera, plane_z)[0, :2]
    correction = target_xy - next_xy
    distance = np.linalg.norm(correction)
    if distance > MAX_XY_STEP_M:
        correction *= MAX_XY_STEP_M / distance
    position[:2] += correction
    return Alignment(float(error_m * 1000), yaw_error, position, next_rotation)
