"""desk_core.stereo_sweep: centring the object of interest before the stereo slide (offline, no Station)."""
import json

import cv2
import numpy as np
import pytest

from desk_core.stereo_sweep import (camera_pose, cylinder_distance, detect_object, look_at_pose, object_in_base,
                                    pixel_ray, plan_view_pose, project, view_grid)
from desk_core.wrist_stereo import DEFAULT_WRIST_INTRINSICS


@pytest.fixture(scope="module")
def wrist():
    data = json.loads(DEFAULT_WRIST_INTRINSICS.read_text(encoding="utf-8"))["wrist"]
    return np.array(data["camera_matrix"]), np.array(data["distortion_coefficients"]), data["image_size"]


def centre_ray(K, dist, size):
    return pixel_ray(K, dist, (size[0] / 2, size[1] / 2))


@pytest.mark.parametrize("tilt, azimuth, roll", [(0, 0, 0), (20, 0, 0), (30, -40, 0), (15, 20, 20)])
def test_look_at_pose_puts_target_on_image_centre(wrist, tilt, azimuth, roll):
    K, dist, size = wrist
    target = np.array([0.02, -0.19, 0.05])
    T = look_at_pose(target, tilt, 0.22, azimuth, roll, x_hint=np.array([1.0, 0, 0]), centre_ray=centre_ray(K, dist, size))
    assert np.allclose(project(T, K, dist, target), np.array(size) / 2, atol=0.5)
    assert np.linalg.norm(T[:3, 3] - target) == pytest.approx(0.22)
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-9) and np.linalg.det(T[:3, :3]) == pytest.approx(1)
    if roll == 0:
        # Within the ~1 deg the image centre sits off the principal point.
        assert abs(T[2, 0]) < 0.03, "image x axis should stay level"
        assert T[0, 0] > 0, "x axis sign follows the hint"


def test_object_in_base_from_robot_frames(calib, front_motors):
    p = np.array([0.01, -0.2, 0.03])
    assert np.allclose(object_in_base(calib, front_motors, p, "base_link"), p)
    T_cam = camera_pose(calib, front_motors)
    assert np.allclose(object_in_base(calib, front_motors, [0, 0, 0.25], "wrist_camera"), T_cam[:3, 3] + 0.25 * T_cam[:3, 2])
    flange = calib.fk_ee_pose(calib.ticks_to_rad(front_motors[:7]))
    assert np.allclose(object_in_base(calib, front_motors, [0, 0, 0], calib.wrist_mount_link()), flange[:3, 3])


def test_cylinder_distance_for_an_upright_can():
    can = dict(axis_xy=(0.1, -0.2), z_low=0.0, z_high=0.14, radius=0.03)
    points = [[0.1, -0.2, 0.07], [0.15, -0.2, 0.07], [0.1, -0.2, 0.2], [0.17, -0.2, 0.17]]
    assert np.allclose(cylinder_distance(points, **can), [0.0, 0.02, 0.06, 0.05])


def test_detectors_find_the_object_near_the_hint():
    image = np.full((720, 1280, 3), 225, np.uint8)
    cv2.rectangle(image, (300, 300), (330, 330), (40, 40, 40), -1)      # dark square (checker-like)
    cv2.rectangle(image, (600, 300), (700, 400), (30, 30, 220), -1)     # saturated "cube"
    cv2.circle(image, (360, 500), 20, (90, 100, 110), -1)               # dark disc ("coin")
    centre, bbox = detect_object(image, (630, 340), "color")
    assert np.allclose(centre, (650, 350), atol=2) and bbox == (600, 300, 101, 101)
    centre, _ = detect_object(image, (340, 470), "round")
    assert np.allclose(centre, (360, 500), atol=2), "should pick the disc, not the square"
    assert detect_object(image, (1000, 600), "color") is None
    assert detect_object(image, (650, 350), "none") is None


def test_view_pose_centres_a_reachable_target(calib, front_motors, wrist):
    K, dist, size = wrist
    T_now = camera_pose(calib, front_motors)
    target = T_now[:3, 3] + 0.22 * (T_now[:3, :3] @ centre_ray(K, dist, size))
    grid = view_grid(T_now, target)
    assert grid[0]["standoff_m"] == pytest.approx(0.22)
    views = plan_view_pose(calib, front_motors, target, K, dist, size, grid=grid[:1])
    assert views, "the live viewing geometry itself must be reachable"
    assert views[0].move_deg < 1.0, "the live view needs (almost) no move"
    views = plan_view_pose(calib, front_motors, target, K, dist, size, grid=grid[:12])
    best = views[0]
    T_cam = calib.fk_ee_pose(best.q) @ np.asarray(calib.data["T_ee_cam_wrist"], dtype=float)
    assert np.allclose(project(T_cam, K, dist, target), np.array(size) / 2, atol=3)
    assert best.move_deg < 40 and best.tip_clearance_m >= 0.06
    assert np.allclose(best.path[0], calib.ticks_to_rad(front_motors[:7])) and np.allclose(best.path[-1], best.q)
