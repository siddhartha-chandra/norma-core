"""Project the collision scene and a preflighted flange path into the top camera."""
from itertools import product

import cv2
import numpy as np

from .interfaces import SceneObstacle


def obstacle_edges(obstacle: SceneObstacle):
    center = np.array([obstacle.x, obstacle.y, obstacle.z])
    if obstacle.shape == "box" and obstacle.size_xyz is not None:
        signs = list(product((-1, 1), repeat=3))
        vertices = center + np.array(signs) * np.asarray(obstacle.size_xyz) / 2
        for i, start in enumerate(signs):
            for j in range(i + 1, len(signs)):
                if sum(a != b for a, b in zip(start, signs[j])) == 1:
                    yield vertices[i], vertices[j]
    elif obstacle.shape == "cylinder":
        angles = np.arange(32) * (2 * np.pi / 32)
        xy = obstacle.radius_m * np.column_stack((np.cos(angles), np.sin(angles)))
        bottom, top = (center + np.column_stack((xy, np.full(32, dz)))
                       for dz in (-obstacle.height_m / 2, obstacle.height_m / 2))
        for ring in (bottom, top):
            yield from zip(ring, np.roll(ring, 1, axis=0))
        for i in range(0, len(top), 8):
            yield bottom[i], top[i]


def overlay_jpeg(jpeg, project, obstacles, path):
    if not obstacles and len(path) < 2:
        return jpeg
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jpeg
    height, width = image.shape[:2]

    def point(xyz):
        uv = project(xyz)
        if uv is None or not np.isfinite(uv).all():
            return None
        return tuple(round(float(v)) for v in np.clip(uv, -1000000, 1000000))

    def line(a, b, color, thickness):
        if a is not None and b is not None:
            visible, a, b = cv2.clipLine((0, 0, width, height), a, b)
            if visible:
                cv2.line(image, a, b, color, thickness, cv2.LINE_AA)
                return True
        return False

    drawn = False
    for obstacle in obstacles:
        for a, b in obstacle_edges(obstacle):
            drawn |= line(point(a), point(b), (40, 40, 240), 2)
    if len(path) >= 2:
        pixels = [point(p) for p in path]
        for a, b in zip(pixels, pixels[1:]):
            drawn |= line(a, b, (30, 195, 25), 3)
    if not drawn:
        return jpeg
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return encoded.tobytes() if ok else jpeg
