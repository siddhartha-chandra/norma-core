"""Segmentation and metric cuboid geometry shared by the stereo workflow and measurement CLI."""

import cv2
import numpy as np

def segment(image, box, open_px, dark_v_max=0):
    """GrabCut mask of the object inside ``box`` (x, y, w, h), with thin parts opened away.

    ``dark_v_max`` > 0 (a dark object) also requires HSV value below it, with holes (glossy highlights)
    filled: GrabCut alone confuses a black object with adjacent dark checkerboard squares.
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, tuple(int(v) for v in box), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    if dark_v_max > 0:
        dark = (cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2] < dark_v_max).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        fg &= dark
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(fg)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return out.astype(bool)


def fit_cuboid(points, table_z, trim_pct, top_pct):
    """Table-aligned cuboid: footprint = min-area rectangle of the trimmed XY points, from the table up."""
    xy = points[:, :2]
    centre = np.median(xy, axis=0)
    radius = np.linalg.norm(xy - centre, axis=1)
    core = points[radius <= np.percentile(radius, 100 - trim_pct)]
    (cx, cy), (a, b), angle = cv2.minAreaRect((core[:, :2] * 1000).astype(np.float32))
    top = float(np.percentile(core[:, 2], top_pct))
    footprint = cv2.boxPoints(((cx, cy), (a, b), angle)) / 1000
    corners = np.vstack([np.column_stack([footprint, np.full(4, table_z)]), np.column_stack([footprint, np.full(4, top)])])
    length, width = max(a, b), min(a, b)
    yaw = angle if a >= b else angle + 90
    return corners, {"length_mm": round(length, 1), "width_mm": round(width, 1),
                     "height_mm": round((top - table_z) * 1000, 1), "yaw_deg": round(yaw % 180, 1),
                     "centre_base_mm": [round(cx, 1), round(cy, 1), round((table_z + top) / 2 * 1000, 1)],
                     "points_used": int(len(core))}

