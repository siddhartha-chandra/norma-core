#!/usr/bin/env python3
"""Render a wrist stereo sweep's point clouds with Open3D.

Reads ``cloud.ply`` (dense, camera colors), ``sparse.ply`` (epipolar matches on the target) and
``stereo_result.json`` (camera poses, table height) from a sweep attempt directory, and saves
``cloud_rgb.png`` and ``cloud_height.png``. ``--show`` opens an interactive window instead of / as well.

    python scripts/plot_stereo_cloud.py calib_frames/stereo_cube/attempt_1 [--show] [--color height]
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402


def frustum(T_base_cam, color, size=0.03):
    """Wire frustum of a camera pose (optical frame, +Z forward)."""
    corners = np.array([[0, 0, 0], [-1, -0.6, 1.2], [1, -0.6, 1.2], [1, 0.6, 1.2], [-1, 0.6, 1.2]]) * size
    pts = corners @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    mesh = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
    mesh.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return mesh


def height_colors(points, table_z, top_m):
    heights = np.clip((points[:, 2] - table_z) / top_m, 0, 1)
    return colormaps["turbo"](heights)[:, :3]


def build(directory, color_mode, voxel_m):
    result = json.loads((directory / "stereo_result.json").read_text())
    dense = o3d.io.read_point_cloud(str(directory / "cloud.ply"))
    if voxel_m > 0:
        dense = dense.voxel_down_sample(voxel_m)
    dense, _ = dense.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    table_z = result.get("table_z_m") or 0.0
    points = np.asarray(dense.points)
    if color_mode == "height" and len(points):
        top = max(0.02, float(np.percentile(points[:, 2] - table_z, 99.5)))
        dense.colors = o3d.utility.Vector3dVector(height_colors(points, table_z, top))
    geometries = [dense, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)]
    sparse_path = directory / "sparse.ply"
    if sparse_path.exists():
        sparse = o3d.io.read_point_cloud(str(sparse_path))
        for p in np.asarray(sparse.points):
            ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
            ball.translate(p)
            ball.paint_uniform_color([1.0, 0.0, 1.0])
            geometries.append(ball)
    for key, color in (("T_base_cam_a", [0.1, 0.9, 0.1]), ("T_base_cam_b", [0.1, 0.4, 1.0])):
        if key in result:
            geometries.append(frustum(np.array(result[key]), color))
    return geometries, result


def view(result, geometries, elevation_deg=35.0, distance_m=0.42):
    """Oblique view of the target (median sparse point, else the cloud centre) from camera A's side."""
    points = np.array(result.get("points_base_m") or [])
    centre = np.median(points, axis=0) if len(points) else geometries[0].get_center()
    cam = np.array(result["T_base_cam_a"])[:3, 3] if "T_base_cam_a" in result else centre + [0, 0.3, 0.3]
    horizontal = (cam - centre) * [1, 1, 0]
    horizontal /= max(np.linalg.norm(horizontal), 1e-6)
    e = np.radians(elevation_deg)
    eye = centre + distance_m * (np.cos(e) * horizontal + [0, 0, np.sin(e)])
    return centre, eye, np.array([0.0, 0.0, 1.0])


def render_png(geometries, centre, eye, up, out, width=1400, height=900):
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([0.06, 0.07, 0.09, 1.0])
        for i, g in enumerate(geometries):
            material = o3d.visualization.rendering.MaterialRecord()
            if isinstance(g, o3d.geometry.LineSet):
                material.shader, material.line_width = "unlitLine", 3.0
            else:
                material.shader, material.point_size = "defaultUnlit", 3.0
            renderer.scene.add_geometry(f"g{i}", g, material)
        renderer.setup_camera(55.0, centre, eye, up)
        renderer.render_to_image()  # the first offscreen frame can come back empty
        o3d.io.write_image(str(out), renderer.render_to_image())
        return "offscreen"
    except Exception:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        for g in geometries:
            vis.add_geometry(g)
        option = vis.get_render_option()
        option.background_color, option.point_size = np.array([0.06, 0.07, 0.09]), 3.0
        control = vis.get_view_control()
        control.set_lookat(centre)
        control.set_front((eye - centre) / np.linalg.norm(eye - centre))
        control.set_up(up)
        control.set_zoom(0.55)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out), do_render=True)
        vis.destroy_window()
        return "visualizer"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="sweep attempt directory with cloud.ply and stereo_result.json")
    parser.add_argument("--color", choices=("both", "rgb", "height"), default="both")
    parser.add_argument("--voxel-mm", type=float, default=1.5, help="voxel size for downsampling (0 = off)")
    parser.add_argument("--show", action="store_true", help="open an interactive Open3D window")
    args = parser.parse_args()

    modes = ("rgb", "height") if args.color == "both" else (args.color,)
    for mode in modes:
        geometries, result = build(args.directory, mode, args.voxel_mm / 1000)
        centre, eye, up = view(result, geometries)
        out = args.directory / f"cloud_{mode}.png"
        backend = render_png(geometries, centre, eye, up, out)
        print(f"saved {out} ({backend}, {len(geometries[0].points)} points)")
    if args.show:
        geometries, _ = build(args.directory, modes[-1], args.voxel_mm / 1000)
        o3d.visualization.draw_geometries(geometries, window_name=f"wrist stereo: {args.directory.name}")


if __name__ == "__main__":
    main()
