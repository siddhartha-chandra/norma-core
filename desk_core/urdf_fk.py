"""Minimal pure-NumPy forward kinematics parsed directly from URDF XML.

Computes exact link transforms ``T_world_link(q_rad)`` on
``hardware/elrobot/simulation/elrobot_follower.urdf`` (or any URDF) without
requiring Pinocchio/MuJoCo at import time.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .station_bridge import REPO_ROOT

DEFAULT_ELROBOT_URDF = REPO_ROOT / "hardware" / "elrobot" / "simulation" / "elrobot_follower.urdf"
DEFAULT_ARM_JOINTS = (
    "rev_motor_01",
    "rev_motor_02",
    "rev_motor_03",
    "rev_motor_04",
    "rev_motor_05",
    "rev_motor_06",
    "rev_motor_07",
)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF extrinsic XYZ (= intrinsic ZYX) rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _axis_angle_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    if abs(theta) < 1e-12:
        return np.eye(3, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    u = axis / norm
    c, s = math.cos(theta), math.sin(theta)
    K = np.array(
        [
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0],
        ],
        dtype=float,
    )
    return c * np.eye(3, dtype=float) + s * K + (1.0 - c) * np.outer(u, u)


def _make_tf(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass(frozen=True)
class URDFJoint:
    name: str
    jtype: str
    parent: str
    child: str
    origin_T: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None


class URDFModel:
    """Lightweight URDF kinematic tree supporting fixed/revolute/continuous/prismatic joints."""

    def __init__(self, urdf_path: Path = DEFAULT_ELROBOT_URDF):
        self.urdf_path = Path(urdf_path)
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        self.robot_name = root.attrib.get("name", "")
        self.joints_by_child: dict[str, URDFJoint] = {}
        self.joints_by_name: dict[str, URDFJoint] = {}

        for je in root.findall("joint"):
            name = je.attrib["name"]
            jtype = je.attrib.get("type", "fixed")
            parent = je.find("parent").attrib["link"]  # type: ignore[union-attr]
            child = je.find("child").attrib["link"]  # type: ignore[union-attr]

            orig = je.find("origin")
            if orig is not None:
                xyz = [float(v) for v in orig.attrib.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in orig.attrib.get("rpy", "0 0 0").split()]
            else:
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
            origin_T = _make_tf(_rpy_matrix(*rpy), np.array(xyz, dtype=float))

            ax_el = je.find("axis")
            axis = np.array([float(v) for v in (ax_el.attrib.get("xyz", "1 0 0") if ax_el is not None else "1 0 0").split()], dtype=float)

            lim_el = je.find("limit")
            lower = float(lim_el.attrib["lower"]) if (lim_el is not None and "lower" in lim_el.attrib) else None
            upper = float(lim_el.attrib["upper"]) if (lim_el is not None and "upper" in lim_el.attrib) else None

            joint = URDFJoint(name=name, jtype=jtype, parent=parent, child=child, origin_T=origin_T, axis=axis, lower=lower, upper=upper)
            self.joints_by_child[child] = joint
            self.joints_by_name[name] = joint

    def chain(self, target_link: str, base_link: str = "base_link") -> list[URDFJoint]:
        links_visited: set[str] = set()
        chain_rev: list[URDFJoint] = []
        curr = target_link
        while curr != base_link:
            if curr in links_visited:
                raise ValueError(f"Cycle detected in URDF while resolving {base_link} -> {target_link}")
            links_visited.add(curr)
            j = self.joints_by_child.get(curr)
            if j is None:
                raise ValueError(f"Link {target_link!r} is not reachable from {base_link!r} in {self.urdf_path}")
            chain_rev.append(j)
            curr = j.parent
        chain_rev.reverse()
        return chain_rev

    def fk(
        self,
        q_rad: list[float] | np.ndarray,
        target_link: str = "Gripper_Base_v1_1",
        base_link: str = "base_link",
        joint_names: list[str] | tuple[str, ...] = DEFAULT_ARM_JOINTS,
    ) -> np.ndarray:
        """Return 4x4 transform ``T_base_target`` mapping points in ``target_link`` into ``base_link``."""
        q_map = {name: float(val) for name, val in zip(joint_names, q_rad)}
        T = np.eye(4, dtype=float)
        for j in self.chain(target_link=target_link, base_link=base_link):
            T = T @ j.origin_T
            if j.jtype in {"revolute", "continuous"}:
                theta = q_map.get(j.name, 0.0)
                R = _axis_angle_matrix(j.axis, theta)
                T = T @ _make_tf(R, np.zeros(3, dtype=float))
            elif j.jtype == "prismatic":
                d = q_map.get(j.name, 0.0)
                norm = float(np.linalg.norm(j.axis))
                u = j.axis / norm if norm > 1e-12 else j.axis
                T = T @ _make_tf(np.eye(3, dtype=float), d * u)
        return T

    def joint_limits(self, joint_names: list[str] | tuple[str, ...] = DEFAULT_ARM_JOINTS) -> list[tuple[float | None, float | None]]:
        return [
            (self.joints_by_name[n].lower, self.joints_by_name[n].upper)
            if n in self.joints_by_name
            else (None, None)
            for n in joint_names
        ]


@lru_cache(maxsize=4)
def load_urdf(urdf_path: str | Path = DEFAULT_ELROBOT_URDF) -> URDFModel:
    return URDFModel(Path(urdf_path))
