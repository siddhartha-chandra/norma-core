"""Pink IK solver plug-in for DESK-CORE.

KINEMATICS CONTRACT:
  - URDF: hardware/elrobot/simulation/elrobot_follower.urdf
  - World frame: base_link
  - EE frame: Gripper_Base_v1_1
  - Arm joints: ["rev_motor_01", ..., "rev_motor_07"] (meters, radians)
  - Target orientation: quaternion (qx, qy, qz, qw); current joint seed: seed_joints_rad
  - URDF q=[0]*7 pose aligns with joint_zero_ticks + joint_signs in calibration JSON

INTERFACE (Option A - Local Python module):
  Implement solve_ik() below. Calling MCP reload_config() picks it up live.

INTERFACE (Option B - HTTP server on your laptop):
  Set PINK_IK_URL=http://<your-ip>:<port>/solve in .env
  Accept POST JSON: {"x", "y", "z", "qx", "qy", "qz", "qw", "seed_joints_rad": [7 floats], "obstacles": [...]}
  Return JSON: {"joints": [j1, j2, j3, j4, j5, j6, j7]}

SceneObstacle schema (for collision avoidance):
  {"name": str, "shape": "cylinder"|"box", "x": float, "y": float, "z": float,
   "radius_m": float, "height_m": float}
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import SceneObstacle

# =============================================================================
# IMPLEMENT YOUR PINK IK SOLVER BELOW
# =============================================================================


def solve_ik(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
    seed_joints_rad: list[float] | None = None,
    obstacles: list[SceneObstacle] | None = None,
) -> list[float]:
    """Solve inverse kinematics for the 7-DoF ElRobot arm.

    Args:
        x, y, z: Target EE position in base_link frame (meters).
        qx, qy, qz, qw: Target orientation quaternion in xyzw order.
        obstacles: List of SceneObstacle for collision avoidance.
        seed_joints_rad: Current arm angles [j1, ..., j7] in radians.

    Returns:
        List of 7 joint angles [j1, ..., j7] in radians, or raises if infeasible.

    Example using Pink/Pinocchio:
        import pink
        import pinocchio as pin

        # Load URDF model
        model = pin.buildModelFromUrdf("hardware/elrobot/simulation/elrobot_follower.urdf")
        data = model.createData()

        # Create IK configuration
        config = pink.Configuration(model, data, seed_joints_rad)

        # Add end-effector task
        ee_task = pink.tasks.FrameTask(
            "Gripper_Base_v1_1",
            position_cost=1.0,
            orientation_cost=0.5,
        )
        ee_task.set_target(target_SE3)

        # Solve
        q_solution = pink.solve_ik(config, [ee_task], solver="quadprog")
        return q_solution[7:14].tolist()  # Extract arm joints
    """
    # --- STUB: Replace with actual Pink/Pinocchio IK implementation ---
    raise NotImplementedError(
        "Pink IK solver not implemented. Implement solve_ik() in this file "
        "or set PINK_IK_URL=http://<your-ip>:<port>/solve in .env."
    )


# =============================================================================
# OPTIONAL: HTTP SERVER EXAMPLE
# =============================================================================
#
# To run Pink IK on your laptop and expose via HTTP:
#
# ```python
# # pink_server.py
# from flask import Flask, request, jsonify
# import pinocchio as pin
# import pink
#
# app = Flask(__name__)
#
# # Load model once
# model = pin.buildModelFromUrdf("elrobot_follower.urdf")
# data = model.createData()
#
# @app.route("/solve", methods=["POST"])
# def solve():
#     req = request.json
#     x, y, z = req["x"], req["y"], req["z"]
#     quat_xyzw = [
#         req.get("qx", 0.0), req.get("qy", 0.0),
#         req.get("qz", 0.0), req.get("qw", 1.0),
#     ]
#     seed_joints_rad = req.get("seed_joints_rad", [0.0] * 7)
#     obstacles = req.get("obstacles", [])
#
#     # ... your Pink IK logic ...
#     joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Result
#
#     return jsonify({"joints": joints})
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5050)
# ```
#
# Then set in .env: PINK_IK_URL=http://<your-laptop-ip>:5050/solve
