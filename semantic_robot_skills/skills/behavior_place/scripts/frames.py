"""Measured EEF conversion; torso FK shared with the existing grasp implementation."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import matrix, pose
from .models import Transform

def torso_in_body(joints: list[float]) -> np.ndarray:
    """Same URDF transform as radio_handling_demo.scripts.geometry.torso_world."""
    values = np.asarray(joints, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Expected four finite torso joint positions")
    q1, q2, q3, yaw = values
    pitch = q1 + q2 - q3
    out = np.eye(4)
    out[:3, 3] = [
        -0.079032 + 0.4 * np.sin(q1) + 0.3 * np.sin(q1 + q2) + 0.09962 * np.sin(pitch),
        0,
        0.34265 + 0.4 * np.cos(q1) + 0.3 * np.cos(q1 + q2) + 0.09962 * np.cos(pitch),
    ]
    out[:3, :3] = (
        Rotation.from_rotvec([0, pitch, 0]).as_matrix()
        @ Rotation.from_rotvec([0, 0, yaw]).as_matrix()
    )
    return out


def eef_in_body(entry, robot_state, side):
    if entry["axis"] != side:
        raise ValueError("末端回读与持物手不一致")
    value = matrix(Transform(position_m=entry["position_m"], orientation_xyzw=entry["orientation_xyzw"]))
    frame = entry["frame_id"]
    if frame == "torso_link4":
        joints = dict(zip(robot_state["joints"]["names"], robot_state["joints"]["positions_rad"]))
        value = torso_in_body([joints[f"torso_joint{i}"] for i in range(1,5)]) @ value
    elif frame not in ("body", "base", "base_link", "base_footprint"):
        base = robot_state["base_pose"]
        if frame != base["frame_id"]:
            raise ValueError(f"末端回读坐标系 {frame} 缺少 body 变换")
        world_body = matrix(Transform(position_m=base["position"], orientation_xyzw=base["orientation_xyzw"]))
        value = np.linalg.inv(world_body) @ value
    return pose(value)
