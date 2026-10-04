"""R1Pro calibrated arm FK, shared geometry with the radio-button planner."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

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


@lru_cache(maxsize=1)
def model():
    return json.loads(Path(__file__).with_name("r1pro_kinematics.json").read_text())


def camera_axis(orientation_xyzw):
    axis = np.asarray(model()["camera_optical_axis_eef"], dtype=float)
    return Rotation.from_quat(orientation_xyzw).apply(axis / np.linalg.norm(axis))


class ArmModel:
    def __init__(self, side, torso):
        if side not in ("left", "right"):
            raise ValueError("未知持物手")
        arm = model()["arms"][side]
        self.mount = torso_in_body(torso)
        self.tool = np.asarray(arm["tool_from_link7"], dtype=float)
        self.chain = arm["chain"]
        moving = [row for row in self.chain if row["kind"] == "revolute"]
        self.names = [row["name"] for row in moving]
        self.lower = np.array([float(row["limits"]["lower"]) for row in moving])
        self.upper = np.array([float(row["limits"]["upper"]) for row in moving])
        self.origins = [np.asarray(row["origin"], dtype=float) for row in self.chain]
        self.axes = []
        for row in self.chain:
            x, y, z = row["axis"]
            self.axes.append(np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]]))
        self.optical = np.asarray(model()["camera_optical_axis_eef"], dtype=float)
        self.optical /= np.linalg.norm(self.optical)

    def forward(self, positions):
        result = self.mount.copy()
        index = 0
        for row, origin, axis in zip(self.chain, self.origins, self.axes):
            result = result @ origin
            if row["kind"] == "revolute":
                angle = positions[index]
                rotation = np.eye(4)
                rotation[:3, :3] = np.eye(3) + np.sin(angle)*axis + (1-np.cos(angle))*(axis@axis)
                result = result @ rotation
                index += 1
        return result @ self.tool

