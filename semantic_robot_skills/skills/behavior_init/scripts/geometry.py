"""Prepare arms-down or raised joint targets while preserving the torso."""

from math import pi

from .models import ArmTarget, JointSample, Pose
from .kinematics import ArmModel
from scipy.spatial.transform import Rotation


def prepare_arms(sample: JointSample, arm_posture: str = "down") -> tuple[float, list[ArmTarget]]:
    if arm_posture not in {"down", "raised"}:
        raise ValueError("arm_posture 必须为 down 或 raised")
    if len(sample.names) != len(sample.positions_rad) or len(set(sample.names)) != len(
        sample.names
    ):
        raise ValueError("机器人关节名与角度数量须一致，关节名须唯一")
    positions = dict(zip(sample.names, sample.positions_rad))
    names = {
        side: [f"{side}_arm_joint{i}" for i in range(1, 8)]
        for side in ("left", "right")
    }
    required = [f"torso_joint{i}" for i in range(1, 5)] + names["left"] + names["right"]
    missing = [name for name in required if name not in positions]
    if missing:
        raise ValueError(f"robot.get_state 缺少关节：{', '.join(missing)}")
    pitch = (
        positions["torso_joint1"]
        + positions["torso_joint2"]
        - positions["torso_joint3"]
    )
    target = {side: [0.0] * 7 for side in ("left", "right")}
    if arm_posture == "raised":
        target = {
            "left": [0.0, 0.6, 0.0, -(pi / 2 + pitch), 0.0, 0.0, 0.0],
            "right": [0.0, -0.6, 0.0, -(pi / 2 + pitch), 0.0, 0.0, 0.0],
        }
    torso = [positions[f"torso_joint{i}"] for i in range(1,5)]
    goals=[]
    for side in ("left","right"):
        arm=ArmModel(side,torso)
        if any(q < lo or q > hi for q, lo, hi in zip(target[side], arm.lower, arm.upper)):
            raise ValueError(f"当前躯干姿态下 {arm_posture} 的 {side} 手臂目标超出关节限位")
        transform=arm.forward(target[side])
        goals.append(ArmTarget(side=side,joint_names=names[side],
            positions_rad=target[side],pose=Pose(
            position_m=transform[:3,3].tolist(),
            orientation_xyzw=Rotation.from_matrix(transform[:3,:3]).as_quat().tolist(),
            revision=f"init:{side}")))
    return pitch,goals
