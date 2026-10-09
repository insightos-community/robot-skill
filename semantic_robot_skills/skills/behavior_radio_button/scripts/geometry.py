# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""R1Pro frame conversion and relative hand geometry; no device I/O."""

import numpy as np
from scipy.spatial.transform import Rotation

from .models import ArmTarget, JointSample, Pose


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


def measured_pose(
    value: dict, joints: list[float], revision: str, observed_at: str
) -> Pose:
    """Convert the motion.get_end_effector_state observation to body."""
    frame = value["frame_id"]
    position = value["position_m"]
    quaternion = value["orientation_xyzw"]
    # Validate before Rotation can normalize a malformed measured quaternion.
    local = Pose(position_m=position, orientation_xyzw=quaternion, revision=revision)
    matrix = np.eye(4)
    matrix[:3, 3] = local.position_m
    matrix[:3, :3] = Rotation.from_quat(local.orientation_xyzw).as_matrix()
    if frame == "torso_link4":
        matrix = torso_in_body(joints) @ matrix
    elif frame != "body":
        raise ValueError(f"Unsupported measured end-effector frame: {frame}")
    return Pose(
        position_m=matrix[:3, 3].tolist(),
        orientation_xyzw=Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
        revision=revision,
        observed_at=observed_at,
    )


def inspection_pose(holding: Pose, operating_side: str) -> Pose:
    """Body-side offset; holding-hand attitude does not affect the approach."""
    up = np.array([0.0, 0.0, 1.0])
    lateral = np.array([0.0, 1.0 if operating_side == "left" else -1.0, 0.0])
    sign = 1.0 if operating_side == "left" else -1.0
    offset = [0.0, sign * 0.40, -0.15]
    operating_z = -lateral  # Horizontal tool axis points toward the holding side.
    operating_x = -up  # Same closed-finger convention as the radio example.
    rotation = np.column_stack(
        (operating_x, np.cross(operating_z, operating_x), operating_z)
    )
    # Mirror the optical compensation: left -X, right +X raises either wrist view.
    # Both cameras look down about 25 degrees at the level tool baseline.
    rotation = (
        Rotation.from_rotvec([-sign * np.deg2rad(15.0), 0, 0]).as_matrix() @ rotation
    )
    return Pose(
        position_m=(np.asarray(holding.position_m) + offset).tolist(),
        orientation_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
        revision=holding.revision,
        observed_at=holding.observed_at,
    )


def chest_position(joints, holding_side, forward_m, side_offset_m, height_above_torso_m):
    """Original chest-center target, mirrored by holding side."""
    torso = torso_in_body(joints)
    center = (torso @ [0., 0., height_above_torso_m, 1.])[:3]
    forward = torso[:3, 0].copy()
    forward[2] = 0
    length = np.linalg.norm(forward)
    if length < 1e-8:
        raise ValueError("Torso forward axis has no horizontal projection")
    forward /= length
    left = np.cross([0., 0., 1.], forward)
    sign = 1. if holding_side == "left" else -1.
    return (center + forward_m*forward + sign*side_offset_m*left
            + np.array([0., sign*.15, 0.])).tolist()


def orientation_error_rad(actual: Pose, target: Pose) -> float:
    return float((Rotation.from_quat(actual.orientation_xyzw)
                  * Rotation.from_quat(target.orientation_xyzw).inv()).magnitude())


def vertical_axis_body(base_pose: dict) -> list[float]:
    """World/odom vertical expressed in body, including measured roll and pitch."""
    if base_pose.get("frame_id") not in {"world", "odom"}:
        raise ValueError("水平转动需要 world/odom 下的机器人位姿")
    q = np.asarray(base_pose["orientation_xyzw"], dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(q @ q - 1) > 1e-5:
        raise ValueError("机器人朝向四元数无效")
    return Rotation.from_quat(q).inv().apply([0, 0, 1]).tolist()


def search_rotation_targets(holding: Pose, holding_side: str, attempt: int,
                            vertical: list[float]) -> list[Pose]:
    """Persist a stage's plan from one measured pose; each increment is 15 degrees."""
    rotation = Rotation.from_quat(holding.orientation_xyzw)
    up = np.asarray(vertical, dtype=float)
    if holding_side not in {"left", "right"} or attempt not in (2, 3, 4):
        raise ValueError("未知观察转动阶段")
    if up.shape != (3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1) > 1e-5:
        raise ValueError("竖直轴无效")
    if attempt == 3:
        # Reuse the former upward flip: local X, initially raising tool Z.
        sign = -1.0 if np.dot(up, -rotation.as_matrix()[:, 1]) < -1e-8 else 1.0
        rotations = [rotation * Rotation.from_rotvec([sign*np.pi*i/12, 0, 0])
                     for i in range(1, 13)]
    else:
        # Outward: left holding hand positive yaw, right negative; stage 4 reverses it.
        sign = (1.0 if holding_side == "left" else -1.0) * (1 if attempt == 2 else -1)
        rotations = [Rotation.from_rotvec(up * sign*np.pi*i/12) * rotation
                     for i in range(1, 7)]
    return [holding.model_copy(update={"orientation_xyzw": r.as_quat().tolist()})
            for r in rotations]


def operating_init_targets(sample: JointSample, side: str) -> list[ArmTarget]:
    """Reuse prepare_arms joint geometry for the operating arm only."""
    if side not in ("left", "right"):
        raise ValueError("Unknown operating arm")
    if len(sample.names) != len(sample.positions_rad) or len(set(sample.names)) != len(
        sample.names
    ):
        raise ValueError("机器人关节名与角度数量须一致，关节名须唯一")
    positions = dict(zip(sample.names, sample.positions_rad))
    names = [f"{side}_arm_joint{i}" for i in range(1, 8)]
    required = [f"torso_joint{i}" for i in range(1, 4)] + names
    missing = [name for name in required if name not in positions]
    if missing:
        raise ValueError(f"robot.get_state 缺少关节：{', '.join(missing)}")
    pitch = (
        positions["torso_joint1"]
        + positions["torso_joint2"]
        - positions["torso_joint3"]
    )
    target = [0, 0.6 if side == "left" else -0.6, 0, -(np.pi / 2 + pitch), 0, 0, 0]
    # R1Pro URDF elbow limits; reject the complete plan before its first segment.
    if not -2.0944 <= target[3] <= 0.3491:
        raise ValueError("操作臂准备姿态超出肘关节限位，请核对躯干恢复结果")
    return [
        ArmTarget(
            arm_side=side,
            joint_names=names,
            positions_rad=[
                positions[name] + step / 10 * (end - positions[name])
                for name, end in zip(names, target)
            ],
        )
        for step in range(1, 11)
    ]
