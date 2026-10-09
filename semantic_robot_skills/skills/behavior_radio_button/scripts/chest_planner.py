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

"""Pure R1Pro kinematics for a chest pose with free camera pitch; no device I/O."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .geometry import torso_in_body
from .models import ArmTarget, JointSample, Pose


class ChestPlanError(ValueError):
    pass


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


def plan_chest(holding: Pose, joints: JointSample, torso, side, target_position):
    arm = ArmModel(side, torso)
    if len(joints.names) != len(joints.positions_rad) or len(set(joints.names)) != len(joints.names):
        raise ValueError("关节观测名称和角度不匹配")
    observed = dict(zip(joints.names, joints.positions_rad))
    current = np.array([observed[name] for name in arm.names], dtype=float)
    position = np.asarray(target_position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(current).all():
        raise ValueError("胸前求解需要有限位置和关节角")
    measured_rotation = Rotation.from_quat(holding.orientation_xyzw)
    initial = arm.forward(current)
    if (np.linalg.norm(initial[:3, 3]-holding.position_m) > .005
            or (Rotation.from_matrix(initial[:3, :3])*measured_rotation.inv()).magnitude() > np.deg2rad(1)):
        raise ValueError("持物手实测位姿与 R1Pro 求解模型不一致")
    lower, upper = arm.lower+np.deg2rad(10), arm.upper-np.deg2rad(10)

    def equality(values):
        transform = arm.forward(values)
        return np.r_[transform[:3, 3]-position, .25*(transform[:3, :3]@arm.optical)[1]]

    def forward_camera(values):
        return float((arm.forward(values)[:3, :3]@arm.optical)[0])

    pitch = torso[0]+torso[1]-torso[2]
    init = [0, .6 if side == "left" else -.6, 0, -(np.pi/2+pitch), 0, 0, 0]
    rng = np.random.default_rng(20260926)
    seeds = [current, np.array(init), (lower+upper)/2] + [rng.uniform(lower, upper) for _ in range(5)]
    candidates = []
    for seed in seeds:
        fit = minimize(lambda values: float(np.sum((values-current)**2)),
            np.clip(seed, lower+1e-8, upper-1e-8), method="SLSQP", bounds=list(zip(lower, upper)),
            constraints=[{"type": "eq", "fun": equality}, {"type": "ineq", "fun": forward_camera}],
            options={"maxiter": 250, "ftol": 1e-10})
        if not np.isfinite(fit.x).all():
            continue
        transform = arm.forward(fit.x)
        axis = transform[:3, :3]@arm.optical
        error = np.linalg.norm(transform[:3, 3]-position)
        # Judge feasibility from FK, independently of the optimizer termination flag.
        if (error <= .001 and abs(axis[1]) <= np.sin(np.deg2rad(.5)) and axis[0] >= 0
                and np.all(fit.x >= lower-1e-8) and np.all(fit.x <= upper+1e-8)):
            candidates.append((float(np.linalg.norm(fit.x-current)), fit.x, transform, axis))
    if not candidates:
        raise ChestPlanError("原胸前位置未找到满足镜头竖直面及 10° 关节余量的解")
    _, target, transform, axis = min(candidates, key=lambda row: row[0])
    count = max(1, int(np.ceil(np.max(abs(target-current))/np.deg2rad(8))))
    waypoints = [ArmTarget(arm_side=side, joint_names=arm.names,
        positions_rad=(current+(target-current)*i/count).tolist()) for i in range(1, count+1)]
    pose = Pose(position_m=position.tolist(),
        orientation_xyzw=Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
        revision=holding.revision, observed_at=holding.observed_at)
    diagnostics = {"camera_axis_body": axis.tolist(),
        "camera_pitch_deg": float(np.rad2deg(np.arctan2(axis[2], axis[0]))),
        "minimum_joint_margin_deg": float(np.rad2deg(np.min(np.r_[target-arm.lower, arm.upper-target]))),
        "position_error_m": float(np.linalg.norm(transform[:3, 3]-position)),
        "candidate_count": len(candidates)}
    return pose, waypoints, diagnostics
