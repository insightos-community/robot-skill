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

"""Pure R1Pro torso geometry and body-frame grasp waypoint calculations."""

import hashlib
import json
from math import acos, atan2, ceil, cos, hypot, isfinite, sin

from .models import GraspPlan, Input, LocatedObject, Pose, ObjectOrientation
from .orientation import automatic_grasp_geometry

LOWER = [-1.1345, -2.7925, -1.8326, -3.0543]
UPPER = [1.8326, 2.5307, 1.5708, 3.0543]


def check_torso(joints: list[float]) -> list[float]:
    if len(joints) != 4 or any(
        not isinstance(q, (int, float))
        or isinstance(q, bool)
        or not isfinite(q)
        or not low <= q <= high
        for q, low, high in zip(joints, LOWER, UPPER)
    ):
        raise ValueError("躯干目标需在 R1Pro 四关节限位内，单位 rad")
    return [float(q) for q in joints]


def torso_pose(joints: list[float]) -> tuple[float, float, float]:
    q1, q2, q3, _ = joints
    pitch = q1 + q2 - q3
    return (
        -0.079032 + 0.4 * sin(q1) + 0.3 * sin(q1 + q2) + 0.09962 * sin(pitch),
        0.34265 + 0.4 * cos(q1) + 0.3 * cos(q1 + q2) + 0.09962 * cos(pitch),
        pitch,
    )


def torso_offset_targets(joints: list[float], offset: list[float]) -> list[list[float]]:
    """Adapt the SDK demo's two-link solution; preserve pitch, knee branch and yaw.

    offset = [forward, up] meters. Evaluate every 1 cm waypoint before execution.
    """
    current = check_torso(joints)
    forward, up = offset
    if not all(isfinite(v) for v in offset):
        raise ValueError("躯干偏移须为有限数值")
    distance = hypot(forward, up)
    if distance > 1.4:
        raise ValueError("躯干偏移超出两连杆可达范围")
    count = ceil(distance / 0.01)
    x, z, pitch = torso_pose(current)
    branch = -1 if current[1] < 0 else 1
    targets = []
    for step in range(1, count + 1):
        horizontal = x + forward * step / count + 0.079032 - 0.09962 * sin(pitch)
        vertical = z + up * step / count - 0.34265 - 0.09962 * cos(pitch)
        cosine = (horizontal**2 + vertical**2 - 0.4**2 - 0.3**2) / (2 * 0.4 * 0.3)
        if abs(cosine) > 1 + 1e-10:
            raise ValueError("躯干偏移包含不可达位置")
        q2 = branch * acos(max(-1.0, min(1.0, cosine)))
        q1 = atan2(horizontal, vertical) - atan2(0.3 * sin(q2), 0.4 + 0.3 * cos(q2))
        targets.append(check_torso([q1, q2, q1 + q2 - pitch, current[3]]))
    return targets


def grasp_plan(inputs: Input, located: LocatedObject, orientation: ObjectOrientation | None = None,
               principal_axis=None) -> GraspPlan:
    if located.object_ref != inputs.object_ref:
        raise ValueError("识别结果的 object_ref 与任务目标不一致")
    if located.identity_confidence < inputs.minimum_confidence:
        raise ValueError("识别置信度低于 minimum_confidence")
    quaternion = inputs.orientation_xyzw
    grasp_offset, pregrasp_offset, lift_offset = inputs.grasp_offset_m, [0., 0., 0.], [0., 0., 0.]
    if quaternion is None:
        if orientation is None or orientation.object_ref != inputs.object_ref:
            raise ValueError("自动抓取需要本次目标的真实朝向")
        if principal_axis is not None:
            from .grasp_axes import axes, seed_quaternion
            from .orientation import rotate, inverse
            axis, up = axes(principal_axis, orientation)
            quaternion = seed_quaternion(axis, up, inputs.side)
            world_body = inverse(orientation.body_orientation_xyzw)
            grasp_offset = rotate(world_body, rotate(orientation.object_orientation_xyzw, grasp_offset))
            pregrasp_offset, lift_offset = (rotate(world_body, x) for x in (pregrasp_offset, lift_offset))
        else:
            quaternion, grasp_offset, pregrasp_offset, lift_offset = automatic_grasp_geometry(
                orientation, grasp_offset, pregrasp_offset, lift_offset)
    center = [x + d for x, d in zip(located.pose.position_m, grasp_offset)]
    orientation_selection = None

    def pose(offset: list[float]) -> Pose:
        return Pose(
            frame_id="body",
            position_m=[x + d for x, d in zip(center, offset)],
            orientation_xyzw=quaternion,
            revision=located.pose.revision,
            observed_at=located.pose.observed_at,
        )

    grasp = pose([0.0, 0.0, 0.0])
    digest = hashlib.sha256(
        json.dumps(
            {
                "object_ref": inputs.object_ref,
                "side": inputs.side,
                "pose": grasp.model_dump(mode="json"),
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()[:16]
    pregrasp = pose(pregrasp_offset)
    # Internal alias retained for the geometric candidate solver.
    clearance = pregrasp.model_copy(deep=True)
    return GraspPlan(
        clearance=clearance,
        orientation_selection=orientation_selection,
        pregrasp=pregrasp,
        grasp=grasp,
        lift=pose(lift_offset),
        candidate_id=f"parameterized:{digest}",
    )
