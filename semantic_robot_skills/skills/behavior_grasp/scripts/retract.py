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

"""Compact arm preparation for an operation-torso transition, using existing FK.

Checks sampled link centre lines against the finite target box. Collision meshes
and other scene obstacles are unavailable through this Skill's existing inputs.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .kinematics import ArmModel, torso_in_body
from .operation import JOINT_MARGIN_RAD

SAMPLE_ANGLE_RAD = np.deg2rad(5)
JOINT_TOLERANCE_RAD = .02


def joint_values(joints):
    names, values = joints["names"], joints["positions_rad"]
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("收臂需要名称唯一且长度一致的关节观测")
    result = dict(zip(names, values))
    if not np.isfinite(values).all():
        raise ValueError("收臂需要有限关节角")
    return result


def link_points(arm, positions):
    """Batch FK of revolute joint centres and tool, in torso_link4 coordinates."""
    positions = np.atleast_2d(positions)
    t = np.broadcast_to(np.eye(4), (len(positions), 4, 4)).copy()
    points = []; index = 0
    for row, origin, axis in zip(arm.chain, arm.origins, arm.axes):
        t = t @ origin
        if row["kind"] == "revolute":
            q = positions[:, index, None, None]
            rotation = np.broadcast_to(np.eye(4), t.shape).copy()
            rotation[:, :3, :3] = np.eye(3) + np.sin(q)*axis + (1-np.cos(q))*(axis@axis)
            t = t @ rotation
            points.append(t[:, :3, 3].copy()); index += 1
    points.append((t @ arm.tool)[:, :3, 3])
    return np.stack(points, axis=1)


def interpolation(start, end):
    count = max(1, int(np.ceil(np.max(np.abs(np.asarray(end)-start))/SAMPLE_ANGLE_RAD)))
    return np.linspace(start, end, count+1)


def target_box(source, base):
    bounds = np.asarray(source["object_bounds"], dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("收臂需要有效目标包围盒")
    if source["frame_id"] != base["frame_id"]:
        raise ValueError("收臂目标几何与底盘坐标系不一致")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(base["orientation_xyzw"]).as_matrix()
    transform[:3, 3] = base["position"]
    if not np.isfinite(transform).all():raise ValueError("收臂需要有限底盘位姿")
    return {"frame_id":source["frame_id"],"bounds":bounds.tolist(),
            "source_from_body":transform.tolist()}


def segment_box_margins(body_points, box):
    """Signed separating-axis gaps for every adjacent link-centre segment.

    Transform into the original box frame to retain its measured height and
    lateral extent. Positive gaps separate the line segment and box; negative
    gaps indicate intersection. This checks entire segments, including cases
    where both joint centres are outside. It does not model link thickness.
    """
    transform=np.asarray(box["source_from_body"],dtype=float)
    points=np.asarray(body_points)@transform[:3,:3].T+transform[:3,3]
    low,high=np.asarray(box["bounds"],dtype=float)
    half=(high-low)/2;center=(high+low)/2
    a,b=points[...,:-1,:],points[...,1:,:]
    midpoint=(a+b)/2-center;direction=(b-a)/2
    face_gaps=np.abs(midpoint)-half-np.abs(direction)
    # Segment/box SAT: box face normals and segment crossed with each box axis.
    axes=np.cross(direction[...,None,:],np.eye(3))
    norms=np.linalg.norm(axes,axis=-1)
    gaps=np.abs(np.sum(midpoint[...,None,:]*axes,axis=-1))-np.sum(half*np.abs(axes),axis=-1)
    cross_gaps=np.full_like(gaps,-np.inf)
    np.divide(gaps,norms,out=cross_gaps,where=norms>np.finfo(float).eps)
    return np.maximum(face_gaps.max(axis=-1),cross_gaps.max(axis=-1))


def in_body(points, mounts):
    return np.einsum("nij,nkj->nki", mounts[:, :3, :3], points) + mounts[:, None, :3, 3]


def sweep_margin(arm, q, mounts, box):
    points = np.repeat(link_points(arm, q), len(mounts), axis=0)
    return float(np.min(segment_box_margins(in_body(points, mounts),box)))


def plan_retraction(joints, target_torso, source, base):
    observed = joint_values(joints)
    torso = np.array([observed[f"torso_joint{i}"] for i in range(1, 5)])
    target_torso = np.asarray(target_torso, dtype=float)
    if target_torso.shape != (4,) or not np.isfinite(target_torso).all():
        raise ValueError("收臂需要四个有限操作躯干目标")
    box = target_box(source, base)
    mounts = np.array([torso_in_body(q) for q in interpolation(torso, target_torso)])
    plans = []; diagnostics = {}
    for side in ("left", "right"):
        arm = ArmModel(side, torso)
        initial = np.array([observed[name] for name in arm.names])
        low, high = arm.lower+JOINT_MARGIN_RAD, arm.upper-JOINT_MARGIN_RAD
        start_points = link_points(arm, initial)[0]
        shoulder = start_points[0]
        reach = np.linalg.norm(np.diff(start_points, axis=0), axis=1).sum()
        sign = 1 if side == "left" else -1
        # A fixed sampling grid gives the optimizer stable constraint dimensions.
        count = int(np.ceil(np.max(high-low)/SAMPLE_ANGLE_RAD))+1
        fractions = np.linspace(0, 1, count)[:, None]

        def constraints(q):
            points = link_points(arm, q)[0]
            path = link_points(arm, initial+fractions*(q-initial))
            world_path = in_body(path, np.repeat(mounts[:1], len(path), axis=0))
            world_sweep = in_body(np.repeat(points[None], len(mounts), axis=0), mounts)
            # Keep each arm on its own side of the torso centre plane throughout
            # retraction, and outside the shoulder plane in the folded pose.
            return np.r_[segment_box_margins(world_path,box).ravel(),
                         segment_box_margins(world_sweep,box).ravel(),
                         sign*path[:, :, 1].ravel(),
                         sign*(points[1:, 1]-shoulder[1]),
                         shoulder[2]-points[3, 2],  # elbow below shoulder
                         points[-1, 2]-points[3, 2],  # hand at/above elbow
                         shoulder[2]-points[-1, 2]]  # hand below shoulder

        def objective(q):
            points = link_points(arm, q)[0]
            # Dimensionless compactness and movement cost; no remembered pose.
            return float(np.mean(np.sum(((points[3:]-shoulder)/reach)**2, axis=1))
                         + np.mean(((q-initial)/(high-low))**2))

        best = None
        for seed in (np.clip(initial, low, high), (low+high)/2):
            fit = minimize(objective, seed, method="SLSQP", bounds=list(zip(low, high)),
                constraints=[{"type": "ineq", "fun": constraints}],
                options={"maxiter": 180, "ftol": 1e-9})
            if np.isfinite(fit.x).all() and constraints(fit.x).min() >= -1e-7:
                if best is None or objective(fit.x) < objective(best): best = fit.x.copy()
        if best is None:
            raise ValueError(f"{side} 收臂及操作躯干过渡未找到避开目标包围盒的姿态")
        plans.append({"arm_side": side, "joint_names": arm.names, "positions_rad": best.tolist()})
        diagnostics[side] = {
            "initial_sweep_margin_m": sweep_margin(arm, initial, mounts, box),
            "retracted_sweep_margin_m": sweep_margin(arm, best, mounts, box),
            "initial_tool_torso_m": start_points[-1].tolist(),
            "retracted_tool_torso_m": link_points(arm, best)[0, -1].tolist(),
        }
    return {"arms": plans, "torso_start_rad": torso.tolist(), "torso_target_rad": target_torso.tolist(),
            "target_box": box,
            "diagnostics": diagnostics,
            "validation": "sampled_link_segments_target_box; no scene mesh collision validation"}


def verify_retraction(plan, joints):
    observed = joint_values(joints)
    torso = np.array([observed[f"torso_joint{i}"] for i in range(1, 5)])
    if np.max(np.abs(torso-plan["torso_start_rad"])) > JOINT_TOLERANCE_RAD:
        raise ValueError("收臂期间躯干偏离计划起点，停止操作躯干调整")
    mounts = np.array([torso_in_body(q) for q in interpolation(torso, plan["torso_target_rad"])])
    for target in plan["arms"]:
        q = np.array([observed[name] for name in target["joint_names"]])
        if np.max(np.abs(q-target["positions_rad"])) > JOINT_TOLERANCE_RAD:
            raise ValueError(f"{target['arm_side']} 收臂实测关节未到位，停止操作躯干调整")
        arm = ArmModel(target["arm_side"], torso)
        if sweep_margin(arm, q, mounts, plan["target_box"]) < -1e-7:
            raise ValueError(f"{target['arm_side']} 实测收臂姿态穿入目标包围盒")
