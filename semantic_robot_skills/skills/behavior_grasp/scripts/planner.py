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

"""Plan downward grasps with the jaw closing axis perpendicular to visual PCA."""
import math
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation

from .kinematics import ArmModel
from .models import PlannedMotion
from .path_replay import validate_endpoint_path, PathReplayError
from .grasp_axes import axes, DOWNWARD_EPSILON, prefer_downward


def plan_camera_plane(plan, orientation, joints, side, principal_axis, *, _prefer_downward=True, candidate_validator=None):
    if orientation is None or not isinstance(joints, dict):
        raise ValueError("主轴抓取求解需要参考坐标朝向与实测关节")
    names, values = joints.get("names", []), joints.get("positions_rad", [])
    if not names or len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("关节观测名称与角度不匹配")
    observed = dict(zip(names, values))
    try:
        torso = [observed[f"torso_joint{i}"] for i in range(1, 5)]
        arm = ArmModel(side, torso)
        initial = np.asarray([observed[n] for n in arm.names], dtype=float)
    except KeyError as error:
        raise ValueError(f"路径计算缺少关节 {error}") from error
    if not np.isfinite(initial).all():
        raise ValueError("手臂关节观测须为有限数值")
    axis, up = axes(principal_axis, orientation)
    lower, upper = arm.lower + np.deg2rad(5), arm.upper - np.deg2rad(5)

    def solve(position, seed, anchor, fixed_rotation=None):
        def eq(q):
            t = arm.forward(q)
            orientation_error=(Rotation.from_matrix(fixed_rotation.T@t[:3,:3]).as_rotvec()
                               if fixed_rotation is not None else np.array([t[:3,1]@axis]))
            return np.r_[t[:3, 3] - position, .25*orientation_error]

        def cost(q):
            return float(.05 * np.sum((q - anchor) ** 2))

        def downward(q):
            return float(-arm.forward(q)[:3,2]@up-DOWNWARD_EPSILON)

        checks=[{"type":"eq","fun":eq},{"type":"ineq","fun":downward}]
        start=np.clip(seed,lower+1e-8,upper-1e-8)
        if fixed_rotation is None and _prefer_downward:
            fit=prefer_downward(start,list(zip(lower,upper)),checks,
                lambda q:float(1.+arm.forward(q)[:3,2]@up),cost)
        else:
            fit=minimize(cost,start,method="SLSQP",bounds=list(zip(lower,upper)),
                constraints=checks,options={"maxiter":250,"ftol":1e-10})
        q = fit.x
        if not np.isfinite(q).all():
            return None
        t = arm.forward(q)
        error = float(np.linalg.norm(t[:3, 3] - position))
        axis_error = float(np.rad2deg(np.arcsin(np.clip(abs(t[:3,1]@axis), 0, 1))))
        fixed_error=(float(np.rad2deg(Rotation.from_matrix(fixed_rotation.T@t[:3,:3]).magnitude()))
                     if fixed_rotation is not None else 0.)
        margin = float(np.rad2deg(np.min(np.r_[q - arm.lower, arm.upper - q])))
        if error > .001 or axis_error > .5 or fixed_error > .5 or downward(q) < -1e-7 or margin < 5 - 1e-6:
            return None
        return q, t, {"position_error_m": error, "closing_axis_error_deg": axis_error,
            "downward_tilt_deg":float(np.rad2deg(np.arccos(np.clip(-t[:3,2]@up,-1,1)))),
            "extension_up_component":float(t[:3,2]@up),"fixed_orientation_error_deg":fixed_error,
            "minimum_joint_margin_deg": margin,
            "joint_positions_rad": q.tolist()}

    def pose(template, transform):
        return template.model_copy(update={"orientation_xyzw": Rotation.from_matrix(transform[:3, :3]).as_quat().tolist()})

    rng = np.random.default_rng(934)
    seeds = [initial, (lower + upper) / 2] + [rng.uniform(lower, upper) for _ in range(12)]
    reference = Rotation.from_quat(plan.grasp.orientation_xyzw).as_matrix()
    for rotation in (reference, reference @ np.diag([-1., -1., 1.])):
        def residual(q):
            transform = arm.forward(q)
            return np.r_[transform[:3, 3] - plan.clearance.position_m,
                         .25 * Rotation.from_matrix(rotation.T @ transform[:3, :3]).as_rotvec()]
        fit = least_squares(residual, np.clip(initial, lower, upper),
                            bounds=(lower, upper), max_nfev=250)
        seeds.append(fit.x)
    candidates = []
    for seed in seeds:
        result = solve(np.asarray(plan.clearance.position_m), seed, initial)
        if result is not None and all(np.linalg.norm(result[0] - c[0]) > .3 for c in candidates):
            candidates.append(result)
    candidates.sort(key=(lambda c:(c[2]["downward_tilt_deg"],np.linalg.norm(c[0]-initial)))
                    if _prefer_downward else lambda c:np.linalg.norm(c[0]-initial))
    replay_failures = []
    for q, transform, metrics in candidates:
        clearance = pose(plan.clearance, transform)
        approach = [PlannedMotion(key="grasp:clearance", purpose="clearance", pose=clearance)]
        lift = []
        diagnostics = [dict(metrics, key="grasp:clearance", position_m=clearance.position_m)]
        endpoints = {"clearance": clearance}
        failed = False
        grasp_rotation = None
        for stage, start, end, purpose, output in [
            ("pregrasp", plan.clearance, plan.pregrasp, "pregrasp", approach),
            ("approach", plan.pregrasp, plan.grasp, "alignment", approach),
            ("lift", plan.grasp, plan.lift, "transport", lift),
        ]:
            start_p, end_p = np.asarray(start.position_m), np.asarray(end.position_m)
            count = max(1, math.ceil(float(np.linalg.norm(end_p - start_p)) / .01 - 1e-8))
            for step in range(1, count + 1):
                position = start_p + (end_p - start_p) * step / count
                last = q.copy()
                result = solve(position, last, last, grasp_rotation)
                if result is None or np.max(abs(result[0] - last)) > np.deg2rad(12):
                    failed = True
                    break
                q, transform, metrics = result
                waypoint = pose(end.model_copy(update={"position_m": position.tolist()}), transform)
                key = f"grasp:{stage}" if step == count else f"grasp:{stage}:{step:03d}"
                output.append(PlannedMotion(key=key, purpose=purpose, pose=waypoint))
                diagnostics.append(dict(metrics, key=key, position_m=position.tolist()))
            if failed:
                break
            endpoints[{"pregrasp": "pregrasp", "approach": "grasp", "lift": "lift"}[stage]] = waypoint
            if stage == "pregrasp": grasp_rotation = transform[:3,:3].copy()
        if not failed:
            diagnostics = {"mode": "visual_principal_axis_downward", "principal_axis_body": axis.tolist(),
                "world_up_body":up.tolist(),"orientation_locked_from":"pregrasp",
                "orientation_priority":"downward_then_joint_motion",
                "downward_search_full_path_feasible":_prefer_downward,
                "candidate_count": len(candidates), "points": diagnostics,
                "validation": "local_fk_samples_and_native_default_ik_replay; collision_load_settling_unverified"}
            result_plan = plan.model_copy(update={**endpoints, "orientation_selection": diagnostics})
            try:
                replay = (candidate_validator or validate_endpoint_path)(result_plan, joints, side)
            except PathReplayError as error:
                replay_failures.append({k:v for k,v in error.diagnostic.items() if k != "samples"})
                continue
            diagnostics["endpoint_replay"] = replay
            diagnostics["rejected_endpoint_paths"] = replay_failures
            # Dense samples validate local reachability. Execute one Cartesian
            # target per stage through the existing Runtime IK controller.
            approach_goals = [step for step in approach if step.key in {
                "grasp:pregrasp", "grasp:approach"}]
            return result_plan, approach_goals, [lift[-1]], diagnostics
    if _prefer_downward:
        # The most vertical local solution can prevent a fixed-orientation
        # descent. Keep complete path feasibility before the tilt preference.
        return plan_camera_plane(plan,orientation,joints,side,principal_axis,_prefer_downward=False, candidate_validator=candidate_validator)
    if replay_failures:
        raise PathReplayError(dict(replay_failures[-1], rejected_endpoint_paths=replay_failures))
    raise ValueError("未找到夹爪朝下、开合轴垂直视觉主轴且从预抓取保持朝向的可达路径")
