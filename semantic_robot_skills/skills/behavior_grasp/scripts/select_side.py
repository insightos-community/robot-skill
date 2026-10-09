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

"""Compare both arms from one observation without sending robot commands."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import grasp_plan, check_torso, LOWER, UPPER
from .kinematics import ArmModel
from .models import TorsoObservation, TorsoAngles
from .operation import operation_pose, validate_fixed_poses
from .perpendicular import approach_candidates, preference_error
from .coupled import coupled_retraction
from .path_replay import validate_endpoint_path, PathReplayError
from .retract import plan_retraction, joint_values
from .torso_motion import duration_seconds
from .enclosure import require_enclosure, EnclosureError
from .self_collision import SelfCollisionError


def evaluate_side(inputs, located, orientation, principal_axis, source, measured, side):
    selected=inputs.model_copy(update={"side":side})
    initial_plan=grasp_plan(selected,located,orientation,principal_axis)
    automatic=inputs.orientation_xyzw is None
    candidates=(approach_candidates(selected,initial_plan,orientation,principal_axis,measured["joints"],side)
                if automatic else [initial_plan])
    failures=[]
    enclosure_failures=[]
    collision_failures=[]
    for index,plan in enumerate(candidates):
        try:
            enclosed=require_enclosure(located,plan.grasp,side,[inputs.opening_m]*2)
            value=_evaluate_plan(inputs,orientation,principal_axis,source,measured,side,plan,automatic)
        except (ValueError,KeyError,TypeError) as error:
            failures.append({"candidate":index,"reason":str(error)})
            if isinstance(error,EnclosureError):enclosure_failures.append(error)
            if isinstance(error,SelfCollisionError):collision_failures.append(error)
            continue
        value["enclosure"]=enclosed
        value["path_diagnostics"]["approach_candidates_tried"]=index+1
        value["path_diagnostics"]["rejected_approaches"]=failures
        return value
    if enclosure_failures and len(enclosure_failures)==len(failures):
        raise enclosure_failures[-1]
    if collision_failures and len(collision_failures)==len(failures):
        raise collision_failures[-1]
    raise ValueError("抓取接近未找到完整可行组合："+"；".join(f"候选 {f['candidate']}: {f['reason']}" for f in failures))


def _evaluate_plan(inputs, orientation, principal_axis, source, measured, side, plan, automatic):
    observed = joint_values(measured["joints"])
    current = np.array([observed[f"torso_joint{i}"] for i in range(1,5)])
    operation_diagnostics = None
    if isinstance(inputs.operation_torso,TorsoObservation):
        target,operation_diagnostics = operation_pose(plan,orientation,measured["joints"],side,False,principal_axis,seed_limit=2 if automatic else None)
    elif isinstance(inputs.operation_torso,TorsoAngles):
        target = check_torso(inputs.operation_torso.positions_rad)
        if abs(target[3]) > np.pi/2:
            raise ValueError("操作躯干的腰部转向须在相对身体正前方左右各 90° 内")
    else:
        target = []
    duration = duration_seconds(current.tolist(),target) if target else None
    retraction = plan_retraction(measured["joints"],target,source,measured["base_pose"]) if target else None
    from .joint_path import plan_joint_paths
    planned=plan_joint_paths(plan,measured["joints"],side,target,retraction,operation_diagnostics)
    up=Rotation.from_quat(orientation.body_orientation_xyzw).inv().apply([0,0,1])
    tilt=float(np.rad2deg(np.arccos(np.clip(-Rotation.from_quat(plan.grasp.orientation_xyzw).apply([0,0,1])@up,-1,1))))
    arm=ArmModel(side,target if target else current)
    wire=planned["trajectories"]["approach"]
    indices=[wire["joint_names"].index(n) for n in arm.names]
    qs=np.asarray(wire["positions_rad"])[:,indices]
    margin=float(np.rad2deg(np.min(np.r_[np.ravel(qs-arm.lower),np.ravel(arm.upper-qs)])))
    torso_motion=float(np.linalg.norm((np.asarray(target)-current)/(np.array(UPPER)-LOWER))) if target else 0.
    arm_motion=float(np.sum(np.linalg.norm(np.diff(qs,axis=0)/(arm.upper-arm.lower),axis=1)))
    return {"side":side,"plan":plan,"operation_target":target,"operation_duration":duration,
            "operation_diagnostics":operation_diagnostics,"retraction_plan":planned["retraction_plan"],
            "path_diagnostics":planned,"metrics":{"tilt_deg":tilt,"minimum_arm_margin_deg":margin,
                "torso_motion_normalized":torso_motion,"arm_motion_normalized":arm_motion}}


def rank_candidates(candidates, preference='auto'):
    """Prefer direction among feasible hand plans, then joint margin and motion."""
    if preference != 'auto':
        from .operation import PLANE_TOLERANCE_RAD
        errors=[preference_error(c['metrics']['tilt_deg'],preference) for c in candidates]
        best=min(errors)
        candidates=[c for c,error in zip(candidates,errors) if error<=best+float(np.rad2deg(PLANE_TOLERANCE_RAD))]
    return min(candidates,key=lambda c:(-c["metrics"]["minimum_arm_margin_deg"],
        c["metrics"]["torso_motion_normalized"],c["metrics"]["arm_motion_normalized"],c["side"]))


def select_side(inputs, located, orientation, principal_axis, source, measured):
    candidates = []; summary = {}
    for side in ("left","right"):
        try:
            candidate = evaluate_side(inputs,located,orientation,principal_axis,source,measured,side)
        except (ValueError,KeyError,TypeError) as error:
            summary[side] = {"feasible":False,"reason":str(error)}
            if isinstance(error,PathReplayError):
                summary[side]["endpoint_failure"] = {k:v for k,v in error.diagnostic.items() if k != "samples"}
        else:
            candidates.append(candidate)
            summary[side] = {"feasible":True,**candidate["metrics"],
                "operation_target_rad":candidate["operation_target"],
                "validation":candidate["path_diagnostics"]["validation"]}
    if not candidates:
        return None,{"candidates":summary,"selected_side":None}
    preference=(getattr(inputs,'approach_preference','auto') if getattr(inputs,'orientation_xyzw',None) is None else 'auto')
    winner = rank_candidates(candidates,preference)
    return winner,{"candidates":summary,"selected_side":winner["side"],
        "approach_preference":preference,
        "priority":["full_path_feasibility",*(["approach_preference"] if preference!='auto' else []),"joint_margin","torso_motion","arm_motion"]}
