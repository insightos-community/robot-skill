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

"""Approach preferences with a closing axis perpendicular to the observed long axis."""
import numpy as np
from scipy.spatial.transform import Rotation
from .grasp_axes import axes
from .kinematics import ArmModel


def preference_error(tilt, preference):
    if preference == 'downward':return float(tilt)
    if preference == 'horizontal':return abs(float(tilt)-90.)
    if preference == 'auto':return 0.
    raise ValueError('未知接近偏好')


def approach_candidates(inputs, plan, orientation, principal_axis, joints, side):
    axis,up=axes(principal_axis,orientation)
    distance=float(np.linalg.norm(inputs.pregrasp_offset_m))
    if distance<=0:raise ValueError('接近距离须为正')
    observed=dict(zip(joints['names'],joints['positions_rad']))
    arm=ArmModel(side,[observed[f'torso_joint{i}'] for i in range(1,5)])
    current=Rotation.from_matrix(arm.forward([observed[n] for n in arm.names])[:3,:3])
    extension=current.apply([0,0,1])
    normal=extension-axis*(axis@extension)
    if np.linalg.norm(normal)<1e-8:
        basis=np.eye(3)[np.argmin(np.abs(axis))]
        normal=basis-axis*(basis@axis)
    normal/=np.linalg.norm(normal)
    tangent=np.cross(axis,normal)
    horizontal=extension-up*(up@extension)
    if np.linalg.norm(horizontal)<1e-8:
        basis=np.eye(3)[np.argmin(np.abs(up))]
        horizontal=basis-up*(up@basis)
    horizontal/=np.linalg.norm(horizontal)
    cross_horizontal=np.cross(up,horizontal)
    directions=[]
    for raw in (extension,normal,tangent,-tangent,-normal,-up,
                horizontal,-horizontal,cross_horizontal,-cross_horizontal,
                horizontal-up,-horizontal-up,cross_horizontal-up,-cross_horizontal-up):
        direction=np.asarray(raw)/np.linalg.norm(raw)
        if not any(np.allclose(direction,old,atol=1e-8,rtol=0) for old in directions):
            directions.append(direction)
    candidates=[]
    # All candidates keep Y perpendicular to both the approach Z and object axis.
    # When Z parallels the object axis (top-down cans), retain bounded roll choices.
    for direction in directions:
        closing=np.cross(direction,axis)
        if np.linalg.norm(closing)<1e-8:
            closing=current.apply([0.,1.,0.])
            closing-=direction*(direction@closing)
            if np.linalg.norm(closing)<1e-8:
                basis=np.eye(3)[np.argmin(np.abs(direction))]
                closing=basis-direction*(direction@basis)
            closing/=np.linalg.norm(closing)
            closing_axes=[closing,-closing,np.cross(direction,closing),-np.cross(direction,closing)]
        else:
            closing/=np.linalg.norm(closing)
            closing_axes=[closing,-closing]
        for closing in closing_axes:
            rotation=Rotation.from_matrix(np.column_stack([np.cross(closing,direction),closing,direction]))
            q=rotation.as_quat().tolist()
            grasp=plan.grasp.model_copy(update={'orientation_xyzw':q})
            pregrasp=plan.pregrasp.model_copy(update={
                'position_m':(np.asarray(grasp.position_m)-distance*direction).tolist(),'orientation_xyzw':q})
            lift=plan.lift.model_copy(update={'orientation_xyzw':q})
            tilt=float(np.rad2deg(np.arccos(np.clip(-direction@up,-1.,1.))))
            diagnostics={'mode':'principal_axis_closing_approach','principal_axis_body':axis.tolist(),
                'approach_direction_body':direction.tolist(),'approach_distance_m':distance,
                'downward_tilt_deg':tilt,'horizontal_deviation_deg':abs(tilt-90.),
                'approach_preference':inputs.approach_preference,'preference_applied':True,
                'preference_error_deg':preference_error(tilt,inputs.approach_preference),
                'orientation_priority':'approach_preference_then_rotation_distance',
                'orientation_locked_from':'pregrasp'}
            candidate=plan.model_copy(update={'clearance':pregrasp.model_copy(deep=True),
                'pregrasp':pregrasp,'grasp':grasp,'lift':lift,'orientation_selection':diagnostics})
            candidates.append((preference_error(tilt,inputs.approach_preference),(current.inv()*rotation).magnitude(),candidate))
    candidates.sort(key=lambda item:(round(item[0],8),item[1]))
    return [item[2] for item in candidates]
