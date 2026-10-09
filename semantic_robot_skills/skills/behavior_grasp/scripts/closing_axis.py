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

"""Finite orientation search with only closing-axis perpendicularity.

The existing SDK accepts full poses. Sample both remaining angular freedoms;
each candidate retains the recognized grasp position and is checked by cuRobo.
The sampling resolution is numerical, not a required approach angle.
"""
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from .grasp_axes import axes
from .kinematics import ArmModel


def closing_axis_candidates(inputs, plan, orientation, principal_axis, joints, side,
                            angular_samples=8):
    axis, _ = axes(principal_axis, orientation)
    observed = dict(zip(joints['names'], joints['positions_rad']))
    arm = ArmModel(side, [observed[f'torso_joint{i}'] for i in range(1, 5)])
    current = Rotation.from_matrix(arm.forward([observed[n] for n in arm.names])[:3, :3])
    y0 = current.apply([0., 1., 0.])
    y0 -= axis * (axis @ y0)
    if np.linalg.norm(y0) < 1e-8:
        basis = np.eye(3)[np.argmin(np.abs(axis))]
        y0 = basis - axis * (axis @ basis)
    y0 /= np.linalg.norm(y0)
    tangent = np.cross(axis, y0)
    candidates = []
    for azimuth in np.arange(angular_samples) * (2 * np.pi / angular_samples):
        closing = np.cos(azimuth) * y0 + np.sin(azimuth) * tangent
        z0 = current.apply([0., 0., 1.])
        z0 -= closing * (closing @ z0)
        if np.linalg.norm(z0) < 1e-8:
            z0 = axis.copy()
        z0 /= np.linalg.norm(z0)
        cross = np.cross(closing, z0)
        for roll in np.arange(angular_samples) * (2 * np.pi / angular_samples):
            extension = np.cos(roll) * z0 + np.sin(roll) * cross
            rotation = Rotation.from_matrix(np.column_stack([
                np.cross(closing, extension), closing, extension]))
            quaternion = rotation.as_quat().tolist()
            grasp = plan.grasp.model_copy(update={'orientation_xyzw': quaternion})
            diagnostic = dict(mode='closing_axis_only', principal_axis_body=axis.tolist(),
                closing_axis_body=closing.tolist(), perpendicular_error=abs(float(closing @ axis)),
                rotation_distance_rad=float((current.inv() * rotation).magnitude()),
                angular_samples_per_dimension=angular_samples,
                approach_offset_used=True, enclosure_filter_used=True)
            digest = hashlib.sha256(np.array([*grasp.position_m, *quaternion]).tobytes()).hexdigest()[:16]
            candidate = plan.model_copy(update={
                'grasp': grasp, 'pregrasp': grasp.model_copy(deep=True),
                'clearance': grasp.model_copy(deep=True),
                'lift': plan.lift.model_copy(update={'orientation_xyzw': quaternion}),
                'candidate_id': f'closing-axis:{side}:{digest}', 'orientation_selection': diagnostic})
            candidates.append(candidate)
    if inputs.orientation_xyzw is not None:
        specified=Rotation.from_quat(inputs.orientation_xyzw)
        if abs(float(specified.apply([0,1,0])@axis))>1e-5:
            raise ValueError('指定朝向的开合轴未垂直于目标主轴')
        candidates=[p for p in candidates if (specified.inv()*Rotation.from_quat(p.grasp.orientation_xyzw)).magnitude()<1e-6] or [plan.model_copy(update={
            'grasp':plan.grasp.model_copy(update={'orientation_xyzw':inputs.orientation_xyzw}),
            'orientation_selection':{'mode':'explicit','rotation_distance_rad':float((current.inv()*specified).magnitude())}})]
    up=Rotation.from_quat(orientation.body_orientation_xyzw).inv().apply([0,0,1])
    def order(candidate):
        z=Rotation.from_quat(candidate.grasp.orientation_xyzw).apply([0,0,1])
        vertical=float(z@up)
        preference=(1+vertical if inputs.approach_preference=='downward' else abs(vertical) if inputs.approach_preference=='horizontal' else 0.)
        candidate.orientation_selection['approach_preference_score']=preference
        return preference,candidate.orientation_selection['rotation_distance_rad']
    candidates.sort(key=order)
    return candidates
