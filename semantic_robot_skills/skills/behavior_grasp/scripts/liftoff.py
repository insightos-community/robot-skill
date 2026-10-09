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

"""Configurable post-contact lift before attached-object cuRobo planning."""
from scipy.spatial.transform import Rotation
from semantic_robot_skill_sdk import Action
from .enclosure import measured_pose_in_body, transform
from .runtime import execute
from .held_geometry import held_geometry

POSITION_TOLERANCE_M = 0.003


def measured_pose(output, measured, side):
    """Resolve the measured EEF in the fixed world/odometry frame."""
    base = measured['base_pose']
    if base['frame_id'] not in ('world', 'odom'):
        raise ValueError('离地抬升需要 world 或 odom 基座位姿')
    world_body = transform(dict(position_m=base['position'], orientation_xyzw=base['orientation_xyzw']))
    body_eef = transform(measured_pose_in_body(output, measured, side))
    world_eef = world_body @ body_eef
    return dict(frame_id=base['frame_id'], position_m=world_eef[:3, 3].tolist(),
                orientation_xyzw=Rotation.from_matrix(world_eef[:3, :3]).as_quat().tolist(),
                observed_at=output.get('end_effector', {}).get('observed_at'))


async def prepare_lift(ctx, state, inputs, read_state):
    async def measure(phase):
        measured = await read_state(f'grasp:liftoff-{phase}-state')
        eef = await execute(ctx, state, f'grasp:liftoff-{phase}-eef', Action(
            type='motion.get_end_effector_state', parameters={'axis':inputs.side},
            timeout_seconds=inputs.timeout_seconds))
        return measured_pose(eef.output or {}, measured, inputs.side)

    if state.liftoff_target is None:
        origin = await measure('before')
        if state.held_geometry is None and state.observation_anchor is not None:
            state.held_geometry = held_geometry(state.located, state.observation_anchor, origin, inputs.object_ref)
        target = dict(origin, position_m=list(origin['position_m']),
                      revision=state.plan.candidate_id + ':liftoff')
        state.liftoff_height_m = inputs.liftoff_height_m
        target['position_m'][2] += state.liftoff_height_m
        state.liftoff_target = target
        ctx.checkpoint(state)


async def lift(ctx, state, inputs, read_state):
    await prepare_lift(ctx, state, inputs, read_state)
    target = state.liftoff_target
    result = await execute(ctx, state, 'grasp:liftoff-ik', Action(
        type='motion.move_end_effector', parameters={
            'axis':inputs.side, 'target':target, 'target_revision':target['revision'],
            'purpose':'extract', 'motion_mode':'ik',
            'position_tolerance_m':POSITION_TOLERANCE_M}, timeout_seconds=inputs.timeout_seconds))
    # execute rejects failed/stopped/interrupted results. Runtime owns the
    # measured completion check; do not sample the pose again after it settles.
    state.liftoff_verification = dict(
        source='runtime_action_result', action_key='grasp:liftoff-ik',
        action_status=result.status, requested_lift_m=state.liftoff_height_m,
        position_tolerance_m=POSITION_TOLERANCE_M, confirmed=True)
    ctx.checkpoint(state)
