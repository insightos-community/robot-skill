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

"""Try direct carry before any motion; fall back only from a rejected preplan."""
import asyncio

from .carry import natural_carry
from .curobo import action
from .liftoff import prepare_lift, lift
from .runtime import execute, ActionFailed


async def carry(ctx, state, inputs, read_state):
    if state.carry_mode is None:
        # Preserve an old checkpoint which has already entered the lift/carry path.
        legacy = state.liftoff_target is not None or 'grasp:carry-curobo' in state.results
        state.carry_mode = 'lift_first' if legacy else inputs.carry_mode
        ctx.checkpoint(state)
    # Retain the original pre-motion EEF measurement for placement handoff and
    # a potential fallback. This is not an extra gripper-state confirmation.
    await prepare_lift(ctx, state, inputs, read_state)
    if state.carry_pose is None:
        state.carry_pose = await asyncio.to_thread(
            natural_carry, state.reference_torso, inputs.side, state.plan.grasp)
        ctx.checkpoint(state)
    if state.carry_mode == 'direct':
        key = 'grasp:carry-direct-plan'
        ctx.check_cancelled()
        result = state.results.get(key)
        if result is None:
            state.stage = key
            ctx.checkpoint(state)
            ctx.report('stage.running', stage=key, stage_status='running', summary='规划直接携物')
            result = await ctx.execute(key=key, action=action(inputs, state.carry_pose, 'extract',
                plan_only=True, attached_object_ref=inputs.object_ref, allow_support_contact=True))
            ctx.check_cancelled()
            state.results[key] = result
            state.evidence_refs = list(dict.fromkeys(state.evidence_refs+result.evidence_refs))
            ctx.checkpoint(state)
        if result.status == 'succeeded':
            # Any failure here propagates: execution may already have moved.
            await execute(ctx, state, 'grasp:carry-direct', action(inputs, state.carry_pose,
                'extract', attached_object_ref=inputs.object_ref, allow_support_contact=True))
            return
        if result.status != 'failed' or result.error_code not in {
                'support_departure_unavailable', 'collision_plan_failed', 'invalid_start_state'}:
            ctx.fail(result.error_code or 'CARRY_PLANNING_FAILED',
                     result.error_message or result.status, state.evidence_refs)
            raise ActionFailed
        state.carry_mode = 'lift_first'
        state.carry_fallback_reason = result.error_code
        ctx.checkpoint(state)
        ctx.report('carry.fallback', summary='直接携物规划失败，使用抬升后携物：'+result.error_code)
    await lift(ctx, state, inputs, read_state)
    await execute(ctx, state, 'grasp:carry-curobo', action(inputs, state.carry_pose,
        'extract', attached_object_ref=inputs.object_ref))
