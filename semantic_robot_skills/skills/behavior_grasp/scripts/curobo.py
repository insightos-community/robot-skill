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

"""Task geometry in Skill; collision planning and timed execution in Runtime SDK."""
from semantic_robot_skill_sdk import Action
from .geometry import grasp_plan
from .closing_axis import closing_axis_candidates
from .runtime import execute, ActionFailed
from .enclosure import enclosure
from .approach import offset_for


def action(inputs, pose, purpose, *, plan_only=False, offset=None, attached_object_ref=None, contact_object_ref=None, allow_support_contact=False):
    return Action(type='motion.move_end_effector', parameters=dict(
        axis=inputs.side, target=pose.model_dump(mode='json'), purpose=purpose,
        target_revision=pose.revision, motion_mode='curobo', plan_only=plan_only,
        use_torso=inputs.operation_torso is not None, approach_offset=offset,
        **({'pregrasp_planner':inputs.pregrasp_planner} if offset is not None else {}),
        attached_object_ref=attached_object_ref, contact_object_ref=contact_object_ref,
        **({'allow_support_contact':True} if allow_support_contact else {})), timeout_seconds=inputs.timeout_seconds)


async def approach(ctx, state, inputs, read_state):
    sides = [inputs.side] if inputs.side != 'auto' else ['left', 'right']
    # Opening is part of the collision model. Keep both candidate hands open during selection.
    for side in sides:
        await execute(ctx, state, f'grasp:curobo-open:{side}', Action(type='gripper.set_opening',
            parameters={'tools':[{'tool_ref':f'component://gripper/{side}',
                'target_position_m':inputs.opening_m, 'maximum_force_n':inputs.maximum_force_n}]},
            timeout_seconds=inputs.timeout_seconds))
    if not state.operation_verified:
        measured = await read_state('grasp:curobo-state')
        candidates=[]
        for side in sides:
            selected=inputs.model_copy(update={'side':side})
            initial=grasp_plan(selected.model_copy(update={'orientation_xyzw':None}),state.located,state.object_orientation,state.principal_axis)
            plans=closing_axis_candidates(selected,initial,state.object_orientation,state.principal_axis,
                measured['joints'],side)
            for index,plan in enumerate(plans):
                if not enclosure(state.located,plan.grasp,side,[inputs.opening_m]*2)['valid']:continue
                preference=(plan.orientation_selection.get('approach_preference_score',0.),plan.orientation_selection['rotation_distance_rad'])
                candidates.append((preference,index,side,selected,plan))
        candidates.sort(key=lambda item:item[:3])
        failures=[]
        for _,index,side,selected,plan in candidates:
            key=f'grasp:curobo-plan:{side}:{index}'
            ctx.check_cancelled()
            state.stage=key;ctx.checkpoint(state)
            result=state.results.get(key)
            if result is None:
                ctx.report('stage.running',stage=key,stage_status='running',summary='cuRobo 规划候选')
                result=await ctx.execute(key=key,action=action(selected,plan.grasp,'alignment',plan_only=True,offset=offset_for(state.located,plan.grasp,side,inputs.opening_m)[0],contact_object_ref=inputs.object_ref))
                ctx.check_cancelled()
                state.results[key]=result
                state.evidence_refs=list(dict.fromkeys(state.evidence_refs+result.evidence_refs))
                ctx.checkpoint(state)
            if result.status != 'succeeded':
                # Only a completed numerical failure permits another no-motion candidate.
                if result.status != 'failed' or result.error_code not in {
                    'collision_plan_failed','invalid_start_state','unsupported_approach_direction'}:
                    ctx.fail(result.error_code or 'CUROBO_PLANNING_FAILED',result.error_message or result.status,state.evidence_refs)
                    raise ActionFailed
                failures.append(dict(side=side,candidate=index,reason=result.error_message,code=result.error_code))
                continue
            state.selected_side=side;state.plan=plan;state.operation_verified=True
            state.enclosure_checks['planned']=enclosure(state.located,plan.grasp,side,[inputs.opening_m]*2)
            state.approach_offset,diagnostic=offset_for(state.located,plan.grasp,side,inputs.opening_m)
            state.plan.orientation_selection.update(approach=diagnostic)
            state.side_selection=dict(selected_side=side,planner=inputs.pregrasp_planner,rejected_candidates=failures,
                selection='first_complete_contact_path_feasible')
            ctx.checkpoint(state)
            break
        if not state.operation_verified:
            raise ValueError('cuRobo 未找到可行抓取路径: '+str(failures))
    selected=inputs.model_copy(update={'side':state.selected_side})
    await execute(ctx,state,'grasp:approach-curobo',action(selected,state.plan.grasp,'alignment',offset=state.approach_offset,contact_object_ref=inputs.object_ref))
    return selected
