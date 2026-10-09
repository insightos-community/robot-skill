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

"""Restore torso and both arms with one SDK-owned cuRobo joint-space motion."""
from semantic_robot_skill_sdk import Action,SkillContext,StopOutcome,StopRequest
from .geometry import fixed_posture,holding,VERIFY_TOLERANCE_RAD
from .models import Input,State,Result
from .runtime import execute,ActionFailed

def positions(result):
    rows=[o.value for o in result.observations if o.kind=="robot.state"]
    if len(rows)!=1:raise ValueError("需要唯一机器人关节状态")
    joints=rows[0]["joints"]
    return dict(zip(joints["names"],joints["positions_rad"]))

async def run(ctx:SkillContext):
    inputs=ctx.input(Input);state=ctx.load_state(State,default=State())
    try:
        if state.joint_goals is None:
            grips=await execute(ctx,state,"upright:read-holding",Action(type="gripper.get_state",parameters={},timeout_seconds=15))
            state.holding_side,state.held_object_ref=holding(grips.output["tools"])
            state.joint_goals=fixed_posture(state.holding_side)
            measured=await execute(ctx,state,"upright:read",Action(type="robot.get_state",parameters={},timeout_seconds=15))
            q=positions(measured)
            state.skipped_motion=max(abs(q[n]-v) for n,v in state.joint_goals.items())<=VERIFY_TOLERANCE_RAD
            ctx.checkpoint(state)
        if not state.skipped_motion:
            await execute(ctx,state,"upright:restore",Action(type="motion.move_arm_joint",parameters={
                "arm_side":state.holding_side or "left","motion_mode":"curobo",
                "joint_names":list(state.joint_goals),"positions_rad":list(state.joint_goals.values())},
                timeout_seconds=inputs.timeout_seconds,label="躯干与双臂联合恢复直立持物姿态"))
        measured=await execute(ctx,state,"upright:verify",Action(type="robot.get_state",parameters={},timeout_seconds=15))
        q=positions(measured);error=max(abs(q[n]-v) for n,v in state.joint_goals.items())
        if error>VERIFY_TOLERANCE_RAD:
            ctx.fail("POSTURE_RESTORE_UNCONFIRMED",f"全身关节最大误差 {error:.4f} rad",state.evidence_refs);return
        grips=await execute(ctx,state,"upright:verify-holding",Action(type="gripper.get_state",parameters={},timeout_seconds=15))
        side,obj=holding(grips.output["tools"])
        if (side,obj)!=(state.holding_side,state.held_object_ref):
            ctx.fail("HOLDING_CHANGED","姿态恢复后持物身份发生变化",state.evidence_refs);return
    except ActionFailed:return
    except (ValueError,KeyError,TypeError) as error:
        ctx.fail("POSTURE_STATE_INVALID",str(error),state.evidence_refs);return
    state.stage="completed";ctx.checkpoint(state)
    ctx.complete(Result(holding_side=state.holding_side,held_object_ref=state.held_object_ref,
        joint_goals=state.joint_goals,measured_positions_rad={n:q[n] for n in state.joint_goals},
        maximum_error_rad=error,skipped_motion=state.skipped_motion,evidence_refs=state.evidence_refs))

async def on_stop(ctx:SkillContext,request:StopRequest)->StopOutcome:
    result=await ctx.execute_stop("upright:hold",Action(type="gripper.hold_object",parameters={
        "reason":request.reason,"mode":request.mode,"preserve_tool_state":True},timeout_seconds=15))
    confirmed=result.status=="succeeded" and (result.output or {}).get("stop_evidence",{}).get("holding") is True
    return ctx.stop_outcome(safe=confirmed,physical_state="hold" if confirmed else "unknown",
        requires_intervention=not confirmed,summary="保持已确认" if confirmed else "保持未确认",evidence_refs=result.evidence_refs)
