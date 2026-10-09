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

"""Place through existing EEF/open/hold actions and Task-provided geometry."""
import asyncio
from .holding_side import infer_holding_side,SAMPLE_COUNT,SAMPLE_INTERVAL_SECONDS
from semantic_robot_skill_sdk import Action,SkillContext,StopRequest,StopOutcome
from .models import Input,State,Result,HeldGeometry
from .perception import held_geometry,rim_top_region
from .geometry import build_plan
from .frames import eef_in_body
from .runtime import execute,ActionFailed


async def resolve_holding_side(ctx,state,inputs):
    if state.hand_selection is None:
        samples=[]
        for index in range(SAMPLE_COUNT):
            key=f"place:gripper-state:{index}"
            if index and key not in state.results:
                await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
            result=await execute(ctx,state,key,Action(type="gripper.get_state",parameters={},timeout_seconds=15))
            tools=(result.output or {}).get("tools")
            if not isinstance(tools,list):raise ValueError("gripper.get_state 缺少双手反馈")
            samples.append(tools)
        state.hand_selection=infer_holding_side(samples,inputs.object_ref)
        ctx.checkpoint(state)
    side=state.hand_selection["holding_side"]
    if side is None:
        ctx.fail("HOLDING_SIDE_UNCERTAIN","夹爪反馈未能确定目标的唯一持物侧，尚未移动手臂",state.evidence_refs)
        raise ActionFailed
    if inputs.side not in ("auto",side):
        ctx.fail("HOLDING_SIDE_MISMATCH",
            f"反馈推断持物侧为 {side}，与指定的 {inputs.side} 不一致",state.evidence_refs)
        raise ActionFailed
    return Input.model_validate({**inputs.model_dump(),"side":side})

async def run(ctx:SkillContext):
    inputs=ctx.input(Input);state=ctx.load_state(State,default=State())
    try:
        inputs=await resolve_holding_side(ctx,state,inputs)
        if state.plan is None:
            measured=await execute(ctx,state,"place:read-eef",Action(type="motion.get_end_effector_state",
                parameters={"axis":inputs.side},timeout_seconds=inputs.timeout_seconds))
            readings=[o.value for o in measured.observations if o.kind=="manipulation.end_effector_state" and o.value.get("axis")==inputs.side]
            if len(readings)!=1:raise ValueError("放置需要持物手唯一的末端位姿")
            robot_state = None
            if readings[0]["frame_id"] not in ("body", "base", "base_link", "base_footprint"):
                measured=await execute(ctx,state,"place:read-body",Action(type="robot.get_state",
                    parameters={},timeout_seconds=inputs.timeout_seconds))
                states=[o.value for o in measured.observations if o.kind=="robot.state"]
                if len(states)!=1:raise ValueError("放置坐标转换需要机器人关节与底盘姿态")
                robot_state=states[0]
            current=eef_in_body(readings[0],robot_state,inputs.side)
            if inputs.object_size_m is None:
                parameters={"object_ref":inputs.object_ref,"prompt":inputs.prompt,
                    "minimum_confidence":inputs.minimum_confidence,
                    "pose_hint":{"frame_id":"body",**current.model_dump(mode="json")}}
                if inputs.perception_profile is not None:parameters["model_profile"]=inputs.perception_profile
                located=await execute(ctx,state,"place:locate-held",Action(type="perception.locate_object",
                    parameters=parameters,timeout_seconds=inputs.timeout_seconds))
                state.object_geometry=held_geometry(inputs,(located.output or {}).get("verification"),current)
            else:
                state.object_geometry=HeldGeometry(object_size_m=inputs.object_size_m,
                    eef_from_object=inputs.eef_from_object,source="task_input")
            resolved=inputs.model_copy(update={"object_size_m":state.object_geometry.object_size_m,
                "eef_from_object":state.object_geometry.eef_from_object})
            if inputs.target.region is None:
                parameters={"object_ref":inputs.target.object_ref,"prompt":inputs.target.prompt,
                    "minimum_confidence":inputs.minimum_confidence}
                profile=inputs.target.perception_profile or inputs.perception_profile
                if profile is not None:parameters["model_profile"]=profile
                located=await execute(ctx,state,"place:locate-target",Action(type="perception.locate_object",
                    parameters=parameters,timeout_seconds=inputs.timeout_seconds))
                measured=await execute(ctx,state,"place:read-body",Action(type="robot.get_state",
                    parameters={},timeout_seconds=inputs.timeout_seconds))
                states=[o.value for o in measured.observations if o.kind=="robot.state"]
                if len(states)!=1:raise ValueError("顶部投放需要底盘姿态")
                region,state.target_geometry=rim_top_region(inputs,located.output or {})
                resolved=resolved.model_copy(update={"target":inputs.target.model_copy(update={"region":region})})
            else:state.target_geometry={"source":"task_input"}
            state.plan=build_plan(resolved,current);ctx.checkpoint(state)
        async def move(w, carrying):
            await execute(ctx,state,w.key,Action(type="motion.move_end_effector",parameters={"axis":inputs.side,"motion_mode":"curobo","use_torso":True,
                "attached_object_ref":inputs.object_ref if carrying else None,
                "target":{"frame_id":"body",**w.body_from_eef.model_dump(mode="json"),"revision":w.key},
                "target_revision":w.key,"purpose":w.purpose},timeout_seconds=inputs.timeout_seconds))
        for step in state.plan.approach:await move(step,True)
        released=await execute(ctx,state,"place:release",Action(type="gripper.set_opening",parameters={
            "tools":[{"tool_ref":f"component://gripper/{inputs.side}","target_position_m":inputs.release_opening_m,
                      "maximum_force_n":inputs.maximum_force_n,"hold":False}]},timeout_seconds=inputs.timeout_seconds))
        if (released.output or {}).get("verified") is not True:raise ValueError("释放开度缺少到位确认")
        if state.release_verification is None:
            feedback=await execute(ctx,state,"place:verify-release",Action(type="gripper.get_state",
                parameters={},timeout_seconds=inputs.timeout_seconds))
            tools=[t for t in (feedback.output or {}).get("tools",[]) if t.get("side")==inputs.side]
            if len(tools)!=1 or "held_object_ref" not in tools[0]:
                raise ValueError("释放后缺少持物手的物体身份反馈")
            if tools[0]["held_object_ref"] is not None:
                raise ValueError("张爪后仍报告持物，停止撤手")
            state.release_verification={"side":inputs.side,"opening_verified":True,
                "held_object_ref":None,"source":"gripper_feedback"}
            ctx.checkpoint(state)
        for step in state.plan.retreat:await move(step,False)
    except ActionFailed:return
    except (ValueError,KeyError,TypeError) as error:
        ctx.fail("PLACEMENT_FAILED",str(error),state.evidence_refs);return
    state.stage="completed";ctx.checkpoint(state)
    ctx.complete(Result(holding_side=inputs.side,hand_selection=state.hand_selection,object_ref=inputs.object_ref,target_ref=inputs.target.object_ref,requested_relation=inputs.target.relation,
        release_mode=inputs.release_mode,target_object_body=state.plan.target_object_body,
        release_object_body=state.plan.release_object_body,object_geometry=state.object_geometry,target_geometry=state.target_geometry,
        release_orientation=state.plan.release_orientation,release_verification=state.release_verification,
        evidence_refs=state.evidence_refs))

async def on_stop(ctx:SkillContext,request:StopRequest)->StopOutcome:
    result=await ctx.execute_stop("place:hold",Action(type="gripper.hold_object",parameters={
        "reason":request.reason,"mode":request.mode,"preserve_tool_state":True},timeout_seconds=15))
    confirmed=result.status=="succeeded" and (result.output or {}).get("stop_evidence",{}).get("holding") is True
    return ctx.stop_outcome(safe=confirmed,physical_state="hold" if confirmed else "unknown",
        requires_intervention=not confirmed,summary="保持已确认" if confirmed else "保持未确认",evidence_refs=result.evidence_refs)
