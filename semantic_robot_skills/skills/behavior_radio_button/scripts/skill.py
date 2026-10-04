"""Natural held-arm posture, wrist inspection and one axial-roll retry."""

import asyncio
from math import dist, radians
from semantic_robot_skill_sdk import Action, SkillContext, StopOutcome, StopRequest
from .geometry import measured_pose, orientation_error_rad
from .chest_planner import ChestPlanError
from .manipulation_planner import plan_natural, plan_observation, plan_roll, plan_button, as_pose
from .models import Input, JointSample, LocatedButton, Pose, Result, State, MotionPlan
from .runtime import ActionFailed, execute
from .holding_side import infer_holding_side, SAMPLE_COUNT, SAMPLE_INTERVAL_SECONDS
from .retraction import plan_two_stage, plan_flip_and_retract


async def read_pose(ctx, state, inputs, prefix, side) -> Pose:
    torso = await execute(
        ctx,
        state,
        f"{prefix}:torso",
        Action(
            type="motion.get_torso_state",
            parameters={},
            timeout_seconds=15,
        ),
    )
    sample = (torso.output or {}).get("torso")
    if not isinstance(sample, dict) or sample.get("units") != "rad":
        raise ValueError("Torso state must contain positions_rad with units=rad")
    result = await execute(
        ctx,
        state,
        f"{prefix}:eef",
        Action(
            type="motion.get_end_effector_state",
            parameters={"axis": side},
            timeout_seconds=15,
        ),
    )
    observations = [
        obs
        for obs in result.observations
        if obs.kind == "manipulation.end_effector_state"
        and isinstance(obs.value, dict)
        and obs.value.get("axis") == side
    ]
    if len(observations) != 1 or not isinstance(observations[0].value, dict):
        raise ValueError(
            "Expected one end-effector state observation for the holding side"
        )
    observation = observations[0]
    return measured_pose(
        observation.value,
        sample["positions_rad"],
        observation.revision or observation.id,
        observation.observed_at.isoformat(),
    )


async def move(ctx, state, inputs, key, side, pose, purpose="alignment", *, motion_mode="curobo"):
    return await execute(
        ctx,
        state,
        key,
        Action(
            type="motion.move_end_effector",
            parameters={
                "axis": side,
                "motion_mode": motion_mode,
                **({"use_torso": False} if motion_mode == "curobo"
                   else {"position_tolerance_m": .003}),
                "target": pose.model_dump(mode="json"),
                "purpose": purpose,
                "target_revision": pose.revision,
            },
            timeout_seconds=inputs.timeout_seconds,
        ),
    )


async def locate_button(ctx, state, inputs, attempt) -> LocatedButton | None:
    result = await execute(
        ctx,
        state,
        f"radio:locate:{attempt}",
        Action(
            type="perception.locate_object",
            parameters={
                "object_ref": inputs.button_ref,
                "prompt": inputs.button_prompt,
                "minimum_confidence": inputs.minimum_confidence,
                "model_profile": inputs.perception_profile,
            },
            timeout_seconds=inputs.timeout_seconds,
        ),
        allow_not_found=True,
    )
    if result.status == "failed":
        return None
    output = result.output or {}
    if output.get("model_profile") != inputs.perception_profile:
        raise ValueError("Perception result does not match the requested wrist profile")
    button = LocatedButton.model_validate(output.get("verification"))
    if (
        button.object_ref != inputs.button_ref
        or button.identity_confidence < inputs.minimum_confidence
    ):
        raise ValueError("Button identity or confidence does not match the request")
    return button


async def joint_sample(ctx, state, key):
    result = await execute(ctx, state, key, Action(type="robot.get_state", parameters={}, timeout_seconds=15))
    observations = [o for o in result.observations if o.kind == "robot.state" and isinstance(o.value, dict)]
    if not observations:
        raise ValueError("robot.get_state 缺少机器人关节观测")
    return JointSample.model_validate(observations[-1].value["joints"])


async def plan_motion(ctx, state, inputs, key, side, planner, *args):
    if key not in state.plans:
        measured = await read_pose(ctx, state, inputs, key+":before", side)
        torso = state.results[key+":before:torso"].output["torso"]["positions_rad"]
        joints = await joint_sample(ctx, state, key+":joints")
        if key == "radio:natural":
            natural,retract=await asyncio.to_thread(plan_two_stage,measured,joints,torso,side,planner)
            state.plans[key]=natural
            state.plans['radio:retract']=retract
        elif key == "radio:axial-flip":
            flip,retract=await asyncio.to_thread(plan_flip_and_retract,measured,joints,torso,side,planner)
            state.plans[key]=flip
            state.plans['radio:flip-retract']=retract
        else:
            pose,diagnostics = await asyncio.to_thread(planner,measured,joints,torso,side,*args)
            state.plans[key] = MotionPlan(pose=pose,side=side,diagnostics=diagnostics)
        ctx.checkpoint(state)
    return state.plans[key]


async def execute_plan(ctx, state, inputs, key, plan, *, motion_mode="curobo"):
    await move(ctx, state, inputs, f"{key}:eef", plan.side, plan.pose, motion_mode=motion_mode)
    actual = await read_pose(ctx, state, inputs, key+":after", plan.side)
    if dist(actual.position_m, plan.pose.position_m) > .03 or orientation_error_rad(actual, plan.pose) > radians(4):
        ctx.fail("ARM_POSE_UNCONFIRMED", "动作结束后实测末端超出 3 cm / 4° 容差", state.evidence_refs)
        raise ActionFailed
    return actual


async def search_button(ctx, state, inputs):
    for attempt in (1, 2):
        if attempt == 2:
            natural = await plan_motion(ctx, state, inputs, "radio:restore-natural", inputs.holding_side, plan_natural)
            state.holding_pose = await execute_plan(ctx, state, inputs, "radio:restore-natural", natural)
            flip = await plan_motion(ctx, state, inputs, "radio:axial-flip", inputs.holding_side, plan_roll)
            state.holding_pose = await execute_plan(ctx, state, inputs, "radio:axial-flip", flip)
            state.holding_pose = await execute_plan(ctx, state, inputs, "radio:flip-retract", state.plans['radio:flip-retract'])
            state.flipped = True
            ctx.checkpoint(state)
        key = f"radio:inspect:{attempt}"
        captured = await execute(ctx,state,key+":camera",Action(type="sensor.capture_rgbd",
            parameters={"sensor_ids":[inputs.capture_sensor]},timeout_seconds=inputs.timeout_seconds))
        camera = next((f for f in (captured.output or {}).get("artifact_candidates",[])
                       if f["sensor_id"]==inputs.capture_sensor),None)
        if camera is None:raise ValueError("观察规划需要操作手相机标定")
        def observe(measured, joints, torso, side):
            return plan_observation(state.holding_pose,measured,joints,torso,side,
                                    camera)
        inspection = await plan_motion(ctx, state, inputs, key, inputs.operating_side, observe)
        actual = await execute_plan(ctx, state, inputs, key, inspection)
        button = await locate_button(ctx, state, inputs, attempt)
        state.recognition_attempts = attempt
        ctx.checkpoint(state)
        if button is not None:
            return button, actual
    ctx.fail("BUTTON_NOT_FOUND", "自然屈肘和轴向翻转 180° 后均未检测到按钮", state.evidence_refs)
    raise ActionFailed


async def resolve_hand_roles(ctx, state, inputs):
    if state.hand_selection is None:
        samples=[]
        for index in range(SAMPLE_COUNT):
            key=f"radio:gripper-state:{index}"
            if index and key not in state.results:
                await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
            result=await execute(ctx,state,key,Action(type="gripper.get_state",parameters={},timeout_seconds=15))
            tools=(result.output or {}).get("tools")
            if not isinstance(tools,list):raise ValueError("gripper.get_state 缺少双手反馈")
            samples.append(tools)
        state.hand_selection=infer_holding_side(samples)
        ctx.checkpoint(state)
    side=state.hand_selection["operating_side"]
    if side is None:
        ctx.fail("HOLDING_SIDE_UNCERTAIN","双手力、开度和稳定性未能确定唯一持物侧，尚未移动手臂",state.evidence_refs)
        raise ActionFailed
    if inputs.operating_side not in ("auto",side):
        ctx.fail("OPERATING_SIDE_MISMATCH",
            f"反馈推断操作侧为 {side}，与指定的 {inputs.operating_side} 不一致；省略 operating_side 可自动选手",state.evidence_refs)
        raise ActionFailed
    # Validate explicit camera/profile against the detected side before motion.
    return Input.model_validate({**inputs.model_dump(),"operating_side":side})


async def run(ctx: SkillContext) -> None:
    inputs = ctx.input(Input)
    state = ctx.load_state(State, default=State())
    try:
        inputs = await resolve_hand_roles(ctx,state,inputs)
        natural = await plan_motion(ctx, state, inputs, "radio:natural", inputs.holding_side, plan_natural)
        state.holding_pose = await execute_plan(ctx, state, inputs, "radio:natural", natural)
        state.holding_pose = await execute_plan(ctx, state, inputs, "radio:retract", state.plans['radio:retract'])
        ctx.checkpoint(state)
        closed = await execute(ctx, state, "radio:close-operating",
            Action(type="gripper.set_opening", parameters={"tools": [{
                "tool_ref": f"component://gripper/{inputs.operating_side}", "target_position_m": 0.0,
                "maximum_force_n": inputs.maximum_force_n}]}, timeout_seconds=inputs.timeout_seconds))
        if (closed.output or {}).get("verified") is not True:
            ctx.fail("OPERATING_GRIPPER_UNCONFIRMED", "操作手夹爪闭合缺少到位确认", state.evidence_refs)
            return
        button, inspection = await search_button(ctx, state, inputs)
        if "radio:approach-button" not in state.plans:
            measured = await read_pose(ctx, state, inputs, "radio:button-plan", inputs.operating_side)
            torso = state.results["radio:button-plan:torso"].output["torso"]["positions_rad"]
            joints = await joint_sample(ctx, state, "radio:button-plan:joints")
            state.approach_pose, state.button_plan_diagnostics = await asyncio.to_thread(plan_button,
                measured, joints, torso, inputs.operating_side, button)
            state.plans['radio:approach-button'] = MotionPlan(pose=state.approach_pose,
                side=inputs.operating_side,diagnostics=state.button_plan_diagnostics)
            state.button = button
            ctx.checkpoint(state)
        await execute_plan(ctx, state, inputs, "radio:approach-button", state.plans['radio:approach-button'],
                           motion_mode="ik")
    except ActionFailed:
        return
    except ChestPlanError as error:
        ctx.fail("RADIO_POSE_UNREACHABLE", str(error), state.evidence_refs)
        return
    except (ValueError, KeyError, TypeError) as error:
        ctx.fail("RADIO_DATA_INVALID", str(error), state.evidence_refs)
        return
    state.stage = "completed"
    ctx.checkpoint(state)
    ctx.complete(Result(operating_side=inputs.operating_side, holding_side=inputs.holding_side,
        hand_selection=state.hand_selection,
        button=state.button, requested_pose=state.approach_pose, flipped=state.flipped,
        recognition_attempts=state.recognition_attempts, evidence_refs=state.evidence_refs))


async def on_stop(ctx: SkillContext, request: StopRequest) -> StopOutcome:
    result = await ctx.execute_stop(
        "radio:hold",
        Action(
            type="gripper.hold_object",
            parameters={
                "reason": request.reason,
                "mode": request.mode,
                "preserve_tool_state": True,
            },
            timeout_seconds=15,
        ),
    )
    evidence = (result.output or {}).get("stop_evidence")
    confirmed = (
        result.status == "succeeded"
        and isinstance(evidence, dict)
        and evidence.get("holding") is True
    )
    return ctx.stop_outcome(
        safe=confirmed,
        physical_state="hold" if confirmed else "unknown",
        requires_intervention=not confirmed,
        summary="双手当前姿态和夹爪状态保持已确认" if confirmed else "保持状态尚未确认",
        evidence_refs=result.evidence_refs,
    )
