"""Task-supplied geometry, fresh recognition, approach, contact close and lift."""

import asyncio
import numpy as np
from scipy.spatial.transform import Rotation
from .observation import natural_pose

from semantic_robot_skill_sdk import Action, SkillContext, StopOutcome, StopRequest

from .geometry import check_torso, grasp_plan, torso_offset_targets
from .models import Input, LocatedObject, Pose, Result, State, TorsoAngles, TorsoOffset, TorsoObservation, ObjectOrientation
from .runtime import ActionFailed, execute
from .torso_motion import duration_seconds as torso_duration
from .grasp_axes import axes as grasp_axes
from .enclosure import require_enclosure, EnclosureError, measured_pose_in_body, measured_fingers
from .held_geometry import observation_anchor, result_fields


def tool_ref(inputs: Input) -> str:
    if inputs.side not in ("left", "right"):
        raise ValueError("抓取侧尚未选定，不能下发夹爪动作")
    return f"component://gripper/{inputs.side}"


def tool_command(inputs: Input, opening: float) -> dict:
    return {
        "tool_ref": tool_ref(inputs),
        "target_position_m": opening,
        "maximum_force_n": inputs.maximum_force_n,
    }


async def run(ctx: SkillContext) -> None:
    inputs = ctx.input(Input)
    state = ctx.load_state(State, default=State())
    if state.selected_side is not None:
        inputs = inputs.model_copy(update={"side":state.selected_side})
    elif inputs.side != "auto":
        state.selected_side = inputs.side
    async def read_state(key):
        result = await execute(ctx, state, key, Action(
            type="robot.get_state", parameters={}, timeout_seconds=inputs.timeout_seconds))
        readings = [o.value for o in result.observations if o.kind == "robot.state"]
        if not readings or not isinstance(readings[-1], dict):
            raise ValueError("姿态计算需要 robot.state 观测")
        return readings[-1]

    try:
        if state.reference_torso is None:
            initial = await read_state("grasp:reference-state")
            joints = dict(zip(initial["joints"]["names"],initial["joints"]["positions_rad"]))
            state.reference_torso = [joints[f"torso_joint{i}"] for i in range(1,5)]
            ctx.checkpoint(state)
        if state.stage == "completed":
            ctx.complete(Result(object_ref=inputs.object_ref,grasp_side=state.selected_side,
                side_selection=state.side_selection,enclosure_checks=state.enclosure_checks,
                approach_selection=state.plan.orientation_selection,tool_ref=tool_ref(inputs),
                grasp_pose=state.plan.grasp,carry_pose=state.carry_pose,liftoff_verification=state.liftoff_verification,
                carry_mode=state.carry_mode, carry_fallback_reason=state.carry_fallback_reason, **result_fields(state), evidence_refs=state.evidence_refs))
            return
        if state.torso_targets is None:
            computed = await execute(ctx, state, "grasp:compute-torso", Action(
                type="motion.compute_observation_pose", parameters={"object_ref":inputs.object_ref},
                timeout_seconds=inputs.timeout_seconds))
            output = computed.output or {}
            state.observation_source = output["source"]
            state.object_orientation = ObjectOrientation.model_validate(output["object_orientation"])
            if state.object_orientation.object_ref != inputs.object_ref:
                raise ValueError("物体朝向与抓取目标不匹配")
            measured = await read_state("grasp:observation-state")
            base = measured["base_pose"]
            if base["frame_id"] != state.observation_source["frame_id"]:
                raise ValueError("目标几何与底盘状态需要一致的坐标系")
            body = np.eye(4)
            body[:3,:3] = Rotation.from_quat(base["orientation_xyzw"]).as_matrix()
            body[:3,3] = base["position"]
            target = np.asarray(state.observation_source["object_position"])
            state.perception_hint = {"frame_id":"body",
                "position_m":(body[:3,:3].T @ (target-body[:3,3])).tolist(),
                "orientation_xyzw":[0.,0.,0.,1.]}
            joints = dict(zip(measured["joints"]["names"], measured["joints"]["positions_rad"]))
            current = [joints[f"torso_joint{i}"] for i in range(1,5)]
            if isinstance(inputs.torso,TorsoObservation):
                captured = await execute(ctx,state,"grasp:observation-camera",Action(
                    type="sensor.capture_rgbd",parameters={"sensor_ids":["head_camera"]},
                    timeout_seconds=inputs.timeout_seconds))
                frames = (captured.output or {}).get("artifact_candidates",[])
                camera = next((f for f in frames if f["sensor_id"]=="head_camera"),None)
                if camera is None:raise ValueError("观察姿态计算需要头部相机标定")
                calibration = camera["calibration"]
                planned = await asyncio.to_thread(natural_pose,
                    state.observation_source["object_bounds"], body, current,
                    np.asarray(calibration["camera_pose_body"]) @ np.diag([1.,-1.,-1.,1.]),
                    calibration["intrinsic_matrix"], [camera["height"],camera["width"]])
                state.torso_targets = [check_torso(planned["positions_rad"])]
                state.observation_diagnostics = planned["diagnostics"]
            elif isinstance(inputs.torso,TorsoAngles):
                state.torso_targets = [check_torso(inputs.torso.positions_rad)]
            elif isinstance(inputs.torso,TorsoOffset):
                state.torso_targets = torso_offset_targets(current,inputs.torso.offset_m)[-1:]
            else:
                state.torso_targets = []
            state.torso_durations = [torso_duration(current,target) for target in state.torso_targets]
            ctx.checkpoint(state)
        for index,target in enumerate(state.torso_targets):
            await execute(ctx,state,f"grasp:torso:{index:03d}",Action(
                type="motion.set_torso_state",parameters={"positions_rad":target,
                    "trajectory_duration_seconds":state.torso_durations[index]},
                timeout_seconds=inputs.timeout_seconds))

        if state.located is None:
            await read_state("grasp:locate-before-state")
            parameters = {"object_ref":inputs.object_ref,"prompt":inputs.prompt,
                "minimum_confidence":inputs.minimum_confidence,"pose_hint":state.perception_hint}
            if inputs.perception_profile is not None:parameters["model_profile"]=inputs.perception_profile
            located = await execute(ctx,state,"grasp:locate",Action(type="perception.locate_object",
                parameters=parameters,timeout_seconds=inputs.timeout_seconds))
            detected = LocatedObject.model_validate((located.output or {}).get("verification"))
            if detected.object_ref != inputs.object_ref:
                raise ValueError("识别物体与抓取目标不一致")
            await read_state("grasp:locate-after-state")
            anchor = observation_anchor(detected, state.results["grasp:locate-before-state"],
                state.results["grasp:locate-after-state"])
            state.located, state.observation_anchor = detected, anchor
            state.principal_axis = (located.output or {}).get("principal_axis")
            if inputs.orientation_xyzw is None:
                grasp_axes(state.principal_axis, state.object_orientation)
            if inputs.side != "auto":
                state.plan = grasp_plan(inputs,state.located,state.object_orientation,state.principal_axis)
            ctx.checkpoint(state)

        from .curobo import approach
        inputs = await approach(ctx, state, inputs, read_state)
        state.closing_started = True
        ctx.checkpoint(state)
        closed = await execute(
            ctx,
            state,
            "grasp:close",
            Action(
                type="gripper.close",
                parameters={
                    "object_ref": inputs.object_ref,
                    "tools": [{"tool_ref": tool_ref(inputs), "maximum_force_n": inputs.maximum_force_n}],
                    "candidate_id": state.plan.candidate_id,
                    "grasp_pose": state.plan.grasp.model_dump(mode="json"),
                },
                timeout_seconds=inputs.timeout_seconds,
            ),
        )
        contact = closed.output or {}
        if (
            contact.get("contact") is not True
            or contact.get("object_ref") != inputs.object_ref
            or contact.get("candidate_id") != state.plan.candidate_id
        ):
            ctx.fail(
                "GRASP_CONTACT_UNCONFIRMED",
                "闭爪结果缺少本次抓取的接触确认",
                state.evidence_refs,
            )
            return
        from .carry_motion import carry
        await carry(ctx, state, inputs, read_state)
    except ActionFailed:
        return
    except EnclosureError as error:
        state.stage = "grasp:enclosure-verify" if error.phase == "measured" else "grasp:enclosure-plan"
        state.enclosure_checks[error.phase] = error.diagnostic
        ctx.checkpoint(state)
        ctx.fail("GRASP_ENCLOSURE_UNCONFIRMED" if error.phase == "measured" else "GRASP_ENCLOSURE_INVALID",
                 str(error), state.evidence_refs)
        return
    except (ValueError, KeyError, TypeError) as error:
        ctx.fail("GRASP_DATA_INVALID", str(error), state.evidence_refs)
        return
    state.stage = "completed"
    ctx.checkpoint(state)
    ctx.complete(
        Result(
            object_ref=inputs.object_ref,
            grasp_side=state.selected_side,
            side_selection=state.side_selection,
            enclosure_checks=state.enclosure_checks,
            approach_selection=state.plan.orientation_selection,
            tool_ref=tool_ref(inputs),
            grasp_pose=state.plan.grasp,
            carry_pose=state.carry_pose,
            liftoff_verification=state.liftoff_verification,
            carry_mode=state.carry_mode, carry_fallback_reason=state.carry_fallback_reason, **result_fields(state),
            evidence_refs=state.evidence_refs,
        )
    )


async def on_stop(ctx: SkillContext, request: StopRequest) -> StopOutcome:
    inputs = ctx.input(Input)
    state = ctx.load_state(State, default=State())
    parameters = {
        "reason": request.reason,
        "mode": request.mode,
        "preserve_tool_state": True,
    }
    if state.closing_started:
        parameters["object_ref"] = inputs.object_ref
    result = await ctx.execute_stop(
        "grasp:hold",
        Action(
            type="gripper.hold_object",
            parameters=parameters,
            timeout_seconds=15,
        ),
    )
    stop_evidence = (result.output or {}).get("stop_evidence")
    confirmed = (
        result.status == "succeeded"
        and isinstance(stop_evidence, dict)
        and stop_evidence.get("holding") is True
    )
    return ctx.stop_outcome(
        safe=confirmed,
        physical_state="hold" if confirmed else "unknown",
        requires_intervention=not confirmed,
        summary="当前姿态保持已确认；保留夹爪状态"
        if confirmed
        else "当前姿态保持尚未确认",
        evidence_refs=result.evidence_refs,
    )
