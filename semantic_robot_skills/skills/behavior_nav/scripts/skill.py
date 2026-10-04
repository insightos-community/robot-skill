"""Resolve an object approach pose, navigate, and check measured arrival."""

import json
from math import cos, sin

from semantic_robot_skill_sdk import Action, SkillContext, StopOutcome, StopRequest

from .models import Approach, Arrival, Input, Result, Selection, State
from .runtime import ActionFailed, execute


async def run(ctx: SkillContext) -> None:
    inputs = ctx.input(Input)
    state = ctx.load_state(State, default=State())
    try:
        navigation = await execute(
            ctx,
            state,
            "nav:follow",
            Action(
                type="navigation.follow_route",
                parameters={
                    "route_ref": json.dumps(
                        ({"object_name": inputs.object_name} if inputs.object_name is not None else
                         {"candidate_object_names": inputs.candidate_object_names}), ensure_ascii=False
                    ),
                    "navigation_purpose": "transit",
                    "maximum_speed_mps": inputs.maximum_speed_mps,
                    "minimum_clearance_m": inputs.minimum_clearance_m,
                },
                timeout_seconds=inputs.timeout_seconds,
            ),
        )
        approach = (navigation.output or {}).get("object_approach")
        if not isinstance(approach, dict):
            raise ValueError("导航能力缺少 object_approach")
        state.approach = Approach.model_validate(approach)
        requested = [inputs.object_name] if inputs.object_name is not None else inputs.candidate_object_names
        if state.approach.object_name not in requested:
            raise ValueError("导航能力返回的目标物体与请求不匹配")
        if (navigation.output or {}).get('selection') is not None:
            state.selection = Selection.model_validate(navigation.output['selection'])
        target = state.approach.target
        position = target.position_m
        yaw = target.yaw_rad
        ctx.checkpoint(state)
        result = await execute(
            ctx,
            state,
            "nav:verify",
            Action(
                type="navigation.verify_arrival",
                parameters={
                    "target": {
                        "target_ref": state.approach.object_name,
                        "pose": {
                            "frame_id": target.frame_id,
                            "position_m": position,
                            "orientation_xyzw": [0.0, 0.0, sin(yaw / 2), cos(yaw / 2)],
                        },
                    },
                    "navigation_purpose": "transit",
                    "arrival_radius_m": inputs.arrival_radius_m,
                    "require_visual_confirmation": False,
                },
                timeout_seconds=15,
            ),
        )
        arrival = Arrival.model_validate(result.output)
        if (
            arrival.verdict != "achieved"
            or arrival.distance_to_target_m > inputs.arrival_radius_m
        ):
            ctx.fail("ARRIVAL_NOT_CONFIRMED", "到达检查未通过", state.evidence_refs)
            return
    except ActionFailed:
        return
    except ValueError as error:
        ctx.fail("NAVIGATION_RESULT_INVALID", str(error), state.evidence_refs)
        return
    state.stage = "completed"
    ctx.checkpoint(state)
    ctx.complete(
        Result(
            target=target,
            approach=state.approach,
            selection=state.selection,
            final_pose_ref=arrival.final_pose_ref,
            distance_to_target_m=arrival.distance_to_target_m,
            evidence_refs=state.evidence_refs,
        )
    )


async def on_stop(ctx: SkillContext, request: StopRequest) -> StopOutcome:
    result = await ctx.execute_stop(
        "nav:stop",
        Action(
            type="navigation.follow_route",
            parameters={"reason": request.reason, "mode": request.mode},
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
        summary="底盘保持已确认" if confirmed else "底盘保持尚未确认",
        evidence_refs=result.evidence_refs,
    )
