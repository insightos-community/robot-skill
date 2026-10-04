"""语义导航 Skill 的 Stage 推进、反馈处理和恢复入口。"""

from __future__ import annotations

from typing import Any

from .stage_evidence import capture_stage_rgb

from semantic_robot_skill_sdk import (
    ActionResult,
    SkillContext,
    StopOutcome,
    StopRequest,
)

from .controller import (
    LOCAL_REPLAN_LIMIT,
    assess_follow_feedback,
    build_follow_route_action,
    build_get_robot_state_action,
    build_locate_object_action,
    build_plan_route_action,
    build_safe_stop_action,
    build_verify_arrival_action,
    build_verify_tool_load_action,
)
from .models import (
    CarryingObjectState,
    FollowRouteOutput,
    RobotStateValue,
    NavigationAgentDecision,
    PlanRouteOutput,
    SemanticNavigationInput,
    SemanticNavigationResult,
    SemanticNavigationState,
    TargetPoseValue,
    ToolLoadObservationValue,
    VerifyArrivalOutput,
)


async def run(ctx: SkillContext) -> None:
    """每次从持久化检查点重新进入当前 Stage。"""

    ctx.check_cancelled()
    skill_input = ctx.input(SemanticNavigationInput)
    state = ctx.load_state(
        SemanticNavigationState,
        default=SemanticNavigationState(),
    )
    ctx.log(
        "info",
        "进入语义导航 Stage",
        skill_name="semantic-navigation",
        skill_version="0.4.7",
        stage=state.stage,
    )
    ctx.report(
        "stage.running",
        summary=f"正在执行语义导航阶段：{state.stage}",
        stage=state.stage,
        stage_status="running",
        expectation=_stage_expectation(state.stage),
        next_step="continue_stage",
        evidence_refs=ctx.recent_evidence(),
    )

    if state.stage in {"validate_target", "navigate", "verify_arrival"}:
        await capture_stage_rgb(ctx, state, skill_name="semantic-navigation")

    if state.stage == "validate_target":
        await _validate_target(ctx, skill_input, state)
        return
    if state.stage == "plan_route":
        await _plan_route(ctx, skill_input, state)
        return
    if state.stage == "navigate":
        await _navigate(ctx, skill_input, state)
        return
    if state.stage == "verify_arrival":
        await _verify_arrival(ctx, skill_input, state)
        return
    ctx.fail("INVALID_STAGE", f"未知语义导航 Stage：{state.stage}")


async def _validate_target(
    ctx: SkillContext,
    skill_input: SemanticNavigationInput,
    state: SemanticNavigationState,
) -> None:
    """复核 Agent 已解析目标；Skill 不再自行做候选搜索或语义消歧。"""

    if skill_input.carried_object_ref and not state.carrying_verified:
        carrying = await _observe_carrying_state(
            ctx, skill_input.carried_object_ref, key_suffix="before-navigation"
        )
        if carrying is None:
            return
        state.carrying_object = carrying
        state.carrying_verified = True

    target = state.target or skill_input.target
    state.target = target.model_copy(deep=True)
    state.route_ref = None
    state.stage = "plan_route"
    ctx.checkpoint(state)
    ctx.report(
        "stage.completed",
        stage="validate_target",
        stage_status="completed",
        summary="已复核 Robot Agent 提供的导航目标",
        evidence_refs=ctx.recent_evidence(),
    )


async def _plan_route(
    ctx: SkillContext,
    skill_input: SemanticNavigationInput,
    state: SemanticNavigationState,
) -> None:
    """规划路线；失败时先使用本地恢复预算，再请求 Agent。"""

    state.plan_attempt += 1
    ctx.checkpoint(state)
    result = await ctx.execute(
        key=f"plan-route:{state.plan_attempt}",
        action=build_plan_route_action(skill_input, state),
    )
    if result.status == "succeeded" and result.output is not None:
        output = PlanRouteOutput.model_validate(result.output)
        state.route_revision += 1
        state.route_ref = output.route_ref
        if output.resolved_target is not None:
            # target_ref仍是Agent选择的业务目标；这里只采用Provider基于实时占据
            # 栅格解析出的安全Pose，避免执行后又拿原始边缘坐标判定未到达。
            state.target = output.resolved_target.model_copy(deep=True)
        state.stage = "navigate"
        ctx.checkpoint(state)
        ctx.report(
            "stage.completed",
            stage="plan_route",
            stage_status="completed",
            summary="语义导航路线已经生成",
            evidence_refs=result.evidence_refs,
        )
        return

    if not _target_pose_is_invalid(result) and _consume_replan_budget(
        skill_input, state
    ):
        ctx.checkpoint(state)
        ctx.log(
            "warning",
            "路线规划失败，准备进行本地重试",
            replan_count=state.replan_count,
            evidence_refs=result.evidence_refs,
        )
        return

    invalid_target = _target_pose_is_invalid(result)
    decision = await _request_agent(
        ctx,
        state,
        reason=(
            "导航目标位姿不可用于路径规划，需要重新计算目标"
            if invalid_target
            else "路线规划恢复预算已经耗尽"
        ),
        evidence_refs=result.evidence_refs,
        extra_context={
            "last_action_error": {
                "code": result.error_code,
                "message": result.error_message,
                "physical_effect": result.physical_effect,
            }
        },
    )
    _apply_agent_decision(ctx, skill_input, state, decision)


def _target_pose_is_invalid(result: ActionResult) -> bool:
    """识别不会因重复同一请求而改变的目标错误。"""

    message = result.error_message or ""
    return result.physical_effect == "none" and (
        result.error_code in {"INVALID_TARGET", "TARGET_IN_OBSTACLE"}
        or "终点位于障碍物中" in message
        or "终点不在占据栅格内" in message
    )


async def _navigate(
    ctx: SkillContext,
    skill_input: SemanticNavigationInput,
    state: SemanticNavigationState,
) -> None:
    """执行路线并根据流式反馈及时停止、重规划或重新解析目标。"""

    handle = await ctx.start_action(
        key=f"follow-route:{state.route_revision}",
        action=build_follow_route_action(skill_input, state),
    )
    stop_reason: str | None = None
    target_changed = False

    async for feedback in handle.feedback():
        ctx.check_cancelled()
        assessment = assess_follow_feedback(feedback, state)
        ctx.report(
            "stage.progress",
            summary=feedback.message or "正在执行语义导航",
            evidence_refs=feedback.evidence_refs,
            stage="navigate",
            stage_status="running",
            expectation=_stage_expectation("navigate"),
            observation_summary=feedback.message or feedback.status,
            deviation=assessment.reason if assessment.decision != "continue" else None,
            progress=feedback.progress,
            next_step=assessment.decision,
        )
        if assessment.decision == "continue":
            continue
        if assessment.decision == "critical_stop":
            # critical 反馈到达时，执行侧必须已经先停止当前 Action。
            result = await handle.result()
            if result.status == "interrupted":
                _fail_interrupted_action(ctx, result, "严重异常后无法确认导航终态")
                return
            load_details = _carrying_load_details(feedback)
            if (
                state.carrying_object is not None
                and load_details is not None
                and _stopped_for_unstable_load(result)
            ):
                await _recover_after_load_stop(
                    ctx,
                    skill_input,
                    state,
                    result,
                    load_details=load_details,
                )
                return
            decision = await _request_agent(
                ctx,
                state,
                reason=assessment.reason or "导航因严重异常停止",
                evidence_refs=result.evidence_refs,
                extra_context={"last_observation": load_details}
                if load_details is not None
                else None,
            )
            _apply_agent_decision(ctx, skill_input, state, decision)
            return
        stop_reason = assessment.reason or "当前导航路线已经失效"
        break

    if stop_reason is not None:
        result = await handle.stop(reason=stop_reason)
        if result.status == "interrupted":
            _fail_interrupted_action(ctx, result, "停止导航后无法确认 Action 终态")
            return
        if _consume_replan_budget(skill_input, state):
            state.route_ref = None
            state.stage = "plan_route"
            ctx.checkpoint(state)
            return
        decision = await _request_agent(
            ctx,
            state,
            reason="导航局部重新规划次数已经耗尽",
            evidence_refs=result.evidence_refs,
        )
        _apply_agent_decision(ctx, skill_input, state, decision)
        return

    result = await handle.result()
    if result.status == "interrupted":
        _fail_interrupted_action(ctx, result, "无法确认导航 Action 是否已经结束")
        return
    if result.status == "succeeded" and result.output is not None:
        output = FollowRouteOutput.model_validate(result.output)
        state.final_pose_ref = output.final_pose_ref
        state.distance_to_target_m = output.distance_to_target_m
        state.stage = "verify_arrival"
        ctx.checkpoint(state)
        ctx.report(
            "stage.completed",
            stage="navigate",
            stage_status="completed",
            summary="路线执行完成，准备独立验证到达状态",
            evidence_refs=result.evidence_refs,
        )
        return
    if _stopped_for_unstable_load(result) and state.carrying_object is not None:
        await _recover_after_load_stop(
            ctx,
            skill_input,
            state,
            result,
            load_details=None,
        )
        return
    if _consume_replan_budget(skill_input, state):
        state.route_ref = None
        state.stage = "plan_route"
        ctx.checkpoint(state)
        return
    decision = await _request_agent(
        ctx,
        state,
        reason="导航 Action 失败且局部恢复预算已经耗尽",
        evidence_refs=result.evidence_refs,
    )
    _apply_agent_decision(ctx, skill_input, state, decision)


async def _verify_arrival(
    ctx: SkillContext,
    skill_input: SemanticNavigationInput,
    state: SemanticNavigationState,
) -> None:
    """独立验证到达结果，不把路线 Action 成功直接当作 Skill 成功。"""

    state.verification_count += 1
    ctx.checkpoint(state)
    result = await ctx.execute(
        key=f"verify-arrival:{state.verification_count}",
        action=build_verify_arrival_action(skill_input, state),
    )
    if result.status != "succeeded" or result.output is None:
        decision = await _request_agent(
            ctx,
            state,
            reason="到达验证 Action 执行失败",
            evidence_refs=result.evidence_refs,
        )
        _apply_agent_decision(ctx, skill_input, state, decision)
        return

    output = VerifyArrivalOutput.model_validate(result.output)
    state.final_pose_ref = output.final_pose_ref
    state.distance_to_target_m = output.distance_to_target_m
    if skill_input.carried_object_ref is not None:
        refreshed = await _observe_carrying_state(
            ctx, skill_input.carried_object_ref, key_suffix="after-navigation"
        )
        if refreshed is None:
            return
        state.carrying_object = refreshed
        state.carrying_verified = True

    ctx.checkpoint(state)
    if output.verdict == "achieved":
        await capture_stage_rgb(ctx, state, skill_name="semantic-navigation", point="completed")
        _complete_navigation(ctx, skill_input, state, result.evidence_refs)
        return
    if output.verdict == "not_achieved" and _consume_replan_budget(
        skill_input,
        state,
    ):
        state.route_ref = None
        state.stage = "plan_route"
        ctx.checkpoint(state)
        return
    decision = await _request_agent(
        ctx,
        state,
        reason="无法可靠判断是否已经到达语义目标",
        evidence_refs=result.evidence_refs,
        extra_context={"monitor_recommended": True},
    )
    _apply_agent_decision(ctx, skill_input, state, decision)


async def _request_agent(
    ctx: SkillContext,
    state: SemanticNavigationState,
    *,
    reason: str,
    evidence_refs: list[str],
    extra_context: dict[str, Any] | None = None,
) -> NavigationAgentDecision:
    """创建可恢复的 Agent 请求，并只发送当前问题所需摘要。"""

    if state.pending_decision_key is None:
        state.decision_revision += 1
        state.pending_decision_key = (
            f"navigation:{state.stage}:decision:{state.decision_revision}"
        )
        ctx.checkpoint(state)
    request_context: dict[str, Any] = {
        "stage": state.stage,
        "state": state.model_dump(mode="json"),
        "replan_count": state.replan_count,
        "evidence_refs": evidence_refs,
    }
    request_context.update(extra_context or {})
    ctx.report(
        "decision.required",
        summary=reason,
        evidence_refs=evidence_refs,
        stage=state.stage,
        stage_status="waiting_agent",
        expectation=_stage_expectation(state.stage),
        deviation=reason,
        next_step="request_agent",
    )
    decision = await ctx.request_agent(
        key=state.pending_decision_key,
        reason=reason,
        context=request_context,
        response_model=NavigationAgentDecision,
    )
    state.pending_decision_key = None
    ctx.checkpoint(state)
    return decision


def _apply_agent_decision(
    ctx: SkillContext,
    skill_input: SemanticNavigationInput,
    state: SemanticNavigationState,
    decision: NavigationAgentDecision,
) -> None:
    """把类型化 Agent 决策映射为本 Skill 明确声明的 Stage 变化。"""

    if decision.action == "abort_subtask":
        ctx.fail("AGENT_ABORTED", decision.reason, decision.evidence_refs)
        return
    if decision.action == "replace_target":
        if decision.target is None:
            ctx.fail(
                "AGENT_TARGET_MISSING",
                "replace_target 决策缺少已解析目标",
                decision.evidence_refs,
            )
            return
        state.target = decision.target.model_copy(deep=True)
        state.route_ref = None
        state.replan_count = 0
        state.stage = "validate_target"
        ctx.checkpoint(state)
        return
    if decision.action == "replan_route":
        state.route_ref = None
        state.replan_count = 0
        state.stage = "plan_route" if state.target is not None else "validate_target"
        ctx.checkpoint(state)
        return
    if decision.action == "retry_navigation":
        state.stage = "navigate" if state.route_ref else "plan_route"
        ctx.checkpoint(state)
        return
    if decision.action == "recheck_arrival":
        state.stage = "verify_arrival"
        ctx.checkpoint(state)
        return
    ctx.fail(
        "INVALID_AGENT_DECISION",
        f"当前恢复点不支持 Agent 决策：{decision.action}",
        decision.evidence_refs,
    )


async def on_stop(ctx: SkillContext, request: StopRequest) -> StopOutcome:
    """通过 Runtime 专用停止入口形成可审计的安全停止结果。"""

    result = await ctx.execute_stop(
        key=f"semantic-navigation:stop:{request.id}",
        action=build_safe_stop_action(reason=request.reason, mode=request.mode),
    )
    output = result.output or {}
    safe = result.status == "succeeded" and bool(output.get("safe", True))
    physical_state = str(output.get("physical_state", "unknown"))
    evidence = _merge_evidence(result.evidence_refs, ctx.recent_evidence())
    return ctx.stop_outcome(
        safe=safe,
        summary=(
            "语义导航已经安全停止" if safe else "无法确认语义导航已经进入安全状态"
        ),
        physical_state=physical_state,
        requires_intervention=not safe,
        evidence_refs=evidence,
    )


def _consume_replan_budget(
    skill_input: SemanticNavigationInput,
    state: SemanticNavigationState,
) -> bool:
    """由脚本统一维护本地重新规划预算。"""

    if state.replan_count >= LOCAL_REPLAN_LIMIT:
        return False
    state.replan_count += 1
    return True


def _stopped_for_unstable_load(result: ActionResult) -> bool:
    output = result.output or {}
    stop_evidence = output.get("stop_evidence")
    return (
        result.status == "stopped"
        and isinstance(stop_evidence, dict)
        and stop_evidence.get("reason") == "carrying_load_unstable"
    )


def _carrying_load_details(feedback: Any) -> dict[str, Any] | None:
    """保留触发停止的真实传感摘要，避免Agent只看到笼统的“持物异常”。"""

    for item in feedback.observations:
        if item.kind == "navigation.carrying_load":
            return dict(item.value or {})
    return None


async def _recover_after_load_stop(
    ctx: SkillContext,
    skill_input: SemanticNavigationInput,
    state: SemanticNavigationState,
    result: ActionResult,
    *,
    load_details: dict[str, Any] | None,
) -> None:
    """hold后只允许一次基于新证据的降速恢复，禁止原路线原速度重放。

    底盘停止会消除运动激励，因此先重新读取工具承载和物体位姿。只有双侧
    重新稳定时才从当前位置规划新路线；如果仍不稳定，Navigation Skill没有
    重新抓取能力，应明确失败并由Task Recovery处理，而不是等待大模型反复
    选择retry_navigation。
    """

    if state.replan_count >= LOCAL_REPLAN_LIMIT:
        ctx.fail(
            "CARRYING_LOAD_UNSTABLE",
            _load_failure_message(load_details, repeated=True),
            result.evidence_refs,
        )
        return
    refreshed = await _observe_carrying_state(
        ctx,
        state.carrying_object.object_ref,
        key_suffix=f"after-load-stop-{state.replan_count + 1}",
    )
    if refreshed is None:
        return
    if not _consume_replan_budget(skill_input, state):
        ctx.fail(
            "CARRYING_LOAD_UNSTABLE",
            _load_failure_message(load_details, repeated=True),
            result.evidence_refs,
        )
        return
    state.carrying_object = refreshed
    state.carrying_verified = True
    state.route_ref = None
    state.stage = "plan_route"
    ctx.checkpoint(state)
    ctx.report(
        "stage.progress",
        summary="停止后持物状态已恢复稳定，将从当前位置降速重新规划一次",
        evidence_refs=_merge_evidence(result.evidence_refs, refreshed.evidence_refs),
        stage="navigate",
        stage_status="running",
        expectation=_stage_expectation("navigate"),
        observation_summary=_load_failure_message(load_details, repeated=False),
        next_step="replan",
    )


def _load_failure_message(
    load_details: dict[str, Any] | None, *, repeated: bool
) -> str:
    reasons = (load_details or {}).get("reasons")
    reason_text = ", ".join(str(item) for item in reasons or [])
    prefix = "降速恢复后携物状态再次异常" if repeated else "携物状态曾触发安全停止"
    return f"{prefix}：{reason_text}" if reason_text else prefix


def _complete_navigation(
    ctx: SkillContext,
    skill_input: SemanticNavigationInput,
    state: SemanticNavigationState,
    evidence_refs: list[str],
) -> None:
    """在最终证据完整时形成 Skill 结果。"""

    if state.final_pose_ref is None or state.distance_to_target_m is None:
        ctx.fail("MISSING_ARRIVAL_STATE", "缺少最终位姿或到目标距离")
        return
    if skill_input.carried_object_ref is not None and not state.carrying_verified:
        ctx.fail(
            "CARRYING_STATE_UNVERIFIED",
            "携物导航尚未确认到达后仍稳定持有物体",
        )
        return
    # Skill 终态不会替代当前 Stage 的终态。二者分别服务于执行收敛与时间线展示；
    # 必须先关闭 verify_arrival，避免前端看到 Execution completed 而 Stage 仍 running。
    ctx.report(
        "stage.completed",
        stage="verify_arrival",
        stage_status="completed",
        summary="已独立确认 Robot 到达导航目标",
        evidence_refs=evidence_refs,
    )
    ctx.complete(
        SemanticNavigationResult(
            reached=True,
            target_ref=skill_input.target.target_ref,
            final_pose_ref=state.final_pose_ref,
            distance_to_target_m=state.distance_to_target_m,
            route_ref=state.route_ref,
            evidence_refs=_merge_evidence(evidence_refs, ctx.recent_evidence()),
        )
    )


async def _observe_carrying_state(
    ctx: SkillContext,
    object_ref: str,
    *,
    key_suffix: str,
) -> CarryingObjectState | None:
    """组合原始Robot状态、通用载荷判断和实时物体观测。

    RobotState不维护全局HeldObjectState，Skill也不复制上一项Execution的业务
    结果。工具集合来自当前Profile返回的tote_clamp，即使某侧失联也不会因按
    接触筛选而被悄悄遗漏。
    """

    state_result = await ctx.execute(
        key=f"navigation.get-state:{key_suffix}",
        action=build_get_robot_state_action(),
    )
    if state_result.status != "succeeded":
        ctx.fail(
            "ROBOT_STATE_READ_FAILED",
            "无法读取当前Robot状态",
            state_result.evidence_refs,
        )
        return None
    state_observation = next(
        (
            item
            for item in reversed(state_result.observations)
            if item.kind == "robot.state"
        ),
        None,
    )
    if state_observation is None:
        ctx.fail("ROBOT_STATE_MISSING", "RobotState未返回robot.state Observation")
        return None
    robot_state = RobotStateValue.model_validate(state_observation.value or {})
    if robot_state.robot_id != ctx.robot_ref:
        ctx.fail("ROBOT_STATE_MISMATCH", "RobotState不属于当前Execution绑定的Robot")
        return None
    tool_refs = tuple(
        ref
        for ref, tool in robot_state.tool_states.items()
        if tool.kind == "tote_clamp"
    )
    if len(tool_refs) != 2:
        ctx.fail("CARRYING_TOOLS_UNAVAILABLE", "当前Robot未返回两个周转箱夹具")
        return None

    load_result = await ctx.execute(
        key=f"navigation.verify-load:{key_suffix}",
        action=build_verify_tool_load_action(tool_refs),
    )
    load_observation = next(
        (
            item
            for item in reversed(load_result.observations)
            if item.kind == "robot.tool_load"
        ),
        None,
    )
    if load_result.status != "succeeded" or load_observation is None:
        ctx.fail(
            "TOOL_LOAD_READ_FAILED", "无法验证当前双工具承载", load_result.evidence_refs
        )
        return None
    load = ToolLoadObservationValue.model_validate(load_observation.value or {})
    if (
        not load.condition_satisfied
        or load.slip_detected
        or load.overload_detected
        or load.sensor_fault
    ):
        ctx.fail(
            "TOOL_LOAD_UNSTABLE",
            "双工具当前没有形成稳定承载",
            load_result.evidence_refs,
        )
        return None

    object_result = await ctx.execute(
        key=f"navigation.locate-object:{key_suffix}",
        action=build_locate_object_action(object_ref),
    )
    object_observation = next(
        (
            item
            for item in reversed(object_result.observations)
            if item.kind == "target_pose"
        ),
        None,
    )
    if object_result.status != "succeeded" or object_observation is None:
        ctx.fail(
            "CARRIED_OBJECT_NOT_OBSERVED",
            "无法重新观测请求携带的物体",
            object_result.evidence_refs,
        )
        return None
    target = TargetPoseValue.model_validate(object_observation.value or {})
    if target.object_ref != object_ref:
        ctx.fail("CARRIED_OBJECT_MISMATCH", "实时物体观测与携物引用不一致")
        return None

    tool_poses = {}
    for ref in tool_refs:
        side = robot_state.tool_states[ref].side
        end_effector = robot_state.end_effectors.get(side)
        if end_effector is None:
            ctx.fail("END_EFFECTOR_POSE_MISSING", f"RobotState未返回{ref}的末端位姿")
            return None
        tool_poses[ref] = target.pose.model_copy(
            update={
                "frame_id": end_effector.frame_id,
                "position_m": end_effector.position,
                "orientation_xyzw": end_effector.quaternion_xyzw,
            }
        )
    evidence = _merge_evidence(
        state_result.evidence_refs,
        load_result.evidence_refs,
        object_result.evidence_refs,
        state_observation.evidence_refs,
        load_observation.evidence_refs,
        object_observation.evidence_refs,
    )
    return CarryingObjectState(
        object_ref=object_ref,
        robot_ref=ctx.robot_ref,
        tool_refs=tool_refs,
        tool_poses=tool_poses,
        object_pose=target.pose,
        object_size_m=target.extent_m,
        robot_state_generation=robot_state.generation,
        evidence_refs=evidence,
    )


def _fail_interrupted_action(
    ctx: SkillContext,
    result: ActionResult,
    message: str,
) -> None:
    """物理终态未知时明确失败，禁止脚本继续启动新 Action。"""

    ctx.fail("ACTION_STATE_UNKNOWN", message, result.evidence_refs)


def _merge_evidence(*groups: list[str]) -> list[str]:
    """保持顺序去重，避免向 Agent 和结果重复传递证据引用。"""

    merged: list[str] = []
    for group in groups:
        for item in group:
            if item and item not in merged:
                merged.append(item)
    return merged


def _stage_expectation(stage: str) -> str:
    """返回当前导航 Stage 的用户可理解期望。"""

    return {
        "validate_target": "目标输入有效；携物时组合实时Robot、工具载荷和物体观测",
        "plan_route": "生成满足速度、净空和携物约束的可执行路线",
        "navigate": "Robot 沿当前路线前进且定位、障碍和携物状态保持正常",
        "verify_arrival": "独立确认 Robot 到达目标，并复核同一物体仍被稳定持有",
    }.get(stage, "导航执行保持在已声明的安全范围内")
