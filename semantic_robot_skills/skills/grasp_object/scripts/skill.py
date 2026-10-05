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

"""去码垛抓取 Robot Skill 的 Stage 脚本。

脚本负责保存 Stage 期望、消费原子能力反馈、执行有限的局部恢复，并在超出
局部边界时向 Robot Agent 请求类型化决定。机械臂轨迹和高频安全控制仍由
Pilot 后面的能力实现负责。
"""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .stage_evidence import capture_stage_rgb

from semantic_robot_skill_sdk import (
    ActionFeedback,
    ActionResult,
    Observation,
    SkillContext,
    StopOutcome,
    StopRequest,
)

from .controller import (
    LIFT_HEIGHT_TOLERANCE_M,
    MAXIMUM_PREGRASP_ERROR_M,
    MINIMUM_TARGET_CONFIDENCE,
    SECONDARY_REALIGN_SHIFT_M,
    STABLE_DURATION_MS,
    DepalletizingGraspController,
    skill_input_pose,
)
from .models import (
    GraspAgentDecision,
    GraspCandidate,
    HeldObjectState,
    GraspCandidatesValue,
    GraspObjectInput,
    GraspObjectResult,
    GraspObjectState,
    GraspStrategy,
    GraspVerificationValue,
    RobotStateValue,
    ToolLoadObservationValue,
    TransportPostureValue,
    LiftProgressValue,
    PregraspStateValue,
    TargetPoseValue,
)


CONTROLLER_NAME = "depalletizing.grasp"
ValueModel = TypeVar("ValueModel", bound=BaseModel)


async def run(ctx: SkillContext) -> None:
    """从最近检查点进入当前 Stage，并推进一次可恢复的执行。"""

    ctx.check_cancelled()
    skill_input = ctx.input(GraspObjectInput)
    state = ctx.load_state(
        GraspObjectState,
        GraspObjectState(active_strategy=skill_input.preferred_strategy),
    )
    controller = _controller(ctx)
    ctx.log(
        "info",
        "进入抓取 Stage",
        skill_name="grasp-object",
        skill_version="0.4.23",
        stage=state.stage,
    )
    ctx.report(
        "stage.running",
        summary=f"正在执行抓取阶段：{state.stage}",
        stage=state.stage,
        stage_status="running",
        expectation=_stage_expectation(state.stage),
        next_step="continue_stage",
        evidence_refs=state.evidence_refs,
    )

    if state.stage in {"observe_target", "approach", "grasp", "lift_and_verify", "prepare_transport"}:
        await capture_stage_rgb(ctx, state, skill_name="grasp-object")

    if state.stage == "observe_target":
        await _observe_target(ctx, controller, skill_input, state)
    elif state.stage == "approach":
        await _approach(ctx, controller, skill_input, state)
    elif state.stage == "grasp":
        await _grasp(ctx, controller, skill_input, state)
    elif state.stage == "lift_and_verify":
        await _lift_and_verify(ctx, controller, skill_input, state)
    elif state.stage == "prepare_transport":
        await _prepare_transport(ctx, controller, skill_input, state)
    else:
        ctx.fail("UNKNOWN_STAGE", f"未知抓取 Stage：{state.stage}")


def _controller(ctx: SkillContext) -> DepalletizingGraspController:
    """读取显式注册的 Controller，不做隐式实现回退。"""

    controller = ctx.controller(CONTROLLER_NAME)
    if not isinstance(controller, DepalletizingGraspController):
        raise TypeError(f"{CONTROLLER_NAME} 不是抓取 Controller")
    return controller


async def _observe_target(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
) -> None:
    """建立带版本的目标观测和抓取候选，满足第一阶段期望。"""

    target_observation = None
    if state.observation_attempts == 0 and state.target_observation_ref is None:
        target_observation = ctx.latest_observation(
            "target_pose",
            subject_ref=skill_input.object_ref,
            max_age_ms=2_000,
        )

    if target_observation is None:
        attempt = state.observation_attempts
        result = await ctx.execute(
            f"grasp:observe:{attempt}",
            controller.locate_target(skill_input),
        )
        _remember_result_evidence(state, result)
        target_observation = _find_observation(result, "target_pose")
        ctx.checkpoint(state)

        if result.status != "succeeded" or target_observation is None:
            await _retry_observation_or_ask_agent(
                ctx,
                skill_input,
                state,
                reason="目标观测失败或没有返回 target_pose",
                details={"action_status": result.status},
            )
            return

    target = _parse_value(target_observation, TargetPoseValue)
    if (
        target is None
        or target.object_ref != skill_input.object_ref
        or target.identity_confidence < MINIMUM_TARGET_CONFIDENCE
    ):
        await _retry_observation_or_ask_agent(
            ctx,
            skill_input,
            state,
            reason="目标身份或位姿观测置信度不足",
            details={
                "observation_ref": target_observation.id,
                "confidence": target_observation.confidence,
            },
        )
        return

    target_revision = target_observation.revision or target.pose.revision
    state.target_observation_ref = target_observation.id
    state.target_revision = target_revision
    state.target_pose = target.pose
    state.target_extent_m = target.extent_m
    _remember_evidence(state, target_observation.evidence_refs)
    ctx.checkpoint(state)

    candidate_result = await ctx.execute(
        f"grasp:candidates:{target_revision}:{state.active_strategy}",
        controller.generate_candidates(
            skill_input,
            target_pose=target.pose,
            object_extent_m=target.extent_m,
            target_revision=target_revision,
            active_strategy=state.active_strategy,
        ),
    )
    _remember_result_evidence(state, candidate_result)
    candidate_observation = _find_observation(candidate_result, "grasp_candidates")
    candidate_value = _parse_value(candidate_observation, GraspCandidatesValue)
    candidates = _rank_candidates(
        candidate_value.candidates if candidate_value else [],
        state.active_strategy,
    )

    if (
        candidate_result.status != "succeeded"
        or candidate_value is None
        or candidate_value.target_revision != target_revision
        or not candidates
    ):
        await _retry_observation_or_ask_agent(
            ctx,
            skill_input,
            state,
            reason="当前目标观测没有生成有效抓取候选",
            details={
                "action_status": candidate_result.status,
                "target_revision": target_revision,
            },
        )
        return

    state.candidates = candidates
    state.selected_candidate_id = candidates[0].candidate_id
    state.approach_plan = []
    state.approach_cursor = 0
    state.pregrasp_observation_ref = None
    state.primary_clamped = False
    state.extraction_planned_from_contact = False
    state.extraction_primary_side = None
    state.extraction_completed = False
    state.extraction_reobserved = False
    state.extraction_adjustments = 0
    state.secondary_replan_attempted = False
    state.object_held = False
    state.lift_completed = False
    state.stage = "approach"
    state.plan_revision += 1
    if candidate_observation:
        _remember_evidence(state, candidate_observation.evidence_refs)
    ctx.checkpoint(state)
    ctx.report(
        "stage.completed",
        stage="observe_target",
        stage_status="completed",
        summary=(
            f"已确认目标 {skill_input.object_ref} 的动态位姿并选择候选 "
            f"{state.selected_candidate_id}"
        ),
        evidence_refs=state.evidence_refs,
    )


async def _approach(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
) -> None:
    """执行当前候选的短 Approach 计划并核对预抓取期望。"""

    candidate = _selected_candidate(state)
    if candidate is None or state.target_revision is None:
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="Approach 阶段缺少候选或目标版本",
            details={},
        )
        return

    if not state.approach_plan:
        state.approach_plan = controller.plan_approach(
            skill_input,
            candidate=candidate,
            target_revision=state.target_revision,
        )
        state.approach_cursor = 0
        state.pregrasp_observation_ref = None
        state.plan_revision += 1
        ctx.checkpoint(state)

    while state.approach_cursor < len(state.approach_plan):
        index = state.approach_cursor
        action = state.approach_plan[index]
        result = await ctx.execute(
            f"grasp:approach:{state.plan_revision}:{index}",
            action,
        )
        _remember_result_evidence(state, result)
        pregrasp_observation = _find_observation(result, "pregrasp_state")
        if pregrasp_observation:
            state.pregrasp_observation_ref = pregrasp_observation.id
            _remember_evidence(state, pregrasp_observation.evidence_refs)

        if result.status != "succeeded":
            physical_effect = result.physical_effect
            if (
                action.parameters.get("purpose") == "insert"
                and physical_effect == "none"
            ):
                # insert 已经驱动末端进入箱沿接触区。即使 Runtime 没有确认接触，
                # 也不能断言物体完全没受影响，否则候选切换会立即执行一条新的
                # transfer 轨迹。保守到这一层即可：保持现场并交给 Robot Agent，
                # 不把普通不可达的 pregrasp/transfer 失败一并升级。
                physical_effect = "possible"
            ctx.checkpoint(state)
            await _recover_approach_or_ask_agent(
                ctx,
                skill_input,
                state,
                reason=f"Approach Action {action.type} 执行失败",
                details={
                    "action_status": result.status,
                    "physical_effect": physical_effect,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                },
            )
            return

        state.approach_cursor += 1
        ctx.checkpoint(state)

    pregrasp_observation = (
        ctx.observation(state.pregrasp_observation_ref)
        if state.pregrasp_observation_ref
        else None
    )
    pregrasp = _parse_value(pregrasp_observation, PregraspStateValue)
    if (
        pregrasp is None
        or pregrasp.candidate_id != candidate.candidate_id
        or pregrasp.target_revision != state.target_revision
        or not pregrasp.reached
    ):
        await _recover_approach_or_ask_agent(
            ctx,
            skill_input,
            state,
            reason="末端没有满足当前候选的预抓取期望",
            details={
                "pregrasp_observation_ref": state.pregrasp_observation_ref,
                "maximum_error_m": MAXIMUM_PREGRASP_ERROR_M,
                "hook_contacts": pregrasp.hook_contacts if pregrasp else {},
                # 所有 insert Action 均已成功返回，但重新观测没有满足候选。
                # 这只能说明接触没有保持，不能证明箱体从未被触碰或推动。
                "physical_effect": "possible",
            },
        )
        return

    state.stage = "grasp"
    ctx.checkpoint(state)
    ctx.report(
        "stage.completed",
        stage="approach",
        stage_status="completed",
        summary=f"已到达候选 {candidate.candidate_id} 的预抓取状态",
        evidence_refs=state.evidence_refs,
    )


async def _grasp(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
) -> None:
    """在固定 grasp Stage 内完成双侧夹持或单侧外拉恢复序列。

    单侧外拉只负责把密集堆叠中的目标移到双侧可操作位置。外拉后首侧必须
    保持承载，Skill 实时重观测并把首侧位姿作为联合 IK 约束，只让第二侧
    接近和接管共同载荷；这样既不复用旧位姿，也不会在二次接近前让箱体脱手。
    """

    candidate = _selected_candidate(state)
    if candidate is None or state.target_revision is None:
        await _request_agent_decision(
            ctx, skill_input, state, reason="抓取阶段缺少有效候选或场景版本", details={}
        )
        return

    state.grasp_attempts += 1
    ctx.checkpoint(state)
    primary_side = state.extraction_primary_side or controller.primary_side(candidate)
    primary_ref = (
        controller.tool_ref_for_side(candidate, primary_side)
        if primary_side is not None
        else None
    )

    if primary_ref is not None and not state.primary_clamped:
        primary_result, primary_load = await _close_seat_and_observe(
            ctx,
            controller,
            skill_input,
            state,
            candidate,
            close_refs=[primary_ref],
            seat_refs=[primary_ref],
            verify_refs=[primary_ref],
            required_contact_tools=[],
            key_suffix="primary",
        )
        if not _contact_achieved(primary_result, primary_load, [primary_ref]):
            await _handle_partial_grasp_failure(
                ctx,
                controller,
                skill_input,
                state,
                primary_result,
                reason="外拉策略的第一侧工具没有形成确定接触",
            )
            return
        state.primary_clamped = True
        state.extraction_primary_side = primary_side
        ctx.checkpoint(state)

    if (
        primary_ref is not None
        and state.primary_clamped
        and not state.extraction_planned_from_contact
    ):
        if not await _refresh_extraction_plan_from_contact(
            ctx,
            controller,
            skill_input,
            state,
            primary_side=primary_side,
            primary_ref=primary_ref,
        ):
            return
        candidate = _selected_candidate(state)
        if candidate is None:
            return

    if primary_ref is not None and not state.extraction_completed:
        if not candidate.pull_path:
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="外拉候选缺少受约束的单侧 pull_path",
                details={"candidate_id": candidate.candidate_id},
            )
            return
        pull_result = await ctx.execute(
            f"grasp:extract:{state.plan_revision}:{candidate.candidate_id}",
            controller.move_targets(
                skill_input,
                candidate,
                candidate.pull_path,
                "extract",
                state.target_revision,
                # 单侧外拉时一只工具同时承担约束和驱动。速度过高会让末端在
                # 箱体克服托盘摩擦的瞬间过冲；这里采用保守的工艺速度，具体
                # 接触、力和相对运动仍由 Ability 根据实时传感器判断。
                0.05,
                required_contact_tools=[primary_ref],
            ),
        )
        _remember_result_evidence(state, pull_result)
        if pull_result.status != "succeeded":
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="单侧外拉没有安全完成，不能沿用旧目标位姿",
                details={
                    "status": pull_result.status,
                    "physical_effect": pull_result.physical_effect,
                    "error_code": pull_result.error_code,
                    "error_message": pull_result.error_message,
                    "failure_phase": "initial_extraction",
                },
            )
            return
        # 外拉已经产生真实物理位移。把它单独记为物理进度，后续若实时重观测
        # 暂时失败，只允许在首侧持续承载的前提下重新观测，绝不能重复外拉。
        state.extraction_completed = True
        state.secondary_replan_attempted = False
        ctx.checkpoint(state)

    if (
        primary_ref is not None
        and state.extraction_completed
        and not state.extraction_reobserved
    ):
        refreshed = await _reobserve_after_extraction(
            ctx, controller, skill_input, state
        )
        if not refreshed:
            return
        candidate = _selected_candidate(state)
        assert candidate is not None and state.target_revision is not None
        state.extraction_reobserved = True
        ctx.checkpoint(state)

    if (
        primary_ref is not None
        and state.primary_clamped
        and state.extraction_reobserved
    ):
        if not await _execute_secondary_engagement(
            ctx,
            controller,
            skill_input,
            state,
            candidate,
            primary_side=primary_side,
        ):
            return

    candidate = _selected_candidate(state)
    assert candidate is not None
    if primary_ref is None:
        close_refs = [
            controller.tool_ref_for_side(candidate, "left"),
            controller.tool_ref_for_side(candidate, "right"),
        ]
        seat_refs = close_refs
        required_contact_tools: list[str] = []
    else:
        secondary_side = "right" if primary_side == "left" else "left"
        close_refs = [controller.tool_ref_for_side(candidate, secondary_side)]
        # 第一侧已经承担真实物理约束，最终seat只能移动刚接近的第二侧。
        seat_refs = close_refs
        required_contact_tools = [primary_ref]

    # 直接双侧策略在一个 Ability Execution 中聚合两条 Runtime 工具命令。
    # Runtime 并没有原子双夹具命令。外拉策略保持首侧夹紧，只闭合第二侧；
    # 随后返回的实时 Observation 必须同时证明两侧承载同一物体，不能把
    # “第二侧命令成功”直接当作共同承载。
    final_result, final_load = await _close_seat_and_observe(
        ctx,
        controller,
        skill_input,
        state,
        candidate,
        close_refs=close_refs,
        seat_refs=seat_refs,
        verify_refs=list(skill_input.tool_refs),
        required_contact_tools=required_contact_tools,
        key_suffix="bilateral" if primary_ref is None else "secondary",
    )
    if not _contact_achieved(
        final_result, final_load, list(skill_input.tool_refs)
    ):
        recovered = await _reseat_single_missing_hook(
            ctx,
            controller,
            skill_input,
            state,
            candidate,
            final_load,
        )
        if recovered is not None:
            final_result, final_load = recovered
    if not _contact_achieved(final_result, final_load, list(skill_input.tool_refs)):
        await _handle_partial_grasp_failure(
            ctx,
            controller,
            skill_input,
            state,
            final_result,
            reason=f"工具 {', '.join(close_refs)} 没有形成确定接触",
        )
        return

    stable_bilateral = bool(
        final_result is not None
        and final_result.status == "succeeded"
        and final_load is not None
        and final_load.condition_satisfied
        and not final_load.slip_detected
        and not final_load.overload_detected
        and not final_load.sensor_fault
    )
    if not stable_bilateral:
        assert final_result is not None
        await _handle_partial_grasp_failure(
            ctx,
            controller,
            skill_input,
            state,
            final_result,
            reason="两侧工具接触完成，但 Robot SDK 未确认稳定双侧承载",
        )
        return

    state.object_held = True
    state.primary_clamped = False
    state.extraction_planned_from_contact = False
    state.extraction_primary_side = None
    state.extraction_completed = False
    state.active_strategy = "direct_bilateral"
    state.lift_completed = False
    state.stage = "lift_and_verify"
    ctx.checkpoint(state)
    ctx.report(
        "stage.completed",
        stage="grasp",
        stage_status="completed",
        summary=f"候选 {candidate.candidate_id} 已形成无滑移的双侧稳定承载",
        evidence_refs=state.evidence_refs,
    )


async def _close_seat_and_observe(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    candidate: GraspCandidate,
    *,
    close_refs: list[str],
    seat_refs: list[str],
    verify_refs: list[str],
    required_contact_tools: list[str],
    key_suffix: str,
) -> tuple[ActionResult, ToolLoadObservationValue | None]:
    """先让下钩就位，再闭合压紧，并以实时Observation确认真实承载。

    周转箱夹具不是普通平行夹爪。若先闭合再移动工具，下钩上提时会改变
    压紧块与箱体上沿的相对位置，真机的力控夹具或仿真的位置保持都可能失去
    一侧压紧接触。因此这里按工具几何先完成短距离 seat，再闭合；稳定承载仍由
    Ability基于实时传感器判断，而不是由动作顺序直接推定。
    """

    seat_result = await ctx.execute(
        f"grasp:seat:{state.plan_revision}:{candidate.candidate_id}:{key_suffix}",
        controller.seat_tools(
            skill_input,
            candidate=candidate,
            tool_refs=seat_refs,
            target_revision=state.target_revision or candidate.observation_revision,
            required_contact_tools=required_contact_tools,
        ),
    )
    _remember_result_evidence(state, seat_result)
    if seat_result.status != "succeeded":
        ctx.checkpoint(state)
        return seat_result, None

    close_result = await ctx.execute(
        f"grasp:close:{state.plan_revision}:{candidate.candidate_id}:{key_suffix}",
        controller.close_tools(skill_input, candidate=candidate, tool_refs=close_refs),
    )
    _remember_result_evidence(state, close_result)
    if close_result.status != "succeeded":
        ctx.checkpoint(state)
        return close_result, None

    load_result = await ctx.execute(
        f"grasp:load:{state.plan_revision}:{candidate.candidate_id}:{key_suffix}",
        controller.verify_tool_load(skill_input, tool_refs=verify_refs),
    )
    _remember_result_evidence(state, load_result)
    load = _parse_value(
        _find_observation(load_result, "robot.tool_load"),
        ToolLoadObservationValue,
    )
    ctx.checkpoint(state)
    return load_result, load


async def _reseat_single_missing_hook(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    candidate: GraspCandidate,
    load: ToolLoadObservationValue | None,
) -> tuple[ActionResult, ToolLoadObservationValue | None] | None:
    """双侧闭合后若只有一个下钩短暂脱离，就地补座该侧一次。

    CloseUntilContact 的成功结果可能已经证明两侧都钩入，但工具停止后的轻微
    回弹会让其中一个下钩在随后负载观察时离开箱沿。此时另一侧仍明确钩入且
    两侧压紧片都接触，退出双臂再重走整条抓取路径反而会把手臂送到工作空间
    边缘。这里保持已承载侧不动，只把缺失侧重新执行当前候选的短 seat 和
    close，然后重新观察双侧负载；没有明确单侧承载证据时仍走原有 hold 流程。
    """

    if (
        load is None
        or load.slip_detected
        or load.overload_detected
        or load.sensor_fault
    ):
        return None
    by_ref = {item.tool_ref: item for item in load.tools}
    requested = list(skill_input.tool_refs)
    if set(by_ref) != set(requested):
        return None
    missing = [
        tool_ref
        for tool_ref in requested
        if by_ref[tool_ref].available
        and by_ref[tool_ref].clamp_contact
        and not by_ref[tool_ref].hook_contact
        and not by_ref[tool_ref].sensor_fault
    ]
    stable = [
        tool_ref
        for tool_ref in requested
        if by_ref[tool_ref].available
        and by_ref[tool_ref].clamp_contact
        and by_ref[tool_ref].hook_contact
        and not by_ref[tool_ref].sensor_fault
    ]
    if len(missing) != 1 or len(stable) != 1:
        return None

    ctx.report(
        "stage.recovering",
        summary=f"{missing[0]} 下钩接触回弹，保持另一侧承载并就地补座",
        evidence_refs=state.evidence_refs,
    )
    return await _close_seat_and_observe(
        ctx,
        controller,
        skill_input,
        state,
        candidate,
        close_refs=missing,
        seat_refs=missing,
        verify_refs=requested,
        required_contact_tools=stable,
        key_suffix="single-hook-reseat",
    )


def _contact_achieved(
    result: ActionResult,
    load: ToolLoadObservationValue | None,
    tool_refs: list[str],
) -> bool:
    """只接受 seat 后由 RobotState Ability 连续确认的真实承载。"""

    if result.status != "succeeded" or load is None or not load.condition_satisfied:
        return False
    if load.slip_detected or load.overload_detected or load.sensor_fault:
        return False
    by_ref = {item.tool_ref: item for item in load.tools}
    for tool_ref in tool_refs:
        value = by_ref.get(tool_ref)
        if value is None or not value.available:
            return False
        if not value.hook_contact or not value.clamp_contact or value.sensor_fault:
            return False
    return True


def _missing_preflight_contact_tools(
    result: ActionResult,
    tool_refs: tuple[str, ...],
) -> list[str]:
    """从抬升预检结果找出需要补座或重新压紧的工具，不放宽承载阈值。"""

    message = result.error_message or ""
    missing_markers = (
        "hook_contact_missing",
        "clamp_contact_missing",
        "hook_force_low",
        "clamp_force_low",
    )
    # Ability 用 ", " 拼接多条 blocking reason（manipulator_motion.py 的
    # LiftHeldObject 预检），只按 "; " 拆分会把两条原因当成一个 token，
    # 白名单校验必然失败，单侧重入位恢复会因此永不触发。
    reasons = [
        item.strip()
        for item in re.split(r"[,;]\s*", message.rsplit("禁止开始抬升: ", 1)[-1])
        if item.strip()
    ]
    recoverable = {f"{ref}:{marker}" for ref in tool_refs for marker in missing_markers}
    if not reasons or any(reason not in recoverable for reason in reasons):
        return []
    return [
        tool_ref
        for tool_ref in tool_refs
        if any(f"{tool_ref}:{marker}" in message for marker in missing_markers)
    ]


def _primary_can_reclamp(load: ToolLoadObservationValue | None, tool_ref: str) -> bool:
    """只在钩入仍明确、压紧状态下降时允许一次重新夹紧。"""

    if (
        load is None
        or load.slip_detected
        or load.overload_detected
        or load.sensor_fault
    ):
        return False
    value = next((item for item in load.tools if item.tool_ref == tool_ref), None)
    return bool(
        value is not None
        and value.available
        and value.hook_contact
        and not value.sensor_fault
        and (not value.clamp_contact or not load.condition_satisfied)
    )


async def _ensure_primary_load(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    candidate: GraspCandidate,
    primary_ref: str,
    *,
    key_suffix: str,
) -> bool:
    """外拉后确认首侧接触；必要时至多重新夹紧一次。"""

    async def verify(suffix: str):
        result = await ctx.execute(
            f"grasp:primary-load:{state.plan_revision}:{key_suffix}:{suffix}",
            controller.verify_tool_load(skill_input, tool_refs=[primary_ref]),
        )
        _remember_result_evidence(state, result)
        load = _parse_value(
            _find_observation(result, "robot.tool_load"),
            ToolLoadObservationValue,
        )
        return result, load

    load_result, load = await verify("verify")
    if _contact_achieved(load_result, load, [primary_ref]):
        return True
    if not state.primary_reclamp_attempted and _primary_can_reclamp(load, primary_ref):
        # close不是每段运动后的固定步骤。只有实时Observation证明钩爪仍在、
        # 传感与受力安全，但压紧接触下降时，才执行一次幂等恢复。
        state.primary_reclamp_attempted = True
        ctx.checkpoint(state)
        reclamp = await ctx.execute(
            f"grasp:primary-load:{state.plan_revision}:{key_suffix}:reclamp",
            controller.close_tools(
                skill_input, candidate=candidate, tool_refs=[primary_ref]
            ),
        )
        _remember_result_evidence(state, reclamp)
        if reclamp.status == "succeeded":
            load_result, load = await verify("reverify")
            if _contact_achieved(load_result, load, [primary_ref]):
                return True

    hold_result = await ctx.execute(
        f"grasp:primary-load:{state.plan_revision}:{key_suffix}:hold",
        controller.hold_object(skill_input, reason="单侧承载状态未满足后续动作要求"),
    )
    _remember_result_evidence(state, hold_result)
    await _request_agent_decision(
        ctx,
        skill_input,
        state,
        reason="单侧承载状态未满足后续动作要求",
        details={
            "tool_ref": primary_ref,
            "load_status": load_result.status,
            "hold_status": hold_result.status,
            "failure_phase": "primary_contact_lost",
            "reclamp_attempted": state.primary_reclamp_attempted,
        },
    )
    return False


async def _handle_partial_grasp_failure(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    result: ActionResult,
    *,
    reason: str,
) -> None:
    """闭合命令失败后保持现场并交给 Robot Agent，不再自动切换候选。

    `gripper.close` 已经驱动至少一个工具运动；即使下游错误地把
    ``physical_effect`` 写成 ``none``，现场也可能已经形成单侧或双侧接触。
    此时执行下一候选的接近轨迹会拖动箱体，正是人工调试中观察到的二次
    推箱问题。因此先复用通用 hold Action 固定当前 Robot，再保留证据请求
    Agent 决策。这里不猜测安全撤离方向，也不增加新的恢复状态机。
    """

    hold_result = await ctx.execute(
        f"grasp:hold-after-close-failure:{state.plan_revision}:{state.grasp_attempts}",
        controller.hold_object(skill_input, reason=reason),
    )
    _remember_result_evidence(state, hold_result)
    ctx.checkpoint(state)
    await _request_agent_decision(
        ctx,
        skill_input,
        state,
        reason=reason,
        details={
            "status": result.status,
            "physical_effect": result.physical_effect,
            "hold_status": hold_result.status,
            "hold_physical_effect": hold_result.physical_effect,
            "failure_phase": "partial_grasp",
        },
    )


async def _recover_unheld_partial_grasp(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
) -> bool:
    """在重新规划前，把没有稳定持物的双工具退出当前箱沿。"""

    candidate = _selected_candidate(state)
    if candidate is None:
        ctx.fail(
            "GRASP_RECOVERY_GEOMETRY_MISSING",
            "部分抓取失败后缺少原候选几何，无法执行确定性退出",
            state.evidence_refs,
        )
        return False

    revision = state.target_revision or candidate.observation_revision
    actions = controller.recover_after_partial_close(
        skill_input,
        candidate=candidate,
        target_revision=revision,
    )
    for index, action in enumerate(actions):
        result = await ctx.execute(
            f"grasp:partial-recovery:{state.decision_revision}:{index}", action
        )
        _remember_result_evidence(state, result)
        if result.status != "succeeded":
            # 这里不能再进入相同的 Agent 重试循环。Ability 已在规划或执行
            # 失败处停止命令；保留准确错误，由上层决定是否人工恢复现场。
            ctx.fail(
                "GRASP_RECOVERY_RETREAT_FAILED",
                result.error_message or "部分抓取失败后的夹具退出未完成",
                state.evidence_refs,
            )
            return False

    ctx.report(
        "stage.recovering",
        summary="未形成稳定承载的双工具已退出箱沿，将按 Robot Agent 决定重新规划",
        evidence_refs=state.evidence_refs,
    )
    return True


async def _refresh_extraction_plan_from_contact(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    *,
    primary_side: str,
    primary_ref: str,
) -> bool:
    """首侧形成真实接触后，再从当前末端与物体位姿规划第一次外拉。"""

    result = await ctx.execute(
        f"grasp:contact-observe:{state.plan_revision}",
        controller.locate_target(skill_input),
    )
    _remember_result_evidence(state, result)
    observation = _find_observation(result, "target_pose")
    target = _parse_value(observation, TargetPoseValue)
    if result.status != "succeeded" or observation is None or target is None:
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="首侧接合后无法取得目标实时位姿",
            details={"status": result.status},
        )
        return False

    revision = observation.revision or target.pose.revision
    candidate_result = await ctx.execute(
        f"grasp:contact-candidates:{state.plan_revision}:{revision}",
        controller.generate_candidates(
            skill_input,
            target_pose=target.pose,
            object_extent_m=target.extent_m,
            target_revision=revision,
            active_strategy="direct_bilateral",
            engaged_tool_ref=primary_ref,
        ),
    )
    _remember_result_evidence(state, candidate_result)
    candidate_observation = _find_observation(candidate_result, "grasp_candidates")
    value = _parse_value(candidate_observation, GraspCandidatesValue)
    candidates = _rank_candidates(value.candidates if value else [], "direct_bilateral")

    state.target_observation_ref = observation.id
    state.target_revision = revision
    state.target_pose = target.pose
    state.target_extent_m = target.extent_m
    state.candidates = candidates
    state.selected_candidate_id = candidates[0].candidate_id if candidates else None
    state.plan_revision += 1
    _remember_evidence(state, observation.evidence_refs)
    if candidate_observation is not None:
        _remember_evidence(state, candidate_observation.evidence_refs)

    if candidate_result.status != "succeeded" or not candidates:
        ctx.checkpoint(state)
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="首侧接合后的实时状态没有生成可用抓取候选",
            details={"status": candidate_result.status},
        )
        return False

    direct = next(
        (item for item in candidates if item.strategy == "direct_bilateral"),
        None,
    )
    selected = direct or candidates[0]
    state.selected_candidate_id = selected.candidate_id
    state.extraction_planned_from_contact = True
    if direct is not None:
        # 实时路径预检已经证明第二侧能够进入时，不为了满足保守间隙公式
        # 继续外拉。第一侧保持当前真实接触，直接进入第二侧接合。
        state.extraction_completed = True
        state.extraction_reobserved = True
        state.active_strategy = "direct_bilateral"
    else:
        expected = f"{primary_side}_extract_first"
        if selected.strategy != expected or not selected.pull_path:
            ctx.checkpoint(state)
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="首侧接合后缺少可执行的实时外拉路径",
                details={"selected_strategy": selected.strategy},
            )
            return False
    ctx.checkpoint(state)
    return True


async def _reobserve_after_extraction(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
) -> bool:
    """外拉后以实时位姿重建双侧候选，旧候选不得继续使用。"""

    previous_candidate = _selected_candidate(state)
    engaged_tool_ref = (
        controller.tool_ref_for_side(previous_candidate, state.extraction_primary_side)
        if previous_candidate is not None and state.extraction_primary_side is not None
        else None
    )
    if engaged_tool_ref is not None and not await _ensure_primary_load(
        ctx,
        controller,
        skill_input,
        state,
        previous_candidate,
        engaged_tool_ref,
        key_suffix=f"extract-{state.extraction_adjustments}",
    ):
        return False

    result = await ctx.execute(
        f"grasp:reobserve:{state.plan_revision}", controller.locate_target(skill_input)
    )
    _remember_result_evidence(state, result)
    observation = _find_observation(result, "target_pose")
    target = _parse_value(observation, TargetPoseValue)
    if result.status != "succeeded" or observation is None or target is None:
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="单侧外拉后无法取得新的目标位姿",
            details={"status": result.status},
        )
        return False
    # 下钩与上夹片接合后，第一次水平外拉同时承担“受载证明”：目标箱体
    # 必须沿计划外拉方向产生可分辨的真实位移。仅有瞬时hook_contact或箱沿
    # 擦碰不足以继续第二侧接合；失败时保持现场并交给Agent，而不是继续拉。
    if previous_candidate is not None and previous_candidate.pull_path:
        followed, follow_details = _object_followed_pull(
            previous_candidate,
            state.target_pose,
            target.pose,
        )
        if not followed:
            hold = await ctx.execute(
                f"grasp:proof-load:{state.plan_revision}:hold",
                controller.hold_object(
                    skill_input, reason="首侧外拉时箱体没有跟随钩脚运动"
                ),
            )
            _remember_result_evidence(state, hold)
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="首侧工具有接触，但箱体没有通过凹槽钩入受载证明",
                details={
                    **follow_details,
                    "hold_status": hold.status,
                    "failure_phase": "primary_load_proof",
                },
            )
            return False

    revision = observation.revision or target.pose.revision
    candidate_result = await ctx.execute(
        f"grasp:recandidates:{state.plan_revision}:{revision}:direct_bilateral",
        controller.generate_candidates(
            skill_input,
            target_pose=target.pose,
            object_extent_m=target.extent_m,
            target_revision=revision,
            active_strategy="direct_bilateral",
            engaged_tool_ref=engaged_tool_ref,
        ),
    )
    _remember_result_evidence(state, candidate_result)
    candidate_observation = _find_observation(candidate_result, "grasp_candidates")
    value = _parse_value(candidate_observation, GraspCandidatesValue)
    # 外拉改变了箱体位姿，下面保存的必须是本次实时观测生成的候选，不能继续
    # 使用外拉前的旧Pose。优先使用完整双侧候选；若常规入口预检只保留了某个
    # extract-first候选，也仍可复用其中按新Pose计算的双侧工具几何。此时第一侧
    # 已经由当前Execution持续承载，Skill只执行第二侧接合，绝不能把候选标签
    # 误解为“再做一次外拉”或交给Agent切换策略。
    candidates = _rank_candidates(value.candidates if value else [], "direct_bilateral")
    direct_candidates = [
        item for item in candidates if item.strategy == "direct_bilateral"
    ]
    _remember_evidence(state, observation.evidence_refs)
    if candidate_observation is not None:
        _remember_evidence(state, candidate_observation.evidence_refs)

    if candidate_result.status != "succeeded" or not candidates:
        # 候选生成尚未成功时保留外拉前候选和Pose。它们不再用于第二手动作，
        # 只用于下一次重观测再次证明“箱体确实跟随已经完成的那次外拉”。若
        # 先把候选清空并覆盖Pose，restart_observation会留在grasp原地循环，
        # 或把没有发生的新位移误判为脱钩。新的候选成功后才原子替换旧计划。
        state.plan_revision += 1
        ctx.checkpoint(state)
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="单侧外拉后的新位姿没有生成可用候选",
            details={
                "status": candidate_result.status,
                "physical_effect": candidate_result.physical_effect,
                "error_code": candidate_result.error_code,
                "error_message": candidate_result.error_message,
                "failure_phase": "secondary_approach",
                "available_candidates": [],
                "grasp_attempts": state.grasp_attempts,
            },
        )
        return False

    state.target_observation_ref = observation.id
    state.target_revision = revision
    state.target_pose = target.pose
    state.target_extent_m = target.extent_m
    state.candidates = candidates
    state.selected_candidate_id = candidates[0].candidate_id
    state.plan_revision += 1
    ctx.checkpoint(state)
    selected = direct_candidates[0] if direct_candidates else candidates[0]
    state.selected_candidate_id = selected.candidate_id
    expected_adjustment_strategy = (
        f"{state.extraction_primary_side}_extract_first"
        if state.extraction_primary_side is not None
        else None
    )
    if selected.strategy == expected_adjustment_strategy and selected.pull_path:
        assert engaged_tool_ref is not None
        adjustment = await ctx.execute(
            f"grasp:extract-adjust:{state.plan_revision}",
            controller.move_targets(
                skill_input,
                selected,
                selected.pull_path,
                "extract",
                revision,
                # 补充外拉仍是接触工艺动作，但不应因为拆成短步而再次降到
                # 几乎不可观察的速度。每步后继续用真实物体位姿判断是否需要
                # 下一步，速度提升不能替代接触、受力与hold边界。
                0.05,
                required_contact_tools=[engaged_tool_ref],
            ),
        )
        _remember_result_evidence(state, adjustment)
        if adjustment.status != "succeeded":
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="补充外拉期间首侧接触或动作状态异常",
                details={
                    "status": adjustment.status,
                    "physical_effect": adjustment.physical_effect,
                    "error_code": adjustment.error_code,
                    "error_message": adjustment.error_message,
                    "failure_phase": "extraction_adjustment",
                },
            )
            return False
        state.extraction_adjustments += 1
        state.secondary_replan_attempted = False
        ctx.checkpoint(state)
        # 每次补拉后都重新读取物体Pose和邻箱间隙；不能累计旧Pose或把命令
        # 位移当成箱体实际位移。何时停止补拉由下一次实时候选的第二侧路径
        # 是否可达决定；机械可达范围、接触保持和Provider声明的最大外拉边界
        # 会自然终止动作，不能用与箱体尺寸无关的固定重试次数提前放弃。
        return await _reobserve_after_extraction(ctx, controller, skill_input, state)
    state.active_strategy = "direct_bilateral"
    ctx.checkpoint(state)
    return True


def _object_followed_pull(candidate, before_pose, after_pose) -> tuple[bool, dict]:
    """比较真实物体位移与本次外拉方向，拒绝“只碰到槽口边缘”的假接合。"""

    if (
        before_pose is None
        or not candidate.pull_path
        or candidate.pull_direction_world is None
        or candidate.pull_distance_m <= 0.0
    ):
        return False, {"reason": "missing_pull_reference"}
    axis = list(candidate.pull_direction_world)
    axis_norm = sum(value * value for value in axis) ** 0.5
    if axis_norm <= 1e-6:
        return False, {"reason": "planned_pull_direction_invalid"}
    axis = [value / axis_norm for value in axis]
    actual = [
        after_pose.position_m[index] - before_pose.position_m[index]
        for index in range(3)
    ]
    followed_distance = sum(actual[index] * axis[index] for index in range(3))
    # 3mm只用于区分MuJoCo ground-truth中的真实随动与接触弹性/数值噪声，
    # 不描述箱型或最终外拉距离；后续仍根据新SceneSnapshot计算剩余净空。
    minimum_follow_distance = min(0.003, candidate.pull_distance_m * 0.25)
    details = {
        "planned_pull_distance_m": candidate.pull_distance_m,
        "object_follow_distance_m": followed_distance,
        "minimum_follow_distance_m": minimum_follow_distance,
    }
    return followed_distance >= minimum_follow_distance, details


async def _refresh_secondary_insertion_targets(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    *,
    primary_side: str,
) -> GraspCandidate | None:
    """第二侧到达pregrasp后，以实时箱体位姿重算最后的insert和seat。"""

    current = _selected_candidate(state)
    if current is None:
        return None
    primary_ref = controller.tool_ref_for_side(current, primary_side)
    result = await ctx.execute(
        f"grasp:reengage-observe:{state.plan_revision}",
        controller.locate_target(skill_input),
    )
    _remember_result_evidence(state, result)
    observation = _find_observation(result, "target_pose")
    target = _parse_value(observation, TargetPoseValue)
    if result.status != "succeeded" or observation is None or target is None:
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="第二侧插入前无法取得箱体最新位姿",
            details={"status": result.status},
        )
        return None

    revision = observation.revision or target.pose.revision
    planned = await ctx.execute(
        f"grasp:reengage-candidates:{state.plan_revision}:{revision}",
        controller.generate_candidates(
            skill_input,
            target_pose=target.pose,
            object_extent_m=target.extent_m,
            target_revision=revision,
            active_strategy="direct_bilateral",
            engaged_tool_ref=primary_ref,
            secondary_resume_phase="insert",
        ),
    )
    _remember_result_evidence(state, planned)
    candidate_observation = _find_observation(planned, "grasp_candidates")
    value = _parse_value(candidate_observation, GraspCandidatesValue)
    candidates = _rank_candidates(value.candidates if value else [], "direct_bilateral")
    direct = next(
        (item for item in candidates if item.strategy == "direct_bilateral"),
        None,
    )
    if planned.status != "succeeded" or direct is None:
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="第二侧已到达预抓取位置，但最新箱体位姿没有可执行插入候选",
            details={"status": planned.status},
        )
        return None

    # 这里只更新尚未执行的insert/seat几何。已经完成的clearance、transfer和
    # pregrasp不能因箱体轻微位移而重放；首侧仍由同一Execution持续承载。
    state.target_observation_ref = observation.id
    state.target_revision = revision
    state.target_pose = target.pose
    state.target_extent_m = target.extent_m
    state.candidates = candidates
    state.selected_candidate_id = direct.candidate_id
    state.plan_revision += 1
    _remember_evidence(state, observation.evidence_refs)
    if candidate_observation is not None:
        _remember_evidence(state, candidate_observation.evidence_refs)
    ctx.checkpoint(state)
    return direct


async def _execute_secondary_engagement(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    candidate: GraspCandidate,
    *,
    primary_side: str,
) -> bool:
    """保持首侧实时物理约束，只移动第二侧完成接近和插入。"""

    secondary_side = "right" if primary_side == "left" else "left"
    secondary_ref = controller.tool_ref_for_side(candidate, secondary_side)
    open_result = await ctx.execute(
        f"grasp:reengage:{state.plan_revision}:open",
        controller.open_tools(
            skill_input, candidate=candidate, tool_refs=[secondary_ref]
        ),
    )
    _remember_result_evidence(state, open_result)
    if open_result.status != "succeeded":
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="第二侧夹具无法打开到接近位置",
            details={"status": open_result.status},
        )
        return False

    actions = controller.plan_secondary_engagement(
        skill_input,
        candidate=candidate,
        primary_side=primary_side,
        target_revision=state.target_revision or "",
    )
    # 第二侧此前保持自然待机位。clearance先在身体附近短前展并抬高，
    # transfer沿箱体外侧送到箱顶上方，pregrasp再斜向下降到槽口外。
    # 三段来自外拉后的同一次实时观测；到达真正需要精确对准的槽口外再
    # 刷新一次，不能在transfer中途重建整条走廊并重复已经完成的路径。
    for index, action in enumerate(actions[:3]):
        result = await ctx.execute(
            f"grasp:reengage:{state.plan_revision}:{index}", action
        )
        _remember_result_evidence(state, result)
        if result.status != "succeeded":
            # required_contact_tools已让Ability在整段动作中持续监控首侧。
            # 接触丢失时由Ability停止并hold，Skill不得重新播放本段物理动作。
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="保持首侧承载时第二侧接近失败",
                details={
                    "status": result.status,
                    "action": action.type,
                    "physical_effect": result.physical_effect,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                    "failure_phase": "secondary_approach",
                },
            )
            return False

    # 到达凹槽入口后只重算最后一段水平插入。该刷新不重复clearance、
    # transfer或pregrasp，也不重新夹紧首侧；它只用实时箱体Pose消除承载侧
    # 微动造成的槽口偏差。
    previous_candidate = candidate
    refreshed = await _refresh_secondary_insertion_targets(
        ctx,
        controller,
        skill_input,
        state,
        primary_side=primary_side,
    )
    if refreshed is None:
        return False
    candidate_options = [
        item for item in state.candidates if item.strategy == "direct_bilateral"
    ]
    if not candidate_options:
        candidate_options = [refreshed]
    secondary_ref = controller.tool_ref_for_side(refreshed, secondary_side)
    current_pregrasp = next(
        item
        for item in previous_candidate.pregrasp_poses
        if item.tool_ref == secondary_ref
    )
    candidate = refreshed
    actions = controller.plan_secondary_engagement(
        skill_input,
        candidate=candidate,
        primary_side=primary_side,
        target_revision=state.target_revision or "",
    )
    for option_index, candidate in enumerate(candidate_options):
        actions = controller.plan_secondary_engagement(
            skill_input,
            candidate=candidate,
            primary_side=primary_side,
            target_revision=state.target_revision or "",
        )
        state.selected_candidate_id = candidate.candidate_id
        ctx.checkpoint(state)
        candidate_pregrasp = next(
            item for item in candidate.pregrasp_poses if item.tool_ref == secondary_ref
        )
        pregrasp_shift_m = (
            sum(
                (current - previous) ** 2
                for current, previous in zip(
                    candidate_pregrasp.target_pose.position_m,
                    current_pregrasp.target_pose.position_m,
                    strict=True,
                )
            )
            ** 0.5
        )
        if pregrasp_shift_m > SECONDARY_REALIGN_SHIFT_M:
            # 连续凹槽允许第二手在槽长方向选择邻近接入点。切换候选时只做
            # 当前槽口外的短对齐；clearance、transfer、外拉和首侧夹紧都不
            # 重放。真实轨迹仍持续监控首侧接触，任何物理异常立即hold。
            realign_result = await ctx.execute(
                (
                    f"grasp:reengage:{state.plan_revision}:3:realign:"
                    f"{candidate.candidate_id}"
                ),
                actions[2],
            )
            _remember_result_evidence(state, realign_result)
            if realign_result.status != "succeeded":
                can_try_next = (
                    option_index + 1 < len(candidate_options)
                    and realign_result.physical_effect == "none"
                    and realign_result.error_code
                    in {"PLANNING_FAILED", "CANDIDATE_UNREACHABLE"}
                )
                if can_try_next:
                    ctx.report(
                        "stage.recovering",
                        summary="当前槽内接入点不可达，尝试同一凹槽内的邻近位置",
                        evidence_refs=state.evidence_refs,
                    )
                    continue
                state.secondary_replan_attempted = True
                ctx.checkpoint(state)
                await _request_agent_decision(
                    ctx,
                    skill_input,
                    state,
                    reason="按最新箱体位姿重新对齐第二侧槽口失败",
                    details={
                        "status": realign_result.status,
                        "action": actions[2].type,
                        "physical_effect": realign_result.physical_effect,
                        "error_code": realign_result.error_code,
                        "error_message": realign_result.error_message,
                        "failure_phase": "secondary_approach",
                        "pregrasp_shift_m": pregrasp_shift_m,
                    },
                )
                return False
            current_pregrasp = candidate_pregrasp

        insert_result = await ctx.execute(
            (f"grasp:reengage:{state.plan_revision}:4:insert:{candidate.candidate_id}"),
            actions[3],
        )
        _remember_result_evidence(state, insert_result)
        if insert_result.status == "succeeded":
            break
        can_try_next = (
            option_index + 1 < len(candidate_options)
            and insert_result.physical_effect == "none"
            and insert_result.error_code in {"PLANNING_FAILED", "CANDIDATE_UNREACHABLE"}
        )
        if can_try_next:
            ctx.report(
                "stage.recovering",
                summary="当前槽内插入点不可达，尝试同一凹槽内的邻近位置",
                evidence_refs=state.evidence_refs,
            )
            continue
        # 局部几何候选已经用完，不能再让Agent通过restart_observation重放
        # 已完成的第二手长接近路径。此时只允许安全终止并保留首侧hold证据。
        state.secondary_replan_attempted = True
        ctx.checkpoint(state)
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="按槽口实时位姿执行第二侧插入失败",
            details={
                "status": insert_result.status,
                "action": actions[3].type,
                "physical_effect": insert_result.physical_effect,
                "error_code": insert_result.error_code,
                "error_message": insert_result.error_message,
                "failure_phase": "secondary_approach",
            },
        )
        return False

    verify_action = actions[4]
    verify_result = await ctx.execute(
        f"grasp:reengage:{state.plan_revision}:5:verify", verify_action
    )
    _remember_result_evidence(state, verify_result)
    final_observation = _find_observation(verify_result, "pregrasp_state")
    if verify_result.status != "succeeded":
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="保持首侧承载时第二侧接近失败",
            details={"status": verify_result.status, "action": verify_action.type},
        )
        return False
    pregrasp = _parse_value(final_observation, PregraspStateValue)
    if pregrasp is None or not pregrasp.reached:
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="重新观测后的第二侧工具未满足插入误差限制",
            details={"candidate_id": candidate.candidate_id},
        )
        return False
    return True


async def _lift_and_verify(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
) -> None:
    """持续检查双侧稳定承载，抬升后用场景真值独立验证物体位移。"""

    candidate = _selected_candidate(state)
    if (
        candidate is None
        or state.target_revision is None
        or state.target_pose is None
        or state.target_extent_m is None
        or not state.object_held
    ):
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="抬升阶段缺少可确认的持物状态",
            details={"object_held": state.object_held},
        )
        return

    maximum_lift_segment_m = min(0.04, skill_input.minimum_lift_height_m * 0.5)
    if not state.lift_completed and not state.verification_only:
        lift_distance_m = min(
            state.pending_lift_distance_m or maximum_lift_segment_m,
            maximum_lift_segment_m,
        )
        # 密集堆垛中双侧刚完成接合时，机械臂通常仍接近当前可达域边缘。
        # 首次直接要求完整抬升高度会把一个可执行动作变成远端单点 IK；
        # 先完成一半抬升，再依据实时物体高度补足距离，不修改通用 IK。
        # 命令位移本身仍不能作为物体已经达到业务高度的证明。
        handle = await ctx.start_action(
            (
                f"grasp:lift:{state.plan_revision}:{candidate.candidate_id}:"
                f"{state.verification_attempts}"
            ),
            controller.lift_object(
                skill_input, candidate=candidate, distance_m=lift_distance_m
            ),
        )
        latest_progress: LiftProgressValue | None = None
        result: ActionResult | None = None
        forced_stop = False
        async for feedback in handle.feedback():
            _report_feedback(ctx, state, "action.progress", feedback)
            progress = _parse_value(
                _find_observation_in_feedback(feedback, "lift_progress"),
                LiftProgressValue,
            )
            if progress is not None:
                latest_progress = progress
            if feedback.severity == "critical" or (
                progress is not None
                and (
                    progress.slip_detected
                    or progress.overloaded
                    or not progress.object_follows_tools
                    or not progress.stable_load
                )
            ):
                result = await handle.stop("抬升过程中稳定承载失效")
                forced_stop = True
                break
        if result is None:
            result = await handle.result()
        _remember_result_evidence(state, result)
        progress = (
            _parse_value(_find_observation(result, "lift_progress"), LiftProgressValue)
            or latest_progress
        )
        if (
            forced_stop
            or result.status != "succeeded"
            or progress is None
            or not progress.object_follows_tools
            or not progress.stable_load
            or progress.slip_detected
            or progress.overloaded
        ):
            # 规划阶段失败可能发生在Runtime命令下发之前，此时没有新的
            # lift_progress。前一Stage已经用VerifyToolLoad确认了双侧承载，
            # 不能因为“本次没有进度Observation”就把这个物理事实清空；否则
            # 随后的安全停止会错误报告未持物并允许释放Robot锁。只有动作反馈
            # 明确证明箱体未跟随或载荷失稳时，才降低当前持物判断。
            if progress is not None:
                state.object_held = bool(
                    progress.object_follows_tools
                    and progress.stable_load
                    and not progress.slip_detected
                    and not progress.overloaded
                )
            if (
                not forced_stop
                and result.status == "failed"
                and result.error_code == "LOAD_NOT_STABLE"
                and result.physical_effect in {None, "", "none"}
                and state.grasp_attempts < 2
            ):
                missing_refs = _missing_preflight_contact_tools(
                    result, skill_input.tool_refs
                )
                remaining_refs = [
                    tool_ref
                    for tool_ref in skill_input.tool_refs
                    if tool_ref not in missing_refs
                ]
                if len(missing_refs) == 1 and len(remaining_refs) == 1:
                    # LiftHeldObject尚未下发运动，错误又明确指出只有一侧接触
                    # 丢失时，另一侧仍通过同一次预检。此时若让双手一起重新
                    # seat，会把仍贴合箱沿的一侧重复压入并触发接触力限制；
                    # 这里只移动失去接触的一侧，Ability同时固定并监控另一侧。
                    # 新revision用于生成新的幂等Action key，不会重放接近路径。
                    state.plan_revision += 1
                    state.grasp_attempts += 1
                    state.object_held = False
                    state.lift_completed = False
                    state.pending_lift_distance_m = None
                    state.verification_attempts = 0
                    state.primary_clamped = True
                    state.extraction_primary_side = next(
                        (
                            target.side
                            for target in candidate.tool_targets
                            if target.tool_ref == remaining_refs[0]
                        ),
                        None,
                    )
                    ctx.checkpoint(state)
                    recovery_result, recovery_load = await _close_seat_and_observe(
                        ctx,
                        controller,
                        skill_input,
                        state,
                        candidate,
                        close_refs=missing_refs,
                        seat_refs=missing_refs,
                        verify_refs=list(skill_input.tool_refs),
                        required_contact_tools=remaining_refs,
                        key_suffix="lift-preflight-reseat",
                    )
                    if _contact_achieved(
                        recovery_result,
                        recovery_load,
                        list(skill_input.tool_refs),
                    ):
                        state.object_held = True
                        state.primary_clamped = False
                        state.extraction_primary_side = None
                        ctx.checkpoint(state)
                        ctx.report(
                            "stage.recovering",
                            stage="lift_and_verify",
                            stage_status="running",
                            summary=(
                                f"抬升前恢复了 {missing_refs[0]} 的接触，"
                                "继续验证双侧承载"
                            ),
                            evidence_refs=state.evidence_refs,
                        )
                        return

                    # 局部恢复没有成功时保留仍承载的一侧，让既有恢复逻辑
                    # 先hold再请求Agent；不得从当前贴箱姿态切换完整抓取策略。
                    await _handle_partial_grasp_failure(
                        ctx,
                        controller,
                        skill_input,
                        state,
                        recovery_result,
                        reason=f"{missing_refs[0]} 的局部重新就位没有恢复双侧承载",
                    )
                    return
            ctx.checkpoint(state)
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="抬升没有形成可自动继续的确定双侧承载状态",
                details={
                    "status": result.status,
                    "physical_effect": result.physical_effect,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                },
            )
            return
        state.lift_completed = True
        state.pending_lift_distance_m = None
        ctx.checkpoint(state)

    state.verification_only = False
    verification_result = await ctx.execute(
        f"grasp:verify:{state.plan_revision}:{candidate.candidate_id}:{state.verification_attempts}",
        controller.verify_grasp(
            skill_input, candidate=candidate, initial_object_pose=state.target_pose
        ),
    )
    state.verification_attempts += 1
    _remember_result_evidence(state, verification_result)
    verification_observation = _find_observation(
        verification_result, "grasp_verification"
    )
    verification = _parse_value(verification_observation, GraspVerificationValue)
    ctx.checkpoint(state)

    if _verification_achieved(
        skill_input, candidate, verification_result, verification
    ):
        assert verification is not None and verification_observation is not None
        state.lift_completed = True
        held_object = HeldObjectState(
            object_ref=skill_input.object_ref,
            robot_ref=ctx.robot_ref,
            tool_refs=skill_input.tool_refs,
            tool_poses=verification.tool_poses,
            grasp_pose=skill_input_pose(candidate),
            object_pose=verification.object_pose,
            object_size_m=state.target_extent_m,
            grasp_candidate_id=candidate.candidate_id,
            grasp_confidence=1.0,
            robot_state_revision=verification_observation.revision
            or state.target_revision,
            evidence_refs=list(
                dict.fromkeys([*state.evidence_refs, verification_observation.id])
            ),
        )
        state.verified_held_object = held_object
        state.stage = "prepare_transport"
        ctx.checkpoint(state)
        ctx.report(
            "stage.completed",
            stage="lift_and_verify",
            stage_status="completed",
            summary="抓取和抬升已独立验证，准备整理携物转运姿态",
            evidence_refs=state.evidence_refs,
        )
        return

    state.object_held = bool(
        verification
        and verification.stable_bilateral_load
        and not verification.slipping
        and not verification.overloaded
    )
    ctx.checkpoint(state)
    if (
        state.object_held
        and verification is not None
        and verification.lift_height_m + LIFT_HEIGHT_TOLERANCE_M
        < skill_input.minimum_lift_height_m
        and state.verification_attempts < 4
    ):
        # 业务要求的是物体实际离开支撑面的最低高度，而不是机械指令恰好发出
        # 同样的距离。真实负载下存在有限跟踪误差，因此根据实时物体位姿只补
        # 足尚缺的高度；双侧承载一旦失效就不会进入这条局部恢复路径。
        state.pending_lift_distance_m = min(
            skill_input.minimum_lift_height_m
            - verification.lift_height_m
            + LIFT_HEIGHT_TOLERANCE_M,
            maximum_lift_segment_m,
        )
        state.lift_completed = False
        ctx.checkpoint(state)
        ctx.report(
            "stage.recovering",
            summary="双侧承载稳定，按真实物体高度补足剩余抬升距离",
            evidence_refs=state.evidence_refs,
        )
        return
    if state.object_held and state.verification_attempts < 4:
        ctx.report(
            "stage.recovering",
            summary="双侧持物状态仍明确，进行一次独立稳定性复核",
            evidence_refs=state.evidence_refs,
        )
        return
    await _request_agent_decision(
        ctx,
        skill_input,
        state,
        reason="独立抓取验证失败或局部复核预算耗尽",
        details={
            "failure_phase": "lift_verification",
            "verification_attempts": state.verification_attempts,
            "object_held": state.object_held,
        },
    )


async def _prepare_transport(
    ctx: SkillContext,
    controller: DepalletizingGraspController,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
) -> None:
    """整理为携物行走姿态，并用新的 RobotState Observation 收敛结果。

    抬升成功只证明箱体离开支撑面；双臂若保持抓取时的外展姿态，底盘转向
    会产生很大的扫掠半径。这个 Stage 使用 GraspPlanning 给出的几何姿态，
    随后重新读取实时持物状态，避免把抬升时的旧 JSON 当成导航事实。
    """

    candidate = _selected_candidate(state)
    if (
        candidate is None
        or state.target_revision is None
        or state.verified_held_object is None
    ):
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="整理转运姿态前缺少已验证抓取结果",
            details={},
        )
        return

    transport_revision = (
        state.verified_held_object.object_pose.revision or state.target_revision
    )
    if not state.transport_poses:
        plan_result = await ctx.execute(
            f"grasp:plan-transport:{state.plan_revision}:{candidate.candidate_id}",
            controller.plan_transport_posture(
                skill_input,
                held_object=state.verified_held_object,
                target_revision=transport_revision,
            ),
        )
        _remember_result_evidence(state, plan_result)
        plan = _parse_value(
            _find_observation(plan_result, "transport_posture"),
            TransportPostureValue,
        )
        if (
            plan_result.status != "succeeded"
            or plan is None
            or plan.object_ref != skill_input.object_ref
        ):
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="无法根据当前真实抓取关系生成携物姿态",
                details={"status": plan_result.status},
            )
            return
        state.transport_clearance_poses = plan.clearance_poses
        state.transport_poses = plan.transport_poses
        ctx.checkpoint(state)

    if not state.transport_clearance_completed:
        result = await ctx.execute(
            f"grasp:prepare-transport-clearance:{state.plan_revision}:{candidate.candidate_id}",
            controller.prepare_transport(
                skill_input,
                candidate=candidate,
                transport_poses=state.transport_clearance_poses,
                target_revision=transport_revision,
            ),
        )
        _remember_result_evidence(state, result)
        if result.status != "succeeded":
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="双臂未能在保持持物的情况下退出来源堆垛",
                details={
                    "status": result.status,
                    "physical_effect": result.physical_effect,
                    "placement_completed": False,
                },
            )
            return
        state.transport_clearance_completed = True
        ctx.checkpoint(state)

    if not state.transport_completed:
        result = await ctx.execute(
            f"grasp:prepare-transport:{state.plan_revision}:{candidate.candidate_id}",
            controller.prepare_transport(
                skill_input,
                candidate=candidate,
                transport_poses=state.transport_poses,
                target_revision=transport_revision,
            ),
        )
        _remember_result_evidence(state, result)
        if result.status != "succeeded":
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason="双臂未能在保持持物的情况下进入转运姿态",
                details={
                    "status": result.status,
                    "physical_effect": result.physical_effect,
                    "placement_completed": False,
                },
            )
            return
        state.transport_completed = True
        ctx.checkpoint(state)

    robot_result = await ctx.execute(
        f"grasp:transport-robot-state:{state.plan_revision}:{candidate.candidate_id}",
        controller.get_robot_state(),
    )
    load_result = await ctx.execute(
        f"grasp:transport-tool-load:{state.plan_revision}:{candidate.candidate_id}",
        controller.verify_tool_load(skill_input),
    )
    object_result = await ctx.execute(
        f"grasp:transport-object:{state.plan_revision}:{candidate.candidate_id}",
        controller.locate_target(skill_input),
    )
    for action_result in (robot_result, load_result, object_result):
        _remember_result_evidence(state, action_result)

    robot_observation = _find_observation(robot_result, "robot.state")
    load_observation = _find_observation(load_result, "robot.tool_load")
    object_observation = _find_observation(object_result, "target_pose")
    robot_state = _parse_value(robot_observation, RobotStateValue)
    tool_load = _parse_value(load_observation, ToolLoadObservationValue)
    target = _parse_value(object_observation, TargetPoseValue)
    if (
        robot_result.status != "succeeded"
        or load_result.status != "succeeded"
        or object_result.status != "succeeded"
        or robot_state is None
        or tool_load is None
        or target is None
        or robot_state.robot_id != ctx.robot_ref
        or target.object_ref != skill_input.object_ref
        or not tool_load.condition_satisfied
        or tool_load.slip_detected
        or tool_load.overload_detected
        or tool_load.sensor_fault
    ):
        await _request_agent_decision(
            ctx,
            skill_input,
            state,
            reason="转运姿态完成后无法确认双侧仍稳定持有同一物体",
            details={
                "robot_status": robot_result.status,
                "load_status": load_result.status,
                "object_status": object_result.status,
            },
        )
        return

    tool_poses = {}
    for ref in skill_input.tool_refs:
        tool_state = robot_state.tool_states.get(ref)
        end_effector = (
            robot_state.end_effectors.get(tool_state.side)
            if tool_state is not None
            else None
        )
        if end_effector is None:
            await _request_agent_decision(
                ctx,
                skill_input,
                state,
                reason=f"转运姿态下缺少{ref}的实时末端位姿",
                details={},
            )
            return
        tool_poses[ref] = target.pose.model_copy(
            update={
                "frame_id": end_effector.frame_id,
                "position_m": end_effector.position,
                "orientation_xyzw": end_effector.quaternion_xyzw,
            }
        )

    # HeldObjectState仍是grasp-object自己的结果模型；navigation/place不会读取或
    # 复制它，而会从各自Execution中的通用状态与实时物体观测重新建立本地状态。
    await capture_stage_rgb(ctx, state, skill_name="grasp-object", point="completed")
    refreshed = state.verified_held_object.model_copy(
        update={
            "tool_poses": tool_poses,
            "object_pose": target.pose,
            "object_size_m": target.extent_m,
            "grasp_candidate_id": candidate.candidate_id,
            "grasp_confidence": target.identity_confidence,
            "robot_state_revision": (
                robot_observation.revision
                if robot_observation and robot_observation.revision
                else f"robot-state-generation:{robot_state.generation}"
            ),
            "evidence_refs": list(dict.fromkeys(state.evidence_refs)),
        }
    )
    ctx.report(
        "stage.completed",
        stage="prepare_transport",
        stage_status="completed",
        summary="双臂已整理为携物转运姿态并确认稳定承载",
        evidence_refs=refreshed.evidence_refs,
    )
    ctx.complete(GraspObjectResult(held_object=refreshed))


async def _retry_observation_or_ask_agent(
    ctx: SkillContext,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    *,
    reason: str,
    details: dict,
) -> None:
    """在观测预算内重新感知，耗尽后再唤醒 Robot Agent。"""
    state.observation_attempts += 1

    if state.observation_attempts < 2:
        _reset_for_observation(state, preserve_attempts=True)
        ctx.checkpoint(state)
        ctx.report(
            "stage.recovering",
            summary=f"{reason}；将刷新目标观测后重试",
            evidence_refs=state.evidence_refs,
        )
        return
    await _request_agent_decision(
        ctx,
        skill_input,
        state,
        reason=reason,
        details=details,
    )


async def _recover_approach_or_ask_agent(
    ctx: SkillContext,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    *,
    reason: str,
    details: dict,
) -> None:
    """无物理接触时切换候选；接触或效果不明时交给 Agent 决策。

    接触约束插入失败后，末端可能仍钩在箱沿。此时直接执行下一候选的 transfer
    会把恢复动作变成一条未经验证的拖拽轨迹。因此 Runtime 先 hold，Skill
    保留当前计划和证据并请求 Robot Agent；只有明确没有物理效果时才允许本地
    切换候选。这里不新增动态 Stage，也不猜测接触后的安全撤离方向。
    """

    hook_contacts = details.get("hook_contacts")
    physical_effect = details.get("physical_effect")
    if (
        physical_effect in {"possible", "confirmed", "unknown"}
        or isinstance(hook_contacts, dict)
        and any(hook_contacts.values())
    ):
        await _request_agent_decision(
            ctx, skill_input, state, reason=reason, details=details
        )
        return

    if _is_precontact_planning_failure(details):
        candidate = _selected_candidate(state)
        candidate_index = next(
            (
                index for index, item in enumerate(state.candidates)
                if item.candidate_id == state.selected_candidate_id
            ),
            -1,
        )
        if (
            candidate is not None
            and candidate.strategy not in state.failed_precontact_strategies
            and not any(
                item.strategy == candidate.strategy
                for item in state.candidates[candidate_index + 1:]
            )
        ):
            state.failed_precontact_strategies.append(candidate.strategy)
            ctx.checkpoint(state)

    if _select_next_candidate(state):
        ctx.checkpoint(state)
        ctx.report(
            "stage.recovering",
            summary=f"{reason}；切换到候选 {state.selected_candidate_id}",
            evidence_refs=state.evidence_refs,
        )
        return
    if _is_precontact_planning_failure(details):
        # 当前候选已用实时观测生成，且动作尚未产生物理效果。相同现场下再次
        # 选择同一抓取策略只会得到相同IK/碰撞错误；记住实际候选策略，让Agent
        # 只能切换到尚未证明失败的策略，全部耗尽后必须结束当前SubTask。
        await _request_agent_decision(
            ctx, skill_input, state, reason=reason, details=details
        )
        return
    if state.observation_attempts < 2:
        state.observation_attempts += 1
        _reset_for_observation(state, preserve_attempts=True)
        ctx.checkpoint(state)
        ctx.report(
            "stage.recovering",
            summary=f"{reason}；候选已耗尽，将刷新目标和候选",
            evidence_refs=state.evidence_refs,
        )
        return
    await _request_agent_decision(
        ctx,
        skill_input,
        state,
        reason=reason,
        details=details,
    )


async def _recover_grasp_or_ask_agent(
    ctx: SkillContext,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    *,
    reason: str,
    details: dict,
) -> None:
    """在抓取预算内重新接近候选，避免无界自主重试。"""

    if state.grasp_attempts < 2:
        if not _select_next_candidate(state):
            state.observation_attempts = 0
            _reset_for_observation(state, preserve_attempts=True)
        ctx.checkpoint(state)
        ctx.report(
            "stage.recovering",
            summary=f"{reason}；将更新接近方案后重试",
            evidence_refs=state.evidence_refs,
        )
        return
    await _request_agent_decision(
        ctx,
        skill_input,
        state,
        reason=reason,
        details=details,
    )


def _is_precontact_planning_failure(details: dict) -> bool:
    return details.get("physical_effect") == "none" and details.get("error_code") in {
        "PLANNING_FAILED",
        "CANDIDATE_UNREACHABLE",
    }


def _allowed_agent_actions(
    state: GraspObjectState,
    details: dict,
) -> list[str]:
    phase = details.get("failure_phase")
    if phase == "lift_verification" and not state.object_held:
        # 未通过承载验证不等于已经脱离箱沿；没有安全退出证据时不能换候选。
        return ["abort_subtask"]
    if details.get("error_code") == "REQUIRED_TOOL_CONTACT_LOST":
        # 受约束运动期间下钩已经脱离凹槽。重新观测只能看到脱钩后的状态，
        # 不能恢复首侧承载；允许模型重启观察会重复同一物理动作并扩大风险。
        # Action 已经执行 hold，因此这里只允许结束当前 SubTask，交给上层恢复。
        return ["abort_subtask"]
    if phase == "primary_contact_lost":
        # 下钩接触已经丢失时，重新观测不会恢复物理承载，只会让模型在
        # VerifyToolLoad与hold之间循环。此时Robot已有hold证据，只允许终止。
        return ["abort_subtask"]
    if phase == "primary_load_proof":
        # 外拉Action已经结束，实时物体位姿却没有随动。重复Locate不会产生
        # 新的物理进展，继续让模型选择restart只会把同一失败循环放大。
        return ["abort_subtask"]
    if state.primary_clamped:
        if phase in {"initial_extraction", "extraction_adjustment"} and (
            details.get("error_code") in {"PLANNING_FAILED", "CANDIDATE_UNREACHABLE"}
            and details.get("physical_effect") in {None, "", "none"}
        ):
            return ["abort_subtask"]
        if phase == "secondary_approach" and state.secondary_replan_attempted:
            return ["abort_subtask"]
        return ["restart_observation", "abort_subtask"]
    if state.object_held:
        return (
            ["retry_verification", "abort_subtask"]
            if state.verification_attempts < 4 else ["abort_subtask"]
        )
    if _is_precontact_planning_failure(details):
        available = {
            "direct_bilateral",
            "left_extract_first",
            "right_extract_first",
        }.difference(state.failed_precontact_strategies)
        return ["change_strategy", "abort_subtask"] if available else ["abort_subtask"]
    return [
        "restart_observation",
        "select_candidate",
        "change_strategy",
        "abort_subtask",
    ]


async def _request_agent_decision(
    ctx: SkillContext,
    skill_input: GraspObjectInput,
    state: GraspObjectState,
    *,
    reason: str,
    details: dict,
) -> None:
    """把语义不确定性升级给 Robot Agent，并校验计划版本。"""

    if state.pending_decision_key is None:
        state.decision_revision += 1
        state.pending_decision_key = f"grasp:decision:{state.decision_revision}"
        ctx.checkpoint(state)
    decision_key = state.pending_decision_key
    assert decision_key is not None
    ctx.report(
        "decision.required",
        summary=reason,
        evidence_refs=state.evidence_refs,
        stage=state.stage,
        stage_status="waiting_agent",
        expectation=_stage_expectation(state.stage),
        deviation=reason,
        next_step="request_agent",
    )
    allowed_actions = _allowed_agent_actions(state, details)
    decision = await ctx.request_agent(
        decision_key,
        reason,
        {
            "object_ref": skill_input.object_ref,
            "stage": state.stage,
            "plan_revision": state.plan_revision,
            "active_strategy": state.active_strategy,
            "selected_candidate_id": state.selected_candidate_id,
            "candidate_ids": [item.candidate_id for item in state.candidates],
            "object_held": state.object_held,
            "primary_clamped": state.primary_clamped,
            "extraction_primary_side": state.extraction_primary_side,
            "allowed_actions": allowed_actions,
            "failed_precontact_strategies": list(state.failed_precontact_strategies),
            "available_strategies": [
                strategy
                for strategy in (
                    "direct_bilateral",
                    "left_extract_first",
                    "right_extract_first",
                )
                if strategy not in state.failed_precontact_strategies
            ],
            "details": details,
            "evidence_refs": state.evidence_refs,
        },
        GraspAgentDecision,
    )

    if decision.action not in allowed_actions:
        ctx.fail(
            "INVALID_AGENT_DECISION",
            f"当前物理状态不允许 {decision.action}；允许动作：{allowed_actions}",
            decision.evidence_refs,
        )
        return

    if decision.expected_plan_revision != state.plan_revision:
        ctx.fail(
            "STALE_AGENT_DECISION",
            "Robot Agent 的决定基于过期计划版本，拒绝修改当前执行",
            decision.evidence_refs,
        )
        return

    _remember_evidence(state, decision.evidence_refs)
    if decision.action == "abort_subtask":
        if (
            state.primary_clamped or state.object_held
            or details.get("failure_phase") == "lift_verification"
        ) and details.get("hold_status") != "succeeded":
            # Agent决定终止时，第一侧可能已经完成夹紧和外拉。此时直接返回
            # failed会让Pilot释放Worker，而Robot仍停在被改变的物理状态；外层
            # Recovery随后重跑整个Skill既不安全，也无法再取得真实停止证据。
            # 复用既有stop Action先确认hold；此前失败处理已经hold时不重复下发。
            controller = _controller(ctx)
            hold_action = (
                controller.hold_object(skill_input, reason=decision.reason)
                if state.object_held
                else controller.safe_stop(
                    skill_input, reason=decision.reason, mode="safe"
                )
            )
            hold_result = await ctx.execute_stop(
                f"grasp:abort:hold:{state.decision_revision}", hold_action
            )
            _remember_result_evidence(state, hold_result)
            if not (
                hold_result.status == "succeeded"
                and hold_result.physical_effect == "confirmed"
            ):
                ctx.fail(
                    "AGENT_ABORTED_GRASP_STATE_UNKNOWN",
                    f"{decision.reason}；Robot hold 未确认",
                    state.evidence_refs,
                )
                return
        ctx.fail("AGENT_ABORTED_GRASP", decision.reason, state.evidence_refs)
        return

    if (
        details.get("failure_phase") == "partial_grasp"
        and not state.primary_clamped
        and not state.object_held
        and decision.action
        in {"restart_observation", "change_strategy", "select_candidate"}
    ):
        # Agent 只决定是否重试以及采用什么策略；退出动作本身来自刚执行过的
        # 候选几何，不让模型生成坐标，也不引入新的恢复 Stage。
        if not await _recover_unheld_partial_grasp(
            ctx, _controller(ctx), skill_input, state
        ):
            return

    if decision.action == "restart_observation":
        state.observation_attempts = 0
        state.verification_attempts = 0
        if state.primary_clamped:
            # 第一侧仍真实夹持时，重新观测必须留在当前 grasp Stage，并保留
            # 已完成的外拉进度。普通 observation reset 会清空持物事实，导致
            # 下一轮重复外拉或让另一策略接管，这是物理上不安全的。
            state.stage = "grasp"
            state.extraction_reobserved = False
            if details.get("failure_phase") == "secondary_approach":
                state.secondary_replan_attempted = True
        else:
            state.grasp_attempts = 0
            _reset_for_observation(state, preserve_attempts=True)
    elif decision.action == "change_strategy":
        if state.primary_clamped:
            ctx.fail(
                "UNSAFE_AGENT_DECISION",
                "第一侧仍在承载物体，不能切换抓取策略；只能重新观测或安全终止",
                state.evidence_refs,
            )
            return
        if decision.strategy is None:
            ctx.fail("INVALID_AGENT_DECISION", "change_strategy 缺少 strategy")
            return
        if (
            decision.strategy == "auto"
            or decision.strategy in state.failed_precontact_strategies
        ):
            ctx.fail(
                "INVALID_AGENT_DECISION",
                "change_strategy 必须选择当前现场尚未失败的具体抓取策略",
                decision.evidence_refs,
            )
            return
        state.active_strategy = decision.strategy
        state.observation_attempts = 0
        # 策略切换仍属于同一次抓取恢复，不能把尝试次数清零后在左右外拉之间
        # 无界往返。Robot Agent可以选择新策略，但总尝试预算继续沿用。
        state.verification_attempts = 0
        _reset_for_observation(state, preserve_attempts=True)
    elif decision.action == "select_candidate":
        if state.primary_clamped:
            ctx.fail(
                "UNSAFE_AGENT_DECISION",
                "第一侧仍在承载物体，不能切换候选；只能重新观测或安全终止",
                state.evidence_refs,
            )
            return
        selected = next(
            (
                candidate
                for candidate in state.candidates
                if candidate.candidate_id == decision.selected_candidate_id
            ),
            None,
        )
        if selected is None:
            ctx.fail("INVALID_AGENT_DECISION", "Agent 选择了不存在的抓取候选")
            return
        state.selected_candidate_id = selected.candidate_id
        _reset_for_approach(state)
    elif decision.action == "retry_verification":
        if not state.object_held:
            ctx.fail(
                "INVALID_AGENT_DECISION",
                "没有已确认的持物状态，不能直接重试验证",
            )
            return
        state.stage = "lift_and_verify"
        # 只复核传感状态，不伪造抬升完成，也不清零本次执行的复核预算。
        state.verification_only = True

    state.pending_decision_key = None
    state.plan_revision += 1
    ctx.checkpoint(state)
    ctx.report(
        "agent.decision_applied",
        summary=f"已应用 Robot Agent 决定：{decision.action}；{decision.reason}",
        evidence_refs=state.evidence_refs,
    )


def _reset_for_observation(
    state: GraspObjectState,
    *,
    preserve_attempts: bool,
) -> None:
    """清除依赖旧场景版本的状态，并返回目标观测 Stage。"""

    if not preserve_attempts:
        state.observation_attempts = 0
    state.stage = "observe_target"
    state.target_observation_ref = None
    state.target_revision = None
    state.target_pose = None
    state.target_extent_m = None
    state.candidates = []
    state.selected_candidate_id = None
    state.approach_plan = []
    state.approach_cursor = 0
    state.pregrasp_observation_ref = None
    state.primary_clamped = False
    state.extraction_planned_from_contact = False
    state.extraction_primary_side = None
    state.extraction_completed = False
    state.extraction_adjustments = 0
    state.extraction_reobserved = False
    state.secondary_replan_attempted = False
    state.object_held = False
    state.lift_completed = False
    state.verification_only = False
    state.verified_held_object = None
    state.transport_clearance_poses = []
    state.transport_clearance_completed = False
    state.transport_poses = []
    state.transport_completed = False


def _reset_for_approach(state: GraspObjectState) -> None:
    """保留当前目标和候选，只重新建立 Approach 短计划。"""

    state.stage = "approach"
    state.approach_plan = []
    state.approach_cursor = 0
    state.pregrasp_observation_ref = None
    state.primary_clamped = False
    state.extraction_planned_from_contact = False
    state.extraction_primary_side = None
    state.extraction_completed = False
    state.extraction_reobserved = False
    state.extraction_adjustments = 0
    state.object_held = False
    state.lift_completed = False
    state.verification_only = False
    state.verified_held_object = None
    state.transport_clearance_poses = []
    state.transport_clearance_completed = False
    state.transport_poses = []
    state.transport_completed = False


def _select_next_candidate(state: GraspObjectState) -> bool:
    """切换到排序后的下一个候选；没有候选时返回 False。"""

    current_index = next(
        (
            index
            for index, candidate in enumerate(state.candidates)
            if candidate.candidate_id == state.selected_candidate_id
        ),
        -1,
    )
    next_candidate = next(
        (
            candidate for candidate in state.candidates[current_index + 1:]
            if candidate.strategy not in state.failed_precontact_strategies
        ),
        None,
    )
    if next_candidate is None:
        return False
    state.selected_candidate_id = next_candidate.candidate_id
    state.plan_revision += 1
    _reset_for_approach(state)
    return True


def _selected_candidate(state: GraspObjectState) -> GraspCandidate | None:
    """读取当前候选，不根据数组位置隐式猜测。"""

    return next(
        (
            candidate
            for candidate in state.candidates
            if candidate.candidate_id == state.selected_candidate_id
        ),
        None,
    )


def _rank_candidates(
    candidates: list[GraspCandidate],
    strategy: GraspStrategy,
) -> list[GraspCandidate]:
    """显式策略按评分选择；auto保留实时环境 Provider 的排序。"""

    if strategy == "auto":
        # Ability已经结合当前邻箱净空和真实可达性给候选排序。再次按静态
        # score排序会把密集堆垛需要的单侧外拉重新排到direct之后。
        return list(candidates)

    return sorted(
        candidates,
        key=lambda item: (
            1 if item.strategy == strategy else 0,
            item.score,
        ),
        reverse=True,
    )


def _find_observation(
    result: ActionResult,
    kind: str,
) -> Observation | None:
    """从 Action 最终结果中读取指定类型的最后一条观测。"""

    return next(
        (item for item in reversed(result.observations) if item.kind == kind),
        None,
    )


def _find_observation_in_feedback(
    feedback: ActionFeedback,
    kind: str,
) -> Observation | None:
    """从流式反馈中读取指定类型的最后一条观测。"""

    return next(
        (item for item in reversed(feedback.observations) if item.kind == kind),
        None,
    )


def _parse_value(
    observation: Observation | None,
    model: type[ValueModel],
) -> ValueModel | None:
    """校验观测值；结构不兼容时交由恢复逻辑处理。"""

    if observation is None or observation.value is None:
        return None
    try:
        return model.model_validate(observation.value)
    except ValidationError:
        return None


def _remember_result_evidence(
    state: GraspObjectState,
    result: ActionResult,
) -> None:
    """保存 Action 结果及其观测中的稳定证据引用。"""

    _remember_evidence(state, result.evidence_refs)
    for observation in result.observations:
        _remember_evidence(state, observation.evidence_refs)


def _remember_evidence(state: GraspObjectState, refs: list[str]) -> None:
    """按首次出现顺序去重保存证据引用。"""

    state.evidence_refs = list(dict.fromkeys([*state.evidence_refs, *refs]))


def _report_feedback(
    ctx: SkillContext,
    state: GraspObjectState,
    event: str,
    feedback: ActionFeedback,
) -> None:
    """把低频业务反馈转成前端和 Agent 可观察的事件。"""

    _remember_evidence(state, feedback.evidence_refs)
    for observation in feedback.observations:
        _remember_evidence(state, observation.evidence_refs)
    ctx.report(
        event,
        summary=feedback.message or feedback.status,
        evidence_refs=feedback.evidence_refs,
        stage=state.stage,
        stage_status="running",
        expectation=_stage_expectation(state.stage),
        observation_summary=feedback.message or feedback.status,
        deviation=(
            feedback.message if feedback.severity in {"warning", "critical"} else None
        ),
        progress=feedback.progress,
        next_step="safe_stop" if feedback.severity == "critical" else "continue_stage",
    )


def _verification_achieved(
    skill_input: GraspObjectInput,
    candidate: GraspCandidate,
    result: ActionResult,
    verification: GraspVerificationValue | None,
) -> bool:
    """根据独立证据判断大期望是否达到，而非只看 Action 成功。"""

    return bool(
        result.status == "succeeded"
        and verification is not None
        and verification.candidate_id == candidate.candidate_id
        and verification.held
        and verification.stable_bilateral_load
        and not verification.slipping
        and not verification.overloaded
        and (
            verification.lift_height_m + LIFT_HEIGHT_TOLERANCE_M
            >= skill_input.minimum_lift_height_m
        )
        and verification.stable_duration_ms >= STABLE_DURATION_MS
    )


def _stage_expectation(stage: str) -> str:
    """返回当前抓取 Stage 的用户可理解期望。"""

    return {
        "observe_target": "获得新鲜目标位姿，并生成可用且排序稳定的抓取候选",
        "approach": "末端到达当前候选的预抓取位，误差保持在限制内",
        "grasp": "双侧夹具形成稳定承载；外拉策略在第二侧动作前必须重新观测",
        "lift_and_verify": "稳定承载持续有效，物体完成抬升并形成 HeldObjectState",
        "prepare_transport": "双臂收拢到低扫掠转运姿态，并重新确认稳定持物",
    }.get(stage, "抓取执行保持在已声明的安全范围内")


async def on_stop(
    ctx: SkillContext,
    request: StopRequest,
) -> StopOutcome:
    """从检查点读取物理阶段，并通过专用停止入口收敛。"""

    skill_input = ctx.input(GraspObjectInput)
    state = ctx.load_state(
        GraspObjectState,
        GraspObjectState(active_strategy=skill_input.preferred_strategy),
    )
    controller = _controller(ctx)

    if state.object_held:
        hold_result = await ctx.execute_stop(
            f"grasp:stop:hold:{request.id}",
            controller.hold_object(skill_input, reason=request.reason),
        )
        safe = (
            hold_result.status == "succeeded"
            and hold_result.physical_effect == "confirmed"
        )
        return ctx.stop_outcome(
            safe=safe,
            summary=(
                "已确认夹爪保持物体，等待人工或后续恢复"
                if safe
                else "无法确认夹爪已安全保持物体"
            ),
            physical_state=("holding_object" if safe else "holding_state_unknown"),
            requires_intervention=True,
            evidence_refs=[
                *state.evidence_refs,
                *hold_result.evidence_refs,
            ],
        )

    stop_result = await ctx.execute_stop(
        f"grasp:stop:motion:{request.id}",
        controller.safe_stop(
            skill_input,
            reason=request.reason,
            mode=request.mode,
        ),
    )
    active_state_known = (
        stop_result.status == "succeeded" and stop_result.physical_effect == "confirmed"
    )
    return ctx.stop_outcome(
        safe=active_state_known,
        summary=(
            "抓取未持物，当前执行已停止"
            if active_state_known
            else "停止后机械臂或夹爪物理状态未知"
        ),
        physical_state=("stopped_without_object" if active_state_known else "unknown"),
        requires_intervention=not active_state_known,
        evidence_refs=[
            *state.evidence_refs,
            *stop_result.evidence_refs,
        ],
    )
