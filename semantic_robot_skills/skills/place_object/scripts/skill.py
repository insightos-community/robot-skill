"""place-object 的 schema v2 双工具 Stage 状态机。"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from .stage_evidence import capture_stage_rgb

from semantic_robot_skill_sdk import (
    Action, ActionResult, FeedbackRequest, Observation, SkillCancelled, SkillContext,
    SkillFailure, StopOutcome, StopRequest,
)

from .controller import (
    PLACED_OBJECT_OBSERVATION, RELEASED_OBJECT_OBSERVATION,
    ROBOT_STATE_OBSERVATION, TARGET_POSE_OBSERVATION, TARGET_SLOT_OBSERVATION,
    TOOL_LOAD_OBSERVATION, PlacementController,
)
from .models import (
    GetRobotStateParameters, LocateObjectParameters, MoveToPostureParameters,
    MoveToolTargetsParameters, ObservePlacementSlotParameters, PlaceObjectInput,
    PlaceObjectRunState, PlacementAgentDecision, PlacementConstraints,
    ReleaseToolsParameters, SafeStopPlacementParameters, ToolCommand, ToolPoseTarget,
    ToolLoadObservationValue, VerifyPlacementParameters, VerifyToolLoadParameters,
)

VERIFY_TOOL_LOAD_ACTION = "robot.verify_tool_load"
LOCATE_OBJECT_ACTION = "perception.locate_object"
GET_ROBOT_STATE_ACTION = "robot.get_state"
OBSERVE_SLOT_ACTION = "perception.observe_placement_target"
MOVE_ACTION = "motion.move_end_effector"
RELEASE_ACTION = "gripper.release"
VERIFY_STABILITY_ACTION = "perception.verify_placement"
SAFE_STOP_ACTION = "gripper.hold_object"
MOVE_TO_POSTURE_ACTION = "motion.move_to_posture"

# 这些是 Skill 的受控策略，不是调用方每次下发的设备旋钮。工具力限值最终仍由
# RobotDeployment/Profile 约束；这里负责一个放置流程允许的速度和有限恢复次数。
PLACEMENT_POLICY = PlacementConstraints()
# 只限制密集放置最后一段单侧承载下降。layout001实测中按双侧接近速度
# 下降会放大单侧钩内相对滑动；0.04m/s仍只作用于约一个退钩净空高度，
# 不拖慢双侧preplace、导航或抓取，也不改变通用IK和碰撞边界。
SINGLE_TOOL_LANDING_SPEED_MPS = 0.04
# 仅用于目标上方的长距离无接触下降。最后10～18cm仍使用原接触工艺速度，
# 插入、外拉、抬升和携物导航均不受影响。Robot Profile和实际轨迹碰撞
# 检查仍会进一步限制速度，不能用这个值绕过机械或安全上限。
PREPLACE_SPEED_MPS = 0.12
FREE_SPACE_DESCENT_SPEED_MPS = 0.15


async def run(ctx: SkillContext) -> None:
    """每次从持久化检查点推进一个固定 Stage。"""

    try:
        skill_input = ctx.input(PlaceObjectInput)
        state = ctx.load_state(
            PlaceObjectRunState,
            PlaceObjectRunState(),
        )
        controller = ctx.controller("placement")
        if not isinstance(controller, PlacementController):
            raise SkillFailure("PLACEMENT_CONTROLLER_INVALID", "placement Controller 未注册")
        ctx.check_cancelled()
        ctx.log("info", "进入放置 Stage", skill_name="place-object", skill_version="0.4.42", stage=state.stage)
        ctx.report(
            "stage.running", summary=f"正在执行放置阶段：{state.stage}",
            stage=state.stage, stage_status="running", expectation=_stage_expectation(state.stage),
            next_step="continue_stage", evidence_refs=state.evidence_refs,
        )
        # 不在安全 retreat、停止或等待 Agent 时增加摄影等待。
        if state.stage in {
            "verify_held_object", "observe_target_slot", "approach", "release",
            "verify_stability", "restore_travel_posture",
        }:
            await capture_stage_rgb(ctx, state, skill_name="place-object")
        if state.stage == "verify_held_object":
            await _verify_held_object(ctx, controller, skill_input, state)
        elif state.stage == "observe_target_slot":
            await _observe_target_slot(ctx, controller, skill_input, state)
        elif state.stage == "plan_approach":
            _plan_approach(ctx, controller, skill_input, state)
        elif state.stage == "approach":
            await _approach(ctx, controller, skill_input, state)
        elif state.stage == "release":
            await _release(ctx, controller, skill_input, state)
        elif state.stage == "retreat":
            await _retreat(ctx, controller, skill_input, state)
        elif state.stage == "verify_stability":
            await _verify_stability(ctx, controller, skill_input, state)
        elif state.stage == "restore_travel_posture":
            await _restore_travel_posture(
                ctx, controller, skill_input, state
            )
        elif state.stage == "await_agent_decision":
            await _request_agent_decision(ctx, skill_input, state)
        elif state.stage != "completed":
            raise SkillFailure("UNKNOWN_STAGE", f"未知放置 Stage：{state.stage}")
    except SkillCancelled:
        raise
    except SkillFailure as failure:
        ctx.fail(failure.code, failure.message, failure.evidence)


async def _verify_held_object(ctx, controller, skill_input, state) -> None:
    robot_result = await ctx.execute(
        "place.verify-held.robot-state",
        Action.from_model(
            action_type=GET_ROBOT_STATE_ACTION,
            parameters=GetRobotStateParameters(),
            timeout_seconds=8,
            label="读取当前Robot与工具状态",
        ),
    )
    _require_action_succeeded(robot_result, "ROBOT_STATE_FAILED", "读取Robot状态失败")
    robot_observation = _find_observation(robot_result.observations, ROBOT_STATE_OBSERVATION)
    if robot_observation is None:
        raise SkillFailure("ROBOT_STATE_MISSING", "RobotState未返回robot.state Observation")
    robot_state = controller.parse_robot_state(robot_observation)
    tool_refs = controller.discover_tool_refs(robot_state)

    load_result = await ctx.execute(
        "place.verify-held.tool-load",
        Action.from_model(
            action_type=VERIFY_TOOL_LOAD_ACTION,
            parameters=VerifyToolLoadParameters(tool_refs=tool_refs),
            timeout_seconds=8,
            label="连续验证当前双工具承载",
        ),
    )
    _require_action_succeeded(load_result, "VERIFY_LOAD_FAILED", "工具承载复核失败")
    load_observation = _find_observation(load_result.observations, TOOL_LOAD_OBSERVATION)
    if load_observation is None:
        raise SkillFailure("TOOL_LOAD_MISSING", "RobotState未返回robot.tool_load Observation")

    object_result = await ctx.execute(
        "place.verify-held.object",
        Action.from_model(
            action_type=LOCATE_OBJECT_ACTION,
            parameters=LocateObjectParameters(object_ref=skill_input.object_ref),
            timeout_seconds=8,
            label="重新观测待放置物体",
        ),
    )
    _require_action_succeeded(object_result, "OBJECT_OBSERVATION_FAILED", "实时物体观测失败")
    object_observation = _find_observation(object_result.observations, TARGET_POSE_OBSERVATION)
    if object_observation is None:
        raise SkillFailure("OBJECT_OBSERVATION_MISSING", "ObjectPerception未返回target_pose Observation")

    # place-object只在自己的Execution中组合本地持物状态；它不读取grasp结果，
    # RobotState和Ability也不维护任何全局HeldObjectState。
    state.verified_held_object = controller.verify_held_object(
        skill_input.object_ref,
        robot_observation,
        load_observation,
        object_observation,
        robot_ref=ctx.robot_ref,
    )
    _extend_evidence(
        state,
        robot_result.evidence_refs,
        load_result.evidence_refs,
        object_result.evidence_refs,
        state.verified_held_object.evidence_refs,
    )
    state.stage = "observe_target_slot"
    ctx.checkpoint(state)
    ctx.report(
        "stage.completed",
        stage="verify_held_object",
        stage_status="completed",
        summary="已用实时Robot、载荷和物体观测确认双侧稳定持有同一周转箱",
        evidence_refs=state.evidence_refs,
    )


async def _observe_target_slot(ctx, controller, skill_input, state) -> None:
    observation = None
    if state.target_observation_ref and not state.force_target_refresh:
        observation = ctx.observation(state.target_observation_ref)
    if observation is None and not state.force_target_refresh:
        observation = ctx.latest_observation(
            TARGET_SLOT_OBSERVATION, subject_ref=skill_input.target.target_ref,
            max_age_ms=PLACEMENT_POLICY.target_observation_max_age_ms,
        )
    if observation is None:
        action = Action.from_model(
            action_type=OBSERVE_SLOT_ACTION,
            parameters=ObservePlacementSlotParameters(
                target_ref=skill_input.target.target_ref,
                object_ref=skill_input.object_ref,
                pose_hint=skill_input.target.pose_hint,
                extent_hint_m=skill_input.target.extent_hint_m,
                category_hint=skill_input.target.category_hint,
            ), timeout_seconds=8, label="观测目标槽位",
        )
        result = await ctx.execute(f"place.observe-slot.{state.target_refreshes}", action)
        _require_action_succeeded(result, "OBSERVE_SLOT_FAILED", "目标槽位观测失败")
        observation = _find_observation(result.observations, TARGET_SLOT_OBSERVATION)
        _extend_evidence(state, result.evidence_refs)
    if observation is None:
        raise SkillFailure("TARGET_SLOT_OBSERVATION_MISSING", "ObjectPerception 未返回目标槽位 Observation")
    slot = controller.parse_target_slot(observation, expected_target_ref=skill_input.target.target_ref)
    if not slot.free or not slot.reachable:
        if state.target_refreshes < PLACEMENT_POLICY.max_target_refreshes:
            state.target_refreshes += 1
            state.force_target_refresh = True
            ctx.checkpoint(state)
            ctx.report("stage.recovering", summary="目标槽位被占用或不可达，将刷新一次")
            return
        _wait_for_agent(ctx, state, "目标槽位持续被占用或不可达")
        return
    state.target_slot = slot
    state.target_observation_ref = observation.id
    state.force_target_refresh = False
    state.approach_plan = None
    state.stage = "plan_approach"
    _extend_evidence(state, observation.evidence_refs, slot.evidence_refs)
    ctx.checkpoint(state)
    ctx.report(
        "stage.completed",
        stage="observe_target_slot",
        stage_status="completed",
        summary="已实时确认目标堆叠列空闲且可达",
        evidence_refs=state.evidence_refs,
    )


def _plan_approach(ctx, controller, skill_input, state) -> None:
    held = state.verified_held_object
    if held is None or state.target_slot is None:
        raise SkillFailure("PLACEMENT_PLAN_INPUT_MISSING", "双工具短计划缺少持物或槽位事实")
    horizontal_error = controller.horizontal_alignment_error_m(held, state.target_slot)
    local_limit = controller.local_alignment_limit_m(held, PLACEMENT_POLICY)
    if horizontal_error > local_limit:
        # Robot Agent负责跨工位导航，Place Skill只补偿一个箱体工作区内的末端
        # 接近。局部范围从实时箱体几何推导，不能再把最终3cm放置精度误当成
        # 接近前误差；真正的可达性和碰撞仍由ManipulatorMotion实时规划验证。
        raise SkillFailure(
            "PLACEMENT_BASE_ALIGNMENT_REQUIRED",
            (
                f"持物中心与槽位水平相差 {horizontal_error:.3f}m，超过当前箱体的 "
                f"局部接近范围 {local_limit:.3f}m；请重新选择可通行的基座工作位"
            ),
            state.evidence_refs,
        )
    state.approach_plan = controller.build_approach_plan(held, state.target_slot, PLACEMENT_POLICY)
    state.stage = "approach"
    ctx.checkpoint(state)
    ctx.report("stage.completed", stage="plan_approach", stage_status="completed", summary="已按抓取时真实相对位姿生成双工具放置短计划")


async def _approach(ctx, controller, skill_input, state) -> None:
    plan = state.approach_plan
    if plan is None:
        raise SkillFailure("APPROACH_PLAN_MISSING", "执行接近前缺少双工具短计划")
    held = _require_live_held(state)
    preplace = plan.waypoints[0]
    action = Action.from_model(
        action_type=MOVE_ACTION,
        parameters=MoveToolTargetsParameters(
            targets=preplace.targets,
            purpose="place",
            object_ref=held.object_ref,
            target_ref=skill_input.target.target_ref,
            target_revision=plan.target_revision,
            expected_object_pose=preplace.object_pose,
            maximum_speed_mps=PREPLACE_SPEED_MPS,
            max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
        ),
        timeout_seconds=30,
        feedback=FeedbackRequest(
            observation_kinds=["end_effector_progress", "collision_proximity"],
            interval_ms=200,
        ),
        label="双工具移动到 preplace",
    )
    result = await ctx.execute(
        f"place.approach.{plan.target_revision}.{state.approach_recoveries}.preplace",
        action,
    )
    _extend_evidence(state, result.evidence_refs)
    if result.status != "succeeded":
        recovery = controller.recover_approach(
            error_code=result.error_code,
            recoveries=state.approach_recoveries,
            constraints=PLACEMENT_POLICY,
        )
        if recovery.disposition == "refresh_target":
            state.approach_recoveries += 1
            state.force_target_refresh = True
            state.target_slot = None
            state.approach_plan = None
            state.stage = "observe_target_slot"
            ctx.checkpoint(state)
            return
        _wait_for_agent(ctx, state, recovery.reason)
        return
    tool_load = await _verify_tool_load(
        ctx, state, held.tool_refs, "approach.preplace"
    )
    if not _bilateral_load_stable(tool_load, held):
        raise SkillFailure(
            "LOAD_UNSTABLE_DURING_APPROACH",
            "preplace后双侧稳定承载失效",
            state.evidence_refs,
        )

    release_waypoint = plan.waypoints[1]
    support_waypoint = (
        plan.early_release_waypoints[0]
        if plan.early_release_waypoints
        else release_waypoint
    )
    free_space_descent = controller.build_free_space_descent_waypoint(
        preplace,
        support_waypoint,
        state.target_slot.approach_vector,
        object_height_m=held.object_size_m[2],
    )
    if free_space_descent is not None:
        # 先快速到达目标上方的低速区，再沿用原来的支撑面落座动作。拆段只
        # 缩短无接触长行程，不能跳过双侧承载和实际轨迹碰撞检查。
        action = Action.from_model(
            action_type=MOVE_ACTION,
            parameters=MoveToolTargetsParameters(
                targets=free_space_descent.targets,
                purpose="place",
                object_ref=held.object_ref,
                target_ref=skill_input.target.target_ref,
                target_revision=plan.target_revision,
                expected_object_pose=free_space_descent.object_pose,
                maximum_speed_mps=FREE_SPACE_DESCENT_SPEED_MPS,
                max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
            ),
            timeout_seconds=30,
            feedback=FeedbackRequest(
                observation_kinds=["end_effector_progress", "collision_proximity"],
                interval_ms=200,
            ),
            label="双工具快速下降到支撑面上方低速区",
        )
        result = await ctx.execute(
            f"place.approach.{plan.target_revision}.free-space-descent",
            action,
        )
        _extend_evidence(state, result.evidence_refs)
        if result.status != "succeeded":
            if result.error_code == "PLANNING_FAILED":
                # 该路点只用于加速长距离下降。数值IK在中间点不可达时，
                # 回到原有的直接落座路径，不改变后续放置工艺和完成判断。
                ctx.report(
                    "stage.recovered",
                    stage="approach",
                    stage_status="running",
                    summary="加速下降中间点不可达，改用直接落座路径",
                    evidence_refs=state.evidence_refs,
                )
            else:
                _wait_for_agent(ctx, state, "目标上方无接触下降失败")
                return

    supported_slide = False
    early_clearance_completed = False
    if plan.early_release_tool_ref is not None:
        early_tool = plan.early_release_tool_ref
        remaining_tool = next(ref for ref in held.tool_refs if ref != early_tool)
        staged_support, *early_withdraw = plan.early_release_waypoints
        # 先由双侧把箱体放到空闲侧的临时位置，让托盘真实承重后再撤内侧手。
        # 不能在空中提前释放，否则外侧工具会单独承担整段下降并触发过载。
        action = Action.from_model(
            action_type=MOVE_ACTION,
            parameters=MoveToolTargetsParameters(
                targets=staged_support.targets,
                purpose="place",
                object_ref=held.object_ref,
                target_ref=skill_input.target.target_ref,
                target_revision=plan.target_revision,
                expected_object_pose=staged_support.object_pose,
                maximum_speed_mps=PLACEMENT_POLICY.max_approach_speed_mps,
                max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
            ),
            timeout_seconds=30,
            feedback=FeedbackRequest(
                observation_kinds=["end_effector_progress", "collision_proximity"],
                interval_ms=200,
            ),
            label="双工具下降到空闲侧临时支撑位置",
        )
        result = await ctx.execute(
            f"place.approach.{plan.target_revision}.early.staged-support",
            action,
        )
        _extend_evidence(state, result.evidence_refs)
        staged_support_transfer = controller.support_transfer_confirmed(
            result.observations,
            object_ref=held.object_ref,
            target_ref=skill_input.target.target_ref,
        )
        if result.status != "succeeded" and not staged_support_transfer:
            _wait_for_agent(ctx, state, "双侧下降到目标上方失败")
            return
        if result.status != "succeeded":
            # 接触落座动作可能因为关节终点还差约一度而超时，但同一次动作的
            # 实时观测已经确认箱体在目标范围内并由托盘承重。此时继续等待
            # Robot Agent只会把已完成的物理落座误判为失败；后续仍会执行逐侧
            # 撤手和最终稳定性复核，因此这里只采信本次动作自身的支撑证据。
            ctx.report(
                "stage.recovered",
                stage="approach",
                stage_status="running",
                summary="末端未完全收敛，但箱体已由目标托盘承重，继续完成撤手",
                evidence_refs=state.evidence_refs,
            )
        if not staged_support_transfer:
            # 末端命令到位与MuJoCo接触稳定存在一个很短的物理沉降窗口。
            # 直接读取命令结束瞬间的载荷会偶发看到箱底仍高约几厘米，随后
            # 场景快照却已稳定落在托盘上。复用通用VerifyPlacement等待一次
            # 既有的稳定时长，再决定是否释放；不增加循环试探或新Ability。
            staged_support_transfer, staged_ready_for_slide = (
                await _verify_support_transfer_after_unload(
                    ctx, controller, skill_input, state, attempt="staged"
                )
            )
            # 临时落座会故意保留几厘米侧向偏移，供内侧钩退出相邻箱间隙；
            # 这里仅要求箱体仍在目标区域、有托盘接触且没有明显倾倒。
            # 最终4cm中心误差仍在短推后和完成阶段检查，不能提前拿最终中心
            # 标准阻止本来就是用来消除该偏移的推入动作。
            staged_support_transfer = (
                staged_support_transfer or staged_ready_for_slide
            )
        if not staged_support_transfer:
            raise SkillFailure(
                "SUPPORT_TRANSFER_UNCONFIRMED",
                "箱体尚未确认由目标托盘承重，禁止提前释放内侧工具",
                state.evidence_refs,
            )
        state.support_transfer_confirmed = True
        ctx.checkpoint(state)

        await _release_tool(
            ctx,
            controller,
            skill_input,
            state,
            held,
            tool_ref=early_tool,
            remaining_tool_ref=remaining_tool,
            phase="early",
        )
        # “夹具已打开”只证明上压片不再施力，不能证明下钩已退出箱体凹槽。
        # 从此刻的真实末端Pose重建反向落座、水平退钩和越过箱沿的路径。
        withdrawal_state_result = await ctx.execute(
            f"place.approach.{plan.target_revision}.early.robot-state",
            Action.from_model(
                action_type=GET_ROBOT_STATE_ACTION,
                parameters=GetRobotStateParameters(),
                timeout_seconds=8,
                label="读取内侧下钩撤出前的真实末端位置",
            ),
        )
        _require_action_succeeded(
            withdrawal_state_result,
            "ROBOT_STATE_FAILED",
            "读取内侧下钩撤出前Robot状态失败",
        )
        withdrawal_state_observation = _find_observation(
            withdrawal_state_result.observations, ROBOT_STATE_OBSERVATION
        )
        if withdrawal_state_observation is None:
            raise SkillFailure(
                "ROBOT_STATE_MISSING",
                "内侧下钩撤出前RobotState未返回robot.state Observation",
            )
        withdrawal_state = controller.parse_robot_state(
            withdrawal_state_observation
        )
        released_tool_state = withdrawal_state.tool_states.get(early_tool)
        # 打开夹具后的实时接触状态才决定是否还需要反向落座。第四个密集列
        # 的真实回放中，下钩已经无接触；继续向下 unseat 会让空钩重新碰到
        # 箱沿，随后水平动作虽返回完成，竖直撤离仍会从碰撞状态起步。
        # 这里只跳过已由传感器确认完成的机械步骤，不放宽 IK 或碰撞规则。
        skip_unseat = (
            released_tool_state is not None
            and released_tool_state.hook_contact is False
        )
        early_withdraw = controller.rebase_early_withdrawal(
            staged_support,
            early_withdraw,
            tool_ref=early_tool,
            current_pose=controller.tool_pose_from_robot_state(
                withdrawal_state,
                early_tool,
                revision=plan.target_revision,
            ),
            skip_unseat=skip_unseat,
        )
        _extend_evidence(state, withdrawal_state_result.evidence_refs)
        # 临时偏置给已经退出凹槽的内侧手留出了竖直通道。用实时双末端
        # 约束固定仍贴着箱体的外侧手后，先把内侧空钩抬过箱沿；这样箱体
        # 随后推回最终列时不会再次进入空钩范围。
        early_clearance = early_withdraw[-1]
        horizontal_withdraw = early_withdraw[:-1]
        for waypoint in horizontal_withdraw:
            # 只提交正在退钩的一侧，并把仍贴着箱体的另一侧声明为固定接触端。
            # Ability 会在每条 Action 开始时读取固定侧的实时 FK 位姿，再交给
            # Robot SDK 做固定末端约束；不能把同一个实时 Pose 伪装成普通运动
            # 目标，否则多末端 IK 可能整体换构型，动作虽然返回成功，空钩却
            # 没有真正离开箱体。箱体已由托盘承重时，Ability 会以现有支撑
            # 转移证据接受固定侧自然卸载，不会把正常卸载误判为脱手。
            targets = [waypoint.targets[0]]
            required_contact_tools = [remaining_tool]
            action = Action.from_model(
                action_type=MOVE_ACTION,
                parameters=MoveToolTargetsParameters(
                    targets=targets,
                    purpose=(
                        "disengage" if waypoint.name == "retreat"
                        else waypoint.name
                    ),
                    object_ref=held.object_ref,
                    target_ref=skill_input.target.target_ref,
                    target_revision=plan.target_revision,
                    # 箱体已在本轮 staged-support 中由目标托盘承重。退钩时
                    # 保留这一实时支撑目标，让 Ability 能区分“托盘接管重量”
                    # 与“空中脱手”；remaining_tool 仍作为固定末端参与规划。
                    expected_object_pose=staged_support.object_pose,
                    maximum_speed_mps=(
                        0.02 if waypoint.name == "unseat"
                        else PLACEMENT_POLICY.max_approach_speed_mps
                    ),
                    max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
                    required_contact_tools=required_contact_tools,
                ),
                timeout_seconds=30,
                label=f"托盘承重后撤离内侧工具：{waypoint.name}",
            )
            result = await ctx.execute(
                f"place.approach.{plan.target_revision}.early.{waypoint.name}",
                action,
            )
            _extend_evidence(state, result.evidence_refs)
            if result.status != "succeeded":
                _wait_for_agent(ctx, state, "托盘承重后内侧工具撤离失败，外侧工具保持当前状态")
                return

        if not await _raise_early_tool_clear(
            ctx,
            controller,
            skill_input,
            state,
            held,
            plan,
            early_tool,
            horizontal_withdraw,
            early_clearance,
        ):
            _wait_for_agent(ctx, state, "内侧空钩未能抬过箱沿")
            return
        early_clearance_completed = True

        # 内侧下钩已水平清出最终箱体投影。此时打开外侧夹具，但保持末端
        # 位置不变，使它仍贴着箱壁并能完成一次真实的支撑面推入。
        await _release_tool(
            ctx,
            controller,
            skill_input,
            state,
            held,
            tool_ref=remaining_tool,
            remaining_tool_ref=None,
            phase="supported",
        )
        # 箱体已经由托盘承重；读取外侧工具与箱体实时位置，只补偿当前
        # 水平剩余误差。不能使用固定场景坐标，也不能在手离开箱壁后空推。
        robot_result = await ctx.execute(
            f"place.approach.{plan.target_revision}.supported-slide.robot-state",
            Action.from_model(
                action_type=GET_ROBOT_STATE_ACTION,
                parameters=GetRobotStateParameters(),
                timeout_seconds=8,
                label="读取外侧工具短推前的实时位置",
            ),
        )
        _require_action_succeeded(
            robot_result,
            "ROBOT_STATE_FAILED",
            "读取支撑面短推前Robot状态失败",
        )
        robot_observation = _find_observation(
            robot_result.observations, ROBOT_STATE_OBSERVATION
        )
        if robot_observation is None:
            raise SkillFailure(
                "ROBOT_STATE_MISSING",
                "支撑面短推前RobotState未返回robot.state Observation",
            )
        current_robot_state = controller.parse_robot_state(robot_observation)
        current_tool_pose = controller.tool_pose_from_robot_state(
            current_robot_state,
            remaining_tool,
            revision=plan.target_revision,
        )
        object_result = await ctx.execute(
            f"place.approach.{plan.target_revision}.supported-slide.before-push",
            Action.from_model(
                action_type=LOCATE_OBJECT_ACTION,
                parameters=LocateObjectParameters(object_ref=held.object_ref),
                timeout_seconds=8,
                label="观测外侧工具短推前的箱体位置",
            ),
        )
        _require_action_succeeded(
            object_result,
            "SUPPORTED_SLIDE_OBSERVATION_FAILED",
            "支撑面短推前无法重新观测箱体",
        )
        object_observation = _find_observation(
            object_result.observations, TARGET_POSE_OBSERVATION
        )
        if object_observation is None:
            raise SkillFailure(
                "SUPPORTED_SLIDE_OBSERVATION_MISSING",
                "支撑面短推前ObjectPerception未返回target_pose Observation",
            )
        _extend_evidence(
            state,
            robot_result.evidence_refs,
            object_result.evidence_refs,
        )
        remaining_target = controller.supported_slide_correction_target(
            object_observation,
            plan,
            ToolPoseTarget(
                tool_ref=remaining_tool,
                target_pose=current_tool_pose,
            ),
        )
        release_waypoint = release_waypoint.model_copy(update={
            # 内侧空手已水平退出并抬过箱沿，不再把它钉在世界坐标。
            # 短推只表达保留侧工具与箱体的实际动作；规划器仍会对
            # 整机轨迹做自碰和场景碰撞检查。这避免两个已无刚性关系的
            # 末端被组成不可达的多末端IK，导致推入动作根本没有下发。
            "targets": [remaining_target],
        })
        supported_slide = True

    action = Action.from_model(
        action_type=MOVE_ACTION,
        parameters=MoveToolTargetsParameters(
            targets=release_waypoint.targets,
            # 支撑面短推从工具仍接触当前箱体的状态开始，使用已有 disengage
            # 语义只排除本次目标箱体；邻箱、托盘和 Robot 自碰撞仍照常检查。
            purpose="disengage" if supported_slide else "place",
            object_ref=held.object_ref,
            target_ref=skill_input.target.target_ref,
            target_revision=plan.target_revision,
            expected_object_pose=state.target_slot.placement_pose,
            maximum_speed_mps=(
                min(
                    PLACEMENT_POLICY.max_approach_speed_mps,
                    SINGLE_TOOL_LANDING_SPEED_MPS,
                )
                if supported_slide
                else PLACEMENT_POLICY.max_approach_speed_mps
            ),
            max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
            # 箱体已经由托盘承重，短推时剩余夹具的竖直载荷自然会下降；
            # 继续把“承重接触”作为运动中断条件，会在正常推入刚开始时误停。
            # expected_object_pose 让 Ability 使用正常运动终点容差，动作后的
            # VerifyPlacement 再以实时箱体位姿判定是否真正推到位。
            required_contact_tools=[],
        ),
        timeout_seconds=30,
        feedback=FeedbackRequest(
            observation_kinds=["end_effector_progress", "collision_proximity"],
            interval_ms=200,
        ),
        label=(
            "由外侧工具沿支撑面推入最终位置"
            if supported_slide
            else "将箱体下降到目标支撑面"
        ),
    )
    result = await ctx.execute(
        f"place.approach.{plan.target_revision}.{state.approach_recoveries}.release",
        action,
    )
    _extend_evidence(state, result.evidence_refs)
    if result.status != "succeeded":
        # 支撑面短推的关节目标可能被托盘和箱体的真实接触阻止，此时
        # Runtime的关节终点失败不等于放置失败。只对这个明确工艺动作立即
        # 重新观测实物：箱体已进入目标区并由目标支撑承重时，才允许继续
        # 释放和清出工具；否则保持hold并上报。这避免用不断放宽的关节容差
        # 代替场景完成判定，也不会沿用短推前的过期支撑证据。
        support_transfer, retry_supported_slide = (
            await _verify_support_transfer_after_unload(
                ctx,
                controller,
                skill_input,
                state,
                attempt=0,
                require_final_alignment=True,
            )
            if supported_slide
            else (False, False)
        )
        if not support_transfer and not retry_supported_slide:
            _wait_for_agent(
                ctx,
                state,
                "支撑面推入最终位置失败"
                if supported_slide
                else "箱体下降到目标支撑面失败",
            )
            return
        ctx.report(
            "stage.recovered",
            stage="approach",
            stage_status="completed",
            summary="短推关节终点未收敛，实时观测确认箱体已由目标支撑承重",
            evidence_refs=state.evidence_refs,
        )
    else:
        if supported_slide:
            # 短推改变了箱体位姿，推前的support_transfer_confirmed
            # 不能证明推后仍可安全释放。即使关节命令成功，也要用
            # 新的VerifyPlacement观测确认箱体仍在目标内、由托盘承重
            # 且没有明显倾斜。
            support_transfer, retry_supported_slide = (
                await _verify_support_transfer_after_unload(
                    ctx,
                    controller,
                    skill_input,
                    state,
                    attempt=0,
                    require_final_alignment=True,
                )
            )
        else:
            support_transfer = state.support_transfer_confirmed or controller.support_transfer_confirmed(
                result.observations,
                object_ref=held.object_ref,
                target_ref=skill_input.target.target_ref,
            )
    if supported_slide and not support_transfer and retry_supported_slide:
        # 箱体已经在目标托盘上，但单手第一次推入可能因接触顺应变成
        # 转动或侧向滑动。补推前先把已撤出的空钩抬过箱沿，避免共享躯干
        # 在第二次推入时让它扫到已经放好的相邻箱。
        if not early_clearance_completed and not await _raise_early_tool_clear(
            ctx,
            controller,
            skill_input,
            state,
            held,
            plan,
            early_tool,
            horizontal_withdraw,
            early_clearance,
        ):
            _wait_for_agent(ctx, state, "补推前内侧空钩未能抬过箱沿")
            return
        early_clearance_completed = True
        # 只重新读取一次真实工具/箱体位姿，补一次剩余水平误差；抬手会因
        # 共享躯干改变外侧末端，所以补推从本次抓取关系生成的计划接触位
        # 重新贴住箱壁，不从抬手后的空中Pose继续平移。
        retry_object_result = await ctx.execute(
            f"place.approach.{plan.target_revision}.supported-slide.retry-before-push",
            Action.from_model(
                action_type=LOCATE_OBJECT_ACTION,
                parameters=LocateObjectParameters(object_ref=held.object_ref),
                timeout_seconds=8,
                label="观测支撑面补推前的箱体位置",
            ),
        )
        _require_action_succeeded(
            retry_object_result,
            "SUPPORTED_SLIDE_OBSERVATION_FAILED",
            "支撑面补推前无法重新观测箱体",
        )
        retry_object_observation = _find_observation(
            retry_object_result.observations, TARGET_POSE_OBSERVATION
        )
        if retry_object_observation is None:
            raise SkillFailure(
                "SUPPORTED_SLIDE_OBSERVATION_MISSING",
                "支撑面补推前ObjectPerception未返回target_pose Observation",
            )
        _extend_evidence(
            state,
            retry_object_result.evidence_refs,
            retry_object_observation.evidence_refs,
        )
        retry_remaining = controller.supported_slide_correction_target(
            retry_object_observation,
            plan,
            next(
                target for target in plan.waypoints[1].targets
                if target.tool_ref == remaining_tool
            ),
        )
        await ctx.execute(
            f"place.approach.{plan.target_revision}.1.release",
            Action.from_model(
                action_type=MOVE_ACTION,
                parameters=MoveToolTargetsParameters(
                    targets=[retry_remaining],
                    purpose="disengage",
                    object_ref=held.object_ref,
                    target_ref=skill_input.target.target_ref,
                    target_revision=plan.target_revision,
                    expected_object_pose=state.target_slot.placement_pose,
                    maximum_speed_mps=min(
                        PLACEMENT_POLICY.max_approach_speed_mps,
                        SINGLE_TOOL_LANDING_SPEED_MPS,
                    ),
                    max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
                ),
                timeout_seconds=30,
                feedback=FeedbackRequest(
                    observation_kinds=[
                        "end_effector_progress", "collision_proximity",
                    ],
                    interval_ms=200,
                ),
                label="按实时剩余误差补推一次箱体",
            ),
        )
        support_transfer, _ = await _verify_support_transfer_after_unload(
            ctx,
            controller,
            skill_input,
            state,
            attempt=1,
            require_final_alignment=True,
        )

    if support_transfer:
        state.support_transfer_confirmed = True
    elif supported_slide:
        raise SkillFailure(
            "SUPPORT_TRANSFER_UNCONFIRMED",
            "支撑面推入后无法确认箱体仍在目标内并保持可释放姿态",
            state.evidence_refs,
        )
    else:
        tool_load = await _verify_tool_load(
            ctx, state, held.tool_refs, "approach.release"
        )
        if not _bilateral_load_stable(tool_load, held) and not support_transfer:
            raise SkillFailure(
                "LOAD_UNSTABLE_DURING_APPROACH",
                "下降后双侧稳定承载失效且未确认支撑转移",
                state.evidence_refs,
            )

    if supported_slide and not early_clearance_completed:
        if not await _raise_early_tool_clear(
            ctx,
            controller,
            skill_input,
            state,
            held,
            plan,
            early_tool,
            horizontal_withdraw,
            early_clearance,
        ):
            _wait_for_agent(ctx, state, "箱体推正后内侧工具未能抬过箱沿")
            return

    state.stage = "release"
    ctx.checkpoint(state)
    ctx.report(
        "stage.completed",
        stage="approach",
        stage_status="completed",
        summary=(
            "托盘承重后内侧工具已撤离，外侧工具已推入最终位置"
            if plan.early_release_tool_ref is not None
            else "双工具已沿放置短计划到达释放位置"
        ),
        evidence_refs=state.evidence_refs,
    )


async def _release_tool(
    ctx,
    controller,
    skill_input,
    state,
    held,
    *,
    tool_ref: str,
    remaining_tool_ref: str | None,
    phase: str,
) -> None:
    if tool_ref in state.released_tool_refs:
        return
    plan = state.approach_plan
    if plan is None:
        raise SkillFailure("RELEASE_PLAN_MISSING", "释放前缺少双工具短计划")
    action = Action.from_model(
        action_type=RELEASE_ACTION,
        parameters=ReleaseToolsParameters(
            object_ref=held.object_ref,
            tools=[ToolCommand(
                tool_ref=tool_ref,
                target_position_m=PLACEMENT_POLICY.release_opening_m,
                maximum_force_n=PLACEMENT_POLICY.max_contact_force_n,
            )],
            target_ref=skill_input.target.target_ref,
            target_revision=plan.target_revision,
        ),
        timeout_seconds=15,
        label=f"逐侧释放工具 {tool_ref}",
    )
    result = await ctx.execute(
        f"place.release.{plan.target_revision}.{phase}.{tool_ref}",
        action,
    )
    _extend_evidence(state, result.evidence_refs)
    if (
        result.status in {"interrupted", "stopped"}
        or result.physical_effect in {"unknown", "possible"}
    ):
        raise SkillFailure(
            "RELEASE_STATE_UNKNOWN",
            "单侧释放物理状态无法确认，禁止重放",
            state.evidence_refs,
        )
    _require_action_succeeded(result, "RELEASE_FAILED", "单侧释放命令失败")
    released = _find_observation(result.observations, RELEASED_OBJECT_OBSERVATION)
    if released is None or not controller.release_result_confirms_tool(
        released,
        tool_ref=tool_ref,
        object_ref=held.object_ref,
    ):
        raise SkillFailure(
            "RELEASE_TOOL_STATE_UNKNOWN",
            "单侧释放没有返回明确工具状态",
            state.evidence_refs,
        )
    if remaining_tool_ref is not None:
        if not state.support_transfer_confirmed:
            remaining_load = await _verify_tool_load(
                ctx,
                state,
                (remaining_tool_ref,),
                f"release.{phase}",
            )
            if not controller.remaining_tool_is_stable(
                remaining_load,
                tool_ref=remaining_tool_ref,
                object_ref=held.object_ref,
            ):
                raise SkillFailure(
                    "REMAINING_TOOL_NOT_STABLE",
                    "第一侧释放后另一侧未保持稳定承载",
                    state.evidence_refs,
                )
    elif not controller.release_result_confirms_tools_empty(released):
        raise SkillFailure(
            "TOOLS_NOT_EMPTY",
            "第二侧释放后两个工具未确认清空",
            state.evidence_refs,
        )
    state.released_tool_refs.append(tool_ref)
    state.release_cursor = len(state.released_tool_refs)
    ctx.checkpoint(state)


async def _release(ctx, controller, skill_input, state) -> None:
    """释放尚未打开的工具；已在preplace撤离的一侧不会被重放。"""

    plan = state.approach_plan
    if plan is None:
        raise SkillFailure("RELEASE_PLAN_MISSING", "释放前缺少双工具短计划")
    held = _require_live_held(state)
    pending = [ref for ref in held.tool_refs if ref not in state.released_tool_refs]
    if not pending:
        state.release_confirmed = True
        state.stage = "retreat"
        ctx.checkpoint(state)
        # 密集放置会在 approach 内按“内侧释放—推正—外侧释放”完成双工具
        # 清空；进入显式 release Stage 时虽然无需重复物理命令，仍必须上报
        # 阶段完成，保证执行历史与普通双侧释放路径具有一致的可审阅结构。
        ctx.report(
            "stage.completed",
            stage="release",
            stage_status="completed",
            summary="两个工具已在逐侧推入流程中释放并确认清空",
            evidence_refs=state.evidence_refs,
        )
        return
    tool_ref = pending[0]
    remaining = pending[1] if len(pending) > 1 else None
    await _release_tool(
        ctx,
        controller,
        skill_input,
        state,
        held,
        tool_ref=tool_ref,
        remaining_tool_ref=remaining,
        phase="final",
    )
    if len(state.released_tool_refs) == len(held.tool_refs):
        state.release_confirmed = True
        state.stage = "retreat"
        ctx.report(
            "stage.completed",
            stage="release",
            stage_status="completed",
            summary="两个工具已逐侧释放并确认清空",
            evidence_refs=state.evidence_refs,
        )
    ctx.checkpoint(state)


async def _retreat(ctx, controller, skill_input, state) -> None:
    if not state.release_confirmed or state.approach_plan is None:
        raise SkillFailure("RETREAT_PRECONDITION_FAILED", "两个工具未确认清空，禁止撤离")
    retreat_waypoints = state.approach_plan.waypoints[2:]
    if state.retreat_cursor >= len(retreat_waypoints):
        state.retreat_confirmed = True
        state.stage = "verify_stability"
        ctx.checkpoint(state)
        return
    waypoint = retreat_waypoints[state.retreat_cursor]
    held = _require_live_held(state)
    targets = waypoint.targets
    rebased_retreat_waypoints = None
    if state.approach_plan.early_release_tool_ref is not None:
        early_tool = state.approach_plan.early_release_tool_ref
        remaining_tool = next(
            ref for ref in held.tool_refs if ref != early_tool
        )
        if state.retreat_cursor == 0:
            # 支撑面短推和最终释放会让保留侧末端偏离放置前的理论 Pose。
            # 从此刻的真实末端位置重放原四段退钩增量，避免第一段为了追赶
            # 旧绝对目标反向拖动已经平放的箱体。只有 unseat 成功后才把
            # 重建结果写入检查点，因此相同 Action 重入仍保持完全相同内容。
            release_state = await _get_robot_state(
                ctx, controller, state, "final-retreat-anchor"
            )
            rebased_retreat_waypoints = controller.rebase_early_withdrawal(
                state.approach_plan.waypoints[1],
                retreat_waypoints,
                tool_ref=remaining_tool,
                current_pose=controller.tool_pose_from_robot_state(
                    release_state,
                    remaining_tool,
                    revision=state.approach_plan.target_revision,
                ),
            )
            merged_waypoints = []
            for original, rebased in zip(
                retreat_waypoints, rebased_retreat_waypoints, strict=True
            ):
                merged_waypoints.append(original.model_copy(update={
                    "targets": [
                        rebased.targets[0],
                        *[
                            target for target in original.targets
                            if target.tool_ref != remaining_tool
                        ],
                    ],
                }))
            rebased_retreat_waypoints = merged_waypoints
            waypoint = rebased_retreat_waypoints[0]
        targets = [
            target for target in waypoint.targets
            if target.tool_ref != early_tool
        ]
        if not targets:
            raise SkillFailure("RETREAT_PLAN_INVALID", "最终撤离缺少外侧工具目标")
        if waypoint.name == "retreat":
            # 内侧手在短推前已经抬高。最后释放的外侧手不能从凹槽边缘
            # 斜向抬升，否则下钩会沿箱壁滑动并把已对中的箱体带偏。
            # 先在当前高度水平移到下一路点的完整夹具净空，下一段再抬高。
            robot_state = await _get_robot_state(
                ctx, controller, state, "retreat-anchor"
            )
            live_pose = controller.tool_pose_from_robot_state(
                robot_state,
                remaining_tool,
                revision=state.approach_plan.target_revision,
            )
            clearance_target = next(
                target
                for target in retreat_waypoints[state.retreat_cursor + 1].targets
                if target.tool_ref == remaining_tool
            )
            targets = [clearance_target.model_copy(update={
                "target_pose": clearance_target.target_pose.model_copy(update={
                    "position_m": (
                        clearance_target.target_pose.position_m[0],
                        clearance_target.target_pose.position_m[1],
                        live_pose.position_m[2],
                    ),
                    "orientation_xyzw": live_pose.orientation_xyzw,
                }),
            })]
        elif waypoint.name == "travel_clearance":
            # 两侧工具已经释放，最后只抬仍在低位的外侧手。把先撤手也固定为
            # 同步目标会对共享躯干施加两个完整Pose约束，真实第四箱回放中两
            # 个目标单独都可达、组合后却无解。单末端轨迹仍对整机（包括先撤
            # 手）做扫掠碰撞检查，随后最终状态也会复核两侧工具均已清空。
            targets = [targets[0]]
    action = Action.from_model(
        action_type=MOVE_ACTION,
        parameters=MoveToolTargetsParameters(
            targets=targets,
            # 逐侧释放时，内侧工具已先撤离，外侧工具释放后仍从箱体凹槽
            # 边界起步。上抬和水平撤离都使用 disengage 语义，只允许
            # 空工具离开当前箱体；邻箱、托盘和 Robot 自碰撞仍照常检查。
            purpose=(
                "disengage"
                if state.approach_plan.early_release_tool_ref is not None
                and waypoint.name in {"retreat", "travel_clearance"}
                else waypoint.name
                if waypoint.name in {"unseat", "disengage"}
                else "retreat"
            ),
            object_ref=held.object_ref,
            target_ref=skill_input.target.target_ref,
            target_revision=state.approach_plan.target_revision,
            # unseat仍处于凹槽接触区，沿用抓取seat的接触工艺速度；后续空工具
            # 横向退出和上抬才使用普通放置速度。
            maximum_speed_mps=(
                0.02 if waypoint.name == "unseat"
                # 工具完成水平脱钩后已经为空，抬回本次验证过的preplace
                # 高度和水平清出均属于无接触运动，可使用既有preplace速度。
                else PREPLACE_SPEED_MPS
                if waypoint.name in {"retreat", "travel_clearance"}
                else PLACEMENT_POLICY.max_approach_speed_mps
            ),
            max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
        ), timeout_seconds=30, label="空工具沿安全路径撤离",
    )
    result = await ctx.execute(
        (
            f"place.retreat.{state.approach_plan.target_revision}."
            f"{waypoint.name}.{state.retreat_recoveries}"
        ),
        action,
    )
    _extend_evidence(state, result.evidence_refs)
    if result.status == "succeeded":
        if rebased_retreat_waypoints is not None:
            state.approach_plan = state.approach_plan.model_copy(update={
                "waypoints": [
                    *state.approach_plan.waypoints[:2],
                    *rebased_retreat_waypoints,
                ],
            })
        state.retreat_cursor += 1
        if state.retreat_cursor == len(retreat_waypoints):
            state.retreat_confirmed = True
            state.stage = "verify_stability"
        ctx.checkpoint(state)
        if state.retreat_confirmed:
            ctx.report(
                "stage.completed",
                stage="retreat",
                stage_status="completed",
                summary="两个空工具已沿安全路径撤离目标物体",
                evidence_refs=state.evidence_refs,
            )
        return
    recovery = controller.recover_retreat(
        error_code=result.error_code, recoveries=state.retreat_recoveries,
        constraints=PLACEMENT_POLICY,
    )
    if recovery.disposition == "retry":
        state.retreat_recoveries += 1
        ctx.checkpoint(state)
        return
    # Robot Agent必须看到真实Action诊断，不能把关节终点残差、命令超时和
    # 碰撞都压缩成同一个“无法安全撤离”，否则会选择错误的恢复策略。
    detail = result.error_message or result.error_code or result.status
    _wait_for_agent(ctx, state, f"{recovery.reason}：{detail}")


async def _verify_stability(ctx, controller, skill_input, state) -> None:
    if not state.retreat_confirmed or state.approach_plan is None:
        raise SkillFailure("STABILITY_PRECONDITION_FAILED", "撤离未完成，不能验证放置")
    action = Action.from_model(
        action_type=VERIFY_STABILITY_ACTION,
        parameters=VerifyPlacementParameters(
            object_ref=skill_input.object_ref,
            target_ref=skill_input.target.target_ref,
            target_revision=state.approach_plan.target_revision,
            target_pose_hint=state.target_slot.placement_pose,
            target_extent_hint_m=state.target_slot.extent_m,
            stability_duration_ms=skill_input.target.stability_duration_ms,
        ), timeout_seconds=20, label="独立验证放置稳定性",
    )
    # 清出工具会改变箱体的受力与姿态，复核必须使用新的幂等键。
    # 否则Runtime会合法地返回清出前的旧Action结果，把后续滑动的
    # 箱体误报为已稳定。复用已有postplace_clearance_completed，不引入新状态。
    result = await ctx.execute(
        f"place.verify-stability.{int(state.postplace_clearance_completed)}."
        f"{state.stability_rechecks}",
        action,
    )
    _extend_evidence(state, result.evidence_refs)
    observation = _find_observation(result.observations, PLACED_OBJECT_OBSERVATION)
    if result.status == "succeeded" and observation is not None:
        try:
            placed = controller.parse_placed_state(
                observation, object_ref=skill_input.object_ref,
                target=skill_input.target,
            )
        except SkillFailure as failure:
            if failure.code != "PLACEMENT_NOT_STABLE":
                raise
            supported = controller.supported_placed_state(
                observation,
                object_ref=skill_input.object_ref,
                target=skill_input.target,
            )
            if (
                supported is not None
                and state.approach_plan.early_release_tool_ref is not None
                and not state.postplace_clearance_completed
            ):
                # 目标支撑已经接管箱体，但工具接触使最终稳定条件尚未成立。
                # 先按本次真实位姿清出空钩，再回到同一个正式稳定性校验。
                state.placed_object = supported
                state.stage = "restore_travel_posture"
                ctx.checkpoint(state)
                ctx.report(
                    "stage.recovering",
                    stage="verify_stability",
                    stage_status="running",
                    summary="箱体已由目标支撑承重，先清出仍接触的空工具",
                    evidence_refs=state.evidence_refs,
                )
                return
        else:
            placed.evidence_refs = list(dict.fromkeys([*placed.evidence_refs, *state.evidence_refs, *observation.evidence_refs]))
            # 放置物理结果已经成立，后续姿态恢复失败时不得重新抓取或重放释放。
            state.placed_object = placed
            state.stage = "restore_travel_posture"
            ctx.checkpoint(state)
            ctx.report(
                "stage.completed",
                stage="verify_stability",
                stage_status="completed",
                summary="物体已稳定放置，准备恢复 Robot 行走姿态",
                evidence_refs=placed.evidence_refs,
            )
            return
    if state.stability_rechecks < PLACEMENT_POLICY.max_stability_rechecks:
        state.stability_rechecks += 1
        ctx.checkpoint(state)
        ctx.report("stage.recovering", summary="稳定性证据不足，将进行一次有限复核")
        return
    _wait_for_agent(ctx, state, "正式稳定性 Observation 未通过，局部复核预算耗尽")


async def _clear_postplace_tools(ctx, controller, skill_input, state) -> None:
    """按最终箱体位姿把空工具移出，再重新验证放置结果。"""

    placed = state.placed_object
    if placed is None:
        raise SkillFailure("PLACED_RESULT_MISSING", "清出空工具前缺少已验证的放置结果")
    held = _require_live_held(state)
    robot_state = await _get_robot_state(
        ctx,
        controller,
        state,
        f"postplace-clearance.{state.decision_count}",
    )
    result = await ctx.execute(
        f"place.postplace-clearance.{state.decision_count}",
        Action.from_model(
            action_type=MOVE_ACTION,
            parameters=MoveToolTargetsParameters(
                targets=controller.postplace_clearance_targets(
                    robot_state,
                    held,
                    placed,
                    # 常规退钩仍受6cm工艺上限约束；正式放置完成后的空工具
                    # 清出要覆盖箱体沉降/推正产生的真实平移，使用既有12cm
                    # 局部接近范围。轨迹碰撞和力限仍由Ability逐点检查。
                    maximum_disengage_m=PLACEMENT_POLICY.approach_clearance_m,
                ),
                # 放置结果已经由 VerifyPlacement 确认。这段只允许空钩
                # 离开当前箱体；邻箱、托盘和 Robot 自碰撞仍正常检查。
                purpose="disengage",
                object_ref=placed.object_ref,
                target_ref=placed.target_ref,
                target_revision=state.approach_plan.target_revision,
                maximum_speed_mps=PREPLACE_SPEED_MPS,
                max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
            ),
            timeout_seconds=30,
            label="按实际放置位姿清空两只空工具",
        ),
    )
    _extend_evidence(state, result.evidence_refs)
    if result.status != "succeeded":
        _wait_for_agent(ctx, state, "放置已完成、空工具未离开实时箱体边界")
        return
    cleared_state = await _get_robot_state(
        ctx,
        controller,
        state,
        f"postplace-clearance-verify.{state.decision_count}",
    )
    contacts = [
        tool_ref
        for tool_ref in held.tool_refs
        if (
            (tool := cleared_state.tool_states.get(tool_ref)) is None
            or tool.hook_contact is not False
            or tool.clamp_contact is not False
        )
    ]
    if contacts:
        # MoveEndEffector成功只说明末端到位，不能替代工具接触结果。
        # 箱体已经稳定放置，此时只暂停收尾，不倒退重放 release/grasp。
        _wait_for_agent(
            ctx,
            state,
            f"放置已完成、空工具仍接触实时箱体：{contacts}",
        )
        return
    state.postplace_clearance_completed = True
    # 清出工具会改变箱体接触力；必须重新等待并执行正式稳定性验证，
    # 不能沿用清出前的观测作为最终结果。
    state.placed_object = None
    state.stability_rechecks = 0
    state.stage = "verify_stability"
    ctx.checkpoint(state)


async def _restore_travel_posture(ctx, controller, skill_input, state) -> None:
    """放置完成后收拢上肢，恢复可安全行走的命名姿态。"""

    placed = state.placed_object
    if placed is None:
        raise SkillFailure("PLACED_RESULT_MISSING", "恢复姿态前缺少已验证的放置结果")
    if (
        state.approach_plan is not None
        and state.approach_plan.early_release_tool_ref is not None
        and not state.postplace_clearance_completed
        # 只有工具仍接触箱体的中间支撑状态需要立即清钩；正常终态先尝试
        # travel，若折叠路径碰到邻物，再按同一真实位姿清出一次空工具。
        and not placed.gripper_empty
    ):
        await _clear_postplace_tools(ctx, controller, skill_input, state)
        return
    result = await ctx.execute(
        # 清出空工具前后的Robot状态不同，幂等键必须包含该事实；否则Pilot
        # 会合法重放清出前的PLANNING_FAILED，新的travel动作永远不会下发。
        f"place.restore-travel-posture."
        f"{int(state.postplace_clearance_completed)}.{state.decision_count}",
        Action.from_model(
            action_type=MOVE_TO_POSTURE_ACTION,
            parameters=MoveToPostureParameters(),
            timeout_seconds=30,
            label="恢复 Robot 行走姿态",
        ),
    )
    _extend_evidence(state, result.evidence_refs)
    if result.status != "succeeded":
        if (
            result.error_code == "PLANNING_FAILED"
            and not state.postplace_clearance_completed
        ):
            # 箱体已通过正式稳定性验证，失败只发生在空工具折叠到travel的
            # 场景碰撞路径。先沿放置箱体的真实边界把双钩移到基座侧，再
            # 重新验证箱体并折叠；不会重放下降、释放或抓取。
            ctx.report(
                "stage.recovering",
                stage="restore_travel_posture",
                stage_status="running",
                summary="travel折叠路径被邻物阻挡，先清出双侧空工具",
                evidence_refs=state.evidence_refs,
            )
            await _clear_postplace_tools(ctx, controller, skill_input, state)
            return
        if (
            state.postplace_clearance_completed
            and result.error_code == "PLANNING_FAILED"
        ):
            # 完成标准明确包含恢复travel姿态，不能因为箱体已经放稳就把未完成
            # 的收尾伪装成成功。清钩后状态与路径均已重新计算；若仍为确定性的
            # 规划失败，原地交给Agent重复同一个动作也不会产生新信息，直接返回
            # 准确的Skill失败，让Workflow按正常失败路径结束并保留Action诊断。
            raise SkillFailure(
                "TRAVEL_POSTURE_UNREACHABLE",
                "物体已放稳且双工具已清出，但Robot无法恢复travel姿态："
                f"{result.error_message or result.error_code}",
            )
        # 普通工位没有额外清空动作时仍需恢复姿态；绝不能倒退到 release/grasp。
        _wait_for_agent(ctx, state, "放置已完成、Robot 未恢复行走姿态")
        return
    state.travel_posture_completed = True
    await capture_stage_rgb(ctx, state, skill_name="place-object", point="completed")
    state.stage = "completed"
    placed.evidence_refs = list(
        dict.fromkeys([*placed.evidence_refs, *state.evidence_refs])
    )
    ctx.checkpoint(state)
    ctx.report(
        "stage.completed",
        stage="restore_travel_posture",
        stage_status="completed",
        summary="Robot 已恢复 travel 行走姿态",
        evidence_refs=result.evidence_refs,
    )
    ctx.complete(placed)


async def _get_robot_state(ctx, controller, state, suffix):
    result = await ctx.execute(
        f"place.robot-state.{suffix}",
        Action.from_model(
            action_type=GET_ROBOT_STATE_ACTION,
            parameters=GetRobotStateParameters(), timeout_seconds=5,
            label="读取双工具承载状态",
        ),
    )
    _require_action_succeeded(result, "ROBOT_STATE_FAILED", "读取 Robot 状态失败")
    observation = _find_observation(result.observations, ROBOT_STATE_OBSERVATION)
    if observation is None:
        raise SkillFailure("ROBOT_STATE_MISSING", "RobotState 未返回正式 robot.state Observation")
    _extend_evidence(state, result.evidence_refs, observation.evidence_refs)
    return controller.parse_robot_state(observation)


async def _raise_early_tool_clear(
    ctx,
    controller,
    skill_input,
    state,
    held,
    plan,
    early_tool,
    horizontal_withdraw,
    early_clearance,
) -> bool:
    """按已规划的相对高度抬起先撤出的空工具，并保持另一末端不动。"""

    live_state = await _get_robot_state(
        ctx, controller, state, "early-clearance-anchor"
    )
    previous_target = horizontal_withdraw[-1].targets[0]
    moving_target = early_clearance.targets[0]
    lift_delta = tuple(
        target - previous
        for target, previous in zip(
            moving_target.target_pose.position_m,
            previous_target.target_pose.position_m,
            strict=True,
        )
    )
    live_early_pose = controller.tool_pose_from_robot_state(
        live_state, early_tool, revision=plan.target_revision
    )
    clearance_target = ToolPoseTarget(
        tool_ref=early_tool,
        target_pose=live_early_pose.model_copy(update={
            "position_m": tuple(
                value + delta
                for value, delta in zip(
                    live_early_pose.position_m,
                    lift_delta,
                    strict=True,
                )
            ),
        }),
    )
    remaining_tool = next(ref for ref in held.tool_refs if ref != early_tool)
    result = await ctx.execute(
        f"place.approach.{plan.target_revision}.early.travel_clearance",
        Action.from_model(
            action_type=MOVE_ACTION,
            parameters=MoveToolTargetsParameters(
                # R1 Pro 两条手臂共享躯干自由度。这里只提交正在抬高的空手，
                # 并通过 required_contact_tools 让 Ability 用 Action 开始时的
                # 实时 FK 位姿固定另一末端。这样既允许共享躯干做必要补偿，
                # 又不会把固定侧当成第二个普通运动目标而整体换 IK 构型。
                targets=[clearance_target],
                # 水平退钩已经完成；这一段是在支撑面上把空手抬到箱沿
                # 上方，使用 Ability 已有的 placement clearance 语义。
                # expected_object_pose 使另一接触侧继续作为实时固定末端，
                # 同时允许 SDK 选择可达的无碰撞关节路径，而不是强制沿
                # 退钩阶段的狭窄笛卡尔直线继续求解。
                purpose="clearance",
                object_ref=held.object_ref,
                target_ref=skill_input.target.target_ref,
                target_revision=plan.target_revision,
                expected_object_pose=state.target_slot.placement_pose,
                maximum_speed_mps=PREPLACE_SPEED_MPS,
                max_contact_force_n=PLACEMENT_POLICY.max_contact_force_n,
                required_contact_tools=[remaining_tool],
            ),
            timeout_seconds=30,
            label="箱体推正后抬高已撤出的内侧工具",
        ),
    )
    _extend_evidence(state, result.evidence_refs)
    return result.status == "succeeded"


async def _verify_support_transfer_after_unload(
    ctx,
    controller,
    skill_input,
    state,
    *,
    attempt: int | str,
    require_final_alignment: bool = False,
) -> tuple[bool, bool]:
    """用实时放置证据确认箱体是否已由目标支撑。"""

    if state.approach_plan is None or state.target_slot is None:
        return False, False
    result = await ctx.execute(
        f"place.verify-support-transfer.{attempt}",
        Action.from_model(
            action_type=VERIFY_STABILITY_ACTION,
            parameters=VerifyPlacementParameters(
                object_ref=skill_input.object_ref,
                target_ref=skill_input.target.target_ref,
                target_revision=state.approach_plan.target_revision,
                target_pose_hint=state.target_slot.placement_pose,
                target_extent_hint_m=state.target_slot.extent_m,
                stability_duration_ms=skill_input.target.stability_duration_ms,
            ),
            timeout_seconds=20,
            label="工具卸载后确认箱体已转移到目标支撑",
        ),
    )
    _extend_evidence(state, result.evidence_refs)
    observation = _find_observation(
        result.observations, PLACED_OBJECT_OBSERVATION
    )
    if result.status != "succeeded" or observation is None:
        return False, False
    support_confirmed = controller.support_contact_confirmed(
        observation,
        object_ref=skill_input.object_ref,
        target=skill_input.target,
    )
    if not support_confirmed:
        return False, controller.supported_slide_retry_allowed(
            observation,
            object_ref=skill_input.object_ref,
            target=skill_input.target,
        )
    if require_final_alignment and not controller.supported_slide_completed(
        observation,
        object_ref=skill_input.object_ref,
        target=skill_input.target,
    ):
        # “已在目标区域”只能证明箱体没有掉落，不能证明支撑面推进已经
        # 完成。真实回放中推进超时后仍差2.7cm，却被旧逻辑直接当成成功。
        # 这里使用2cm级完成条件触发唯一一次补推，不做毫米级循环试探。
        return False, controller.supported_slide_retry_allowed(
            observation,
            object_ref=skill_input.object_ref,
            target=skill_input.target,
        )
    _extend_evidence(state, observation.evidence_refs)
    return True, False


async def _verify_tool_load(ctx, state, tool_refs, suffix):
    result = await ctx.execute(
        f"place.tool-load.{suffix}",
        Action.from_model(
            action_type=VERIFY_TOOL_LOAD_ACTION,
            parameters=VerifyToolLoadParameters(tool_refs=tool_refs),
            timeout_seconds=8,
            label="连续验证工具承载",
        ),
    )
    _require_action_succeeded(result, "TOOL_LOAD_FAILED", "工具承载验证失败")
    observation = _find_observation(result.observations, TOOL_LOAD_OBSERVATION)
    if observation is None:
        raise SkillFailure("TOOL_LOAD_MISSING", "RobotState未返回robot.tool_load Observation")
    _extend_evidence(state, result.evidence_refs, observation.evidence_refs)
    try:
        return ToolLoadObservationValue.model_validate(observation.value or {})
    except ValueError as error:
        raise SkillFailure("TOOL_LOAD_INVALID", f"工具承载观测无法解析：{error}") from error


def _bilateral_load_stable(tool_load, held_object) -> bool:
    return bool(
        tool_load.condition_satisfied
        and not tool_load.slip_detected
        and not tool_load.overload_detected
        and not tool_load.sensor_fault
        and {item.tool_ref for item in tool_load.tools} == set(held_object.tool_refs)
        and all(item.available for item in tool_load.tools)
    )


async def _request_agent_decision(ctx, skill_input, state) -> None:
    reason = state.decision_reason or "放置执行需要语义决策"
    state.decision_count += 1
    allowed_decisions = ["abort"]
    if state.release_cursor == 0:
        allowed_decisions[:0] = ["refresh_target", "retry_approach"]
    if state.release_confirmed:
        allowed_decisions[:0] = ["retry_retreat", "recheck_stability"]
    if state.placed_object is not None:
        allowed_decisions.insert(0, "retry_posture")
    decision = await ctx.request_agent(
        f"place.agent-decision.{state.decision_count}", reason,
        {
            "object_ref": skill_input.object_ref,
            "target_ref": skill_input.target.target_ref,
            "release_cursor": state.release_cursor,
            "release_confirmed": state.release_confirmed,
            "allowed_decisions": allowed_decisions,
            "evidence_refs": state.evidence_refs,
        }, PlacementAgentDecision,
    )
    state.decision_reason = None
    if decision.decision == "abort":
        raise SkillFailure("AGENT_ABORTED_PLACEMENT", decision.reason, state.evidence_refs)
    if decision.decision == "refresh_target" and state.release_cursor == 0:
        state.target_observation_ref = decision.replacement_target_observation_ref
        state.target_slot = None
        state.approach_plan = None
        state.force_target_refresh = True
        state.stage = "observe_target_slot"
    elif decision.decision == "retry_approach" and state.release_cursor == 0:
        state.stage = "plan_approach" if state.target_slot else "observe_target_slot"
    elif decision.decision == "retry_retreat" and state.release_confirmed:
        # 已得到放置观测后，retry_retreat 指的是重新清出仍接触箱体的
        # 空工具，而不是把已完成的四段退钩游标倒回去。新的恢复序号为
        # 本次实时目标提供新幂等键，旧的确定失败动作仍保留在执行历史中。
        state.retreat_recoveries += 1
        state.stage = (
            "restore_travel_posture"
            if state.placed_object is not None
            else "retreat"
        )
    elif decision.decision == "recheck_stability" and state.release_confirmed:
        state.stage = "verify_stability"
    elif decision.decision == "retry_posture" and state.placed_object is not None:
        state.stage = "restore_travel_posture"
    else:
        raise SkillFailure("AGENT_DECISION_NOT_APPLICABLE", "决定不适用于当前物理状态")
    ctx.checkpoint(state)


async def on_stop(ctx: SkillContext, request: StopRequest) -> StopOutcome:
    state = ctx.load_state(
        PlaceObjectRunState,
        PlaceObjectRunState(),
    )
    object_may_be_released = state.release_cursor > 0 or state.release_confirmed
    held = state.verified_held_object
    if held is None:
        return ctx.stop_outcome(
            safe=True,
            summary="尚未下发放置物理动作，无需保持夹具",
            physical_state="not_started",
            requires_intervention=False,
            evidence_refs=state.evidence_refs,
        )
    commands = [
        ToolCommand(
            tool_ref=ref, target_position_m=PLACEMENT_POLICY.release_opening_m,
            maximum_force_n=PLACEMENT_POLICY.max_contact_force_n, hold=True,
        )
        for ref in held.tool_refs
    ]
    action = Action.from_model(
        action_type=SAFE_STOP_ACTION,
        parameters=SafeStopPlacementParameters(
            object_ref=held.object_ref,
            tools=commands, object_may_be_released=object_may_be_released,
            reason=request.reason, mode=request.mode,
        ), timeout_seconds=10, label="保持双工具当前状态并停止放置",
    )
    result = await ctx.execute_stop(f"place.on-stop.{request.id}", action)
    safe = result.status == "succeeded" and result.physical_effect == "confirmed"
    return ctx.stop_outcome(
        safe=safe,
        summary="已保持当前工具状态" if safe else "无法确认放置停止后的工具状态",
        physical_state=(
            "partially_released" if state.release_cursor == 1
            else "released" if state.release_confirmed
            else "hold"
        ) if safe else "unknown",
        requires_intervention=not safe or object_may_be_released,
        evidence_refs=list(dict.fromkeys([*state.evidence_refs, *result.evidence_refs])),
    )


def _wait_for_agent(ctx, state, reason):
    state.decision_reason = reason
    state.stage = "await_agent_decision"
    ctx.checkpoint(state)
    ctx.report(
        "decision.required", summary=reason, evidence_refs=state.evidence_refs,
        stage=state.stage, stage_status="waiting_agent", expectation=_stage_expectation(state.stage),
        deviation=reason, next_step="request_agent",
    )


def _stage_expectation(stage):
    return {
        "verify_held_object": "确认双侧工具仍稳定承载同一周转箱",
        "observe_target_slot": "获得 schema v2 的空闲可达槽位观测",
        "plan_approach": "通过持物刚体关系生成双工具接近、释放和撤离目标",
        "approach": "双工具同步到达释放位，且每段动作后承载仍稳定",
        "release": "逐侧释放；第一侧后另一侧稳定，第二侧后两个工具清空",
        "retreat": "两个空工具沿同一安全几何撤离",
        "verify_stability": "正式 Observation 确认物体稳定、在目标内且工具为空",
        "restore_travel_posture": "物体保持稳定，Robot 上肢恢复可行走的 travel 姿态",
        "await_agent_decision": "保持最近安全状态并等待类型化决策",
    }.get(stage, "放置执行保持在批准边界内")


def _require_action_succeeded(result: ActionResult, code: str, message: str) -> None:
    if result.status != "succeeded":
        raise SkillFailure(code, f"{message}：{result.error_message or result.error_code or result.status}", result.evidence_refs)


def _find_observation(observations: Iterable[Observation], kind: str) -> Observation | None:
    matches = [item for item in observations if item.kind == kind]
    return max(matches, key=lambda item: item.observed_at) if matches else None


def _observation_age_ms(ctx: SkillContext, observed_at: datetime) -> float:
    return (ctx.now() - observed_at).total_seconds() * 1_000


def _require_live_held(state: PlaceObjectRunState):
    held = state.verified_held_object
    if held is None:
        raise SkillFailure("HELD_OBJECT_STATE_MISSING", "缺少本次执行实时读取的持物状态")
    return held


def _extend_evidence(state: PlaceObjectRunState, *groups: Iterable[str]) -> None:
    state.evidence_refs = list(dict.fromkeys([*state.evidence_refs, *(item for group in groups for item in group)]))
