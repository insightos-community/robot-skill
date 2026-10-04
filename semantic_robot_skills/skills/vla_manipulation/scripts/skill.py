"""复用 SkillContext 的 Action/检查点/停止入口，不直接访问 SDK 或模型。"""

from semantic_robot_skill_sdk import Action, SkillContext, StopRequest
import time

from .models import ManipulationInput, ManipulationResult, ManipulationState
from .verification import assess


# 两种目标共用有限尾段；这是 Skill 的完成策略，不是模型参数或 Runtime 判据。
POST_SUCCESS_ACTIONS = 20


def action(kind: str, parameters: dict, timeout=15) -> Action:
    return Action(type=f"vla.{kind}", parameters=parameters, timeout_seconds=timeout)


async def observe(ctx, inputs, key, stage, capture=False):
    result = await ctx.execute(key, action("observe_environment", {
        "target_source_id": inputs.target_source_id, "capture_images": capture,
    }))
    if result.status != "succeeded":
        ctx.fail(result.error_code or "OBSERVATION_FAILED",
                 result.error_message or "无法读取当前场景事实", result.evidence_refs)
        return None
    if capture:
        refs = (result.output or {}).get("artifact_refs", [])
        ctx.report("stage.evidence" if refs else "stage.evidence_unavailable",
                   stage=stage, summary="VLA 阶段相机证据" if refs else "阶段图像尚未同步",
                   evidence_refs=refs)
    return result.output


async def run(ctx: SkillContext):
    inputs = ctx.input(ManipulationInput)
    state = ctx.load_state(ManipulationState, default=ManipulationState())
    ctx.check_cancelled()
    ctx.report("stage.running", stage=state.stage, stage_status="running",
               summary={"validate_target": "确认目标", "prepare": "准备型号 VLA",
                        "execute_policy": "执行策略并检查目标", "verify_result": "独立验收结果"}[
                            state.stage])
    if state.stage == "validate_target":
        data = await observe(ctx, inputs, "vla:initial", state.stage, True)
        if data is None:
            return
        state.generation = data["generation"]
        state.robot_id = data["robot_state"]["robot_id"]
        if inputs.objective == "grasp":
            if data["target"]["state"]["fixture"]:
                ctx.fail("TARGET_IS_FIXTURE", "指定目标是固定设施，不能按自由物体抓取")
                return
            state.initial_target_height = data["target"]["pose"]["position"][2]
        next_stage = "prepare"
    elif state.stage == "prepare":
        binding = await ctx.execute("vla:binding", action("get_model_binding", {}))
        if binding.status != "succeeded" or not binding.output["model_binding"]["ready"]:
            ctx.fail("MODEL_NOT_READY", "该 Robot 的 VLA 模型尚未就绪")
            return
        if binding.output["model_binding"]["robot_id"] != state.robot_id:
            ctx.fail("MODEL_ROBOT_MISMATCH", "VLA 与所选 Robot 不一致")
            return
        next_stage = "execute_policy"
    elif state.stage == "execute_policy":
        started = time.monotonic()
        next_capture = started
        next_monitor = started
        handle = await ctx.start_action("vla:policy", action("execute_policy", {
            "robot_id": state.robot_id, "generation": state.generation,
            "instruction": inputs.instruction, "max_actions": inputs.max_actions,
            "timeout_seconds": inputs.timeout_seconds,
        }, timeout=inputs.timeout_seconds + 15))
        try:
            async for feedback in handle.feedback():
                ctx.check_cancelled()
                phase = feedback.measurements.get("phase")
                if phase not in {"control", "inference"}:
                    continue
                # 反馈可能批量到达，不能为每条历史进度执行一次慢速场景查询。
                # 按墙钟采样当前事实，其余进度快速消费；保留 Pilot 原始记录，
                # 并让流及时抵达终态，避免模型已结束而 Skill 仍“执行中”。
                if time.monotonic() < next_monitor:
                    continue
                # 目标成立只记录首次命中，随后允许同一策略完成有限尾段。
                state.monitor_index += 1
                # 持物验收继续逐反馈执行；相机证据按墙钟限频，避免每个动作都
                # 生成图片。复用阶段事件和证据索引，Agent 不参与高频控制循环。
                capture = time.monotonic() >= next_capture
                data = await observe(ctx, inputs, f"vla:monitor:{state.monitor_index}", state.stage, capture)
                next_monitor = time.monotonic() + .1
                if data is None:
                    await handle.stop("VLA 场景观测失败")
                    return
                state.verified = assess(data, inputs, state)
                actions = feedback.measurements.get("executed_actions", 0)
                if state.verified and state.first_success_actions is None:
                    state.first_success_actions = actions
                    state.tail_action_limit = min(actions + POST_SUCCESS_ACTIONS, inputs.max_actions)
                    ctx.checkpoint(state)
                if state.first_success_actions is not None and not state.tail_limit_applied:
                    # 用同一流的执行身份收紧预算，不重启策略、不清空模型缓存，也
                    # 不用旧反馈计数在 Python 中轮询停止而导致多执行整个动作块。
                    policy_id = feedback.measurements.get("policy_invocation_id")
                    if not policy_id:
                        await handle.stop("策略反馈缺少执行身份，无法设置尾段预算")
                        ctx.fail("POLICY_FEEDBACK_INVALID", "策略反馈缺少执行身份，无法设置尾段预算")
                        return
                    limit_input = {
                        "policy_invocation_id": policy_id, "generation": state.generation,
                        "max_actions": state.tail_action_limit,
                    }
                    # 支持相对预算的 Ability 在接收请求时原子读取当前进度。
                    # 历史反馈只用于展示，不能让其游标导致“成功后尾段已过期”。
                    # 未声明该能力的机器人继续沿用已有绝对预算契约。
                    if feedback.measurements.get("supports_relative_action_limit"):
                        limit_input.update(max_actions=inputs.max_actions,
                                           additional_actions=POST_SUCCESS_ACTIONS)
                    limit = await ctx.execute("vla:tail-limit", action("set_execution_limit", limit_input))
                    if limit.status != "succeeded":
                        await handle.stop("尾段预算设置失败")
                        ctx.fail("TAIL_LIMIT_FAILED", limit.error_message or "尾段预算设置失败", limit.evidence_refs)
                        return
                    state.tail_limit_applied = True
                    if limit_input.get("additional_actions") is not None:
                        state.first_success_actions = limit.output["start_actions"]
                        state.tail_action_limit = limit.output["action_limit"]
                    ctx.report("stage.running", stage=state.stage, stage_status="running",
                               summary=f"首次目标达成 · 继续最多 {POST_SUCCESS_ACTIONS} 个模型动作后复核",
                               evidence_refs=data.get("artifact_refs", []))
                if capture:
                    elapsed = time.monotonic() - started
                    next_capture = time.monotonic() + 2.0
                    activity = (f"收尾动作 {max(0, actions - state.first_success_actions)}/{POST_SUCCESS_ACTIONS}"
                                if state.first_success_actions is not None else
                                "等待模型输出" if phase == "inference" else "持续检查目标")
                    ctx.report("stage.running", stage=state.stage, stage_status="running",
                               summary=f"策略执行 {elapsed:.0f} 秒 · 已执行 {actions} 个动作 · {activity}",
                               progress=feedback.measurements.get("progress", 0),
                               evidence_refs=(data or {}).get("artifact_refs", []))
                ctx.checkpoint(state)
            terminal = await handle.result()
            state.executed_actions = (terminal.output or {}).get("executed_actions", 0)
            if terminal.status != "succeeded":
                await observe(ctx, inputs, "vla:failure", state.stage, True)
                ctx.report("stage.failed", stage=state.stage, stage_status="failed",
                           summary=terminal.error_message or "策略未正常完成")
                ctx.fail(terminal.error_code or "POLICY_FAILED",
                         terminal.error_message or "策略未正常完成", ctx.recent_evidence())
                return
        except Exception:
            # 观测失败也必须先停止物理执行，再传播问题，不能留下后台策略继续运动。
            await handle.stop("VLA Skill 中断")
            raise
        next_stage = "verify_result"
    else:
        data = await observe(ctx, inputs, "vla:final", state.stage, True)
        if data is None:
            return
        state.verified = assess(data, inputs, state)
        if not state.verified:
            ctx.report("stage.failed", stage=state.stage, stage_status="failed",
                       summary="策略已结束，但未通过独立目标验收")
            ctx.fail("OBJECTIVE_NOT_MET", "策略已结束，但未通过独立目标验收",
                     ctx.recent_evidence())
            return
        ctx.report("stage.completed", stage=state.stage, stage_status="completed",
                   summary="目标验收通过", evidence_refs=ctx.recent_evidence())
        ctx.complete(ManipulationResult(
            objective=inputs.objective, target_source_id=inputs.target_source_id,
            generation=state.generation, native_task_success=data["evaluation"]["success"],
            evidence_refs=ctx.recent_evidence(),
            target_reached_once=state.first_success_actions is not None or state.verified,
            first_success_actions=state.first_success_actions,
            post_success_actions=max(0, state.executed_actions - state.first_success_actions)
                if state.first_success_actions is not None else 0,
            post_success_complete=(state.first_success_actions is not None and
                state.executed_actions - state.first_success_actions >= POST_SUCCESS_ACTIONS),
        ))
        return
    ctx.report("stage.completed", stage=state.stage, stage_status="completed", summary="阶段完成")
    state.stage = next_stage
    ctx.checkpoint(state)


async def on_stop(ctx: SkillContext, request: StopRequest):
    result = await ctx.execute_stop(f"vla:hold:{request.id}",
                                    action("hold_robot", {"reason": request.reason}))
    safe = result.status == "succeeded" and (result.output or {}).get("safe") is True
    return ctx.stop_outcome(safe=safe, physical_state="hold" if safe else "unknown",
                            summary="VLA 已停止并保持" if safe else "VLA 停止未确认",
                            requires_intervention=not safe, evidence_refs=result.evidence_refs)
