"""Move each arm to the selected down or raised joint posture."""

from semantic_robot_skill_sdk import Action, SkillContext, StopOutcome, StopRequest

from .geometry import prepare_arms
from .models import Input, JointSample, Result, State
from .runtime import ActionFailed, execute


async def run(ctx: SkillContext) -> None:
    inputs = ctx.input(Input)
    state = ctx.load_state(State, default=State())
    try:
        if state.targets and state.arm_posture != inputs.arm_posture:
            raise ValueError("初始化检查点姿态与输入不一致，不能复用另一种姿态的执行结果")
        if not state.targets:
            sample = await execute(
                ctx,
                state,
                "init:read-state",
                Action(
                    type="robot.get_state",
                    parameters={},
                    timeout_seconds=15,
                ),
            )
            observations = [
                obs for obs in sample.observations if obs.kind == "robot.state"
            ]
            if not observations or not isinstance(observations[-1].value, dict):
                raise ValueError("robot.get_state 未返回 robot.state 观测")
            joints = JointSample.model_validate(observations[-1].value.get("joints"))
            state.torso_pitch_rad, state.targets = prepare_arms(joints, inputs.arm_posture)
            state.arm_posture = inputs.arm_posture
            ctx.checkpoint(state)
        for target in state.targets:
            await execute(
                ctx, state, f"init:{target.side}",
                Action(
                    type="motion.move_arm_joint",
                    parameters={
                        "arm_side": target.side,
                        "motion_mode": "ik",
                        "joint_names": target.joint_names,
                        "positions_rad": target.positions_rad,
                    },
                    timeout_seconds=150,
                    label=f"{target.side} arm {state.arm_posture}",
                ),
            )
    except ActionFailed:
        return
    except ValueError as error:
        ctx.fail("INIT_STATE_INVALID", str(error), state.evidence_refs)
        return
    state.stage = "completed"
    ctx.checkpoint(state)
    ctx.complete(
        Result(
            arm_posture=state.arm_posture,
            torso_pitch_rad=state.torso_pitch_rad,
            targets=state.targets,
            evidence_refs=state.evidence_refs,
        )
    )


async def on_stop(ctx: SkillContext, request: StopRequest) -> StopOutcome:
    # Pilot confirms StopExecution for active Actions before entering on_stop.
    return ctx.stop_outcome(
        safe=True,
        physical_state="pilot_confirmed_terminal",
        summary="Pilot 已确认关节动作结束；保留当前姿态",
        evidence_refs=ctx.recent_evidence(),
    )
