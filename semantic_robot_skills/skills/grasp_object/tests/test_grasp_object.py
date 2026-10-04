"""去码垛抓取 Robot Skill 的最小端到端测试。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


# 测试从仓库根运行时，显式加入共享 depalletizing 示例根目录。
EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from semantic_robot_skills.skills.grasp_object.scripts.controller import (
    DepalletizingGraspController,
)
from semantic_robot_skills.skills.grasp_object.scripts.models import (
    GraspCandidate,
    GraspObjectInput,
    GraspObjectResult,
    GraspVerificationValue,
    HeldObjectState,
    TargetHint,
    GraspObjectState,
    ToolLoadObservationValue,
)
from semantic_robot_skills.skills.grasp_object.scripts.skill import (
    CONTROLLER_NAME,
    _allowed_agent_actions,
    _ensure_primary_load,
    _grasp,
    _handle_partial_grasp_failure,
    _lift_and_verify,
    _missing_preflight_contact_tools,
    _recover_approach_or_ask_agent,
    _rank_candidates,
    _reseat_single_missing_hook,
    _refresh_extraction_plan_from_contact,
    _reobserve_after_extraction,
    _request_agent_decision,
    _verification_achieved,
    on_stop,
    run,
)
from semantic_robot_skill_sdk import (
    ActionFeedback,
    ActionResult,
    MockSkillContext,
    Observation,
    Pose3D,
    StopRequest,
    run_skill,
)


OBJECT_REF = "object://pallet/box-17"
TOOL_REFS = ("component://tool/left", "component://tool/right")


def test_auto_preserves_live_environment_candidate_order() -> None:
    extract = _candidate(
        "candidate-extract", x=0.82, score=0.8, strategy="left_extract_first"
    )
    direct = _candidate(
        "candidate-direct", x=0.82, score=1.0, strategy="direct_bilateral"
    )

    assert _rank_candidates([extract, direct], "auto") == [extract, direct]
    assert _rank_candidates([extract, direct], "direct_bilateral")[0] == direct


def test_grasp_verification_uses_pose_observation_tolerance() -> None:
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    candidate = _candidate("candidate-direct", x=0.0, score=1.0)
    verification = GraspVerificationValue(
        candidate_id=candidate.candidate_id,
        held=True,
        stable_bilateral_load=True,
        lift_height_m=0.079911,
        stable_duration_ms=500,
        object_pose=_pose(0.0, 0.0, 0.079911, "scene-r2"),
        tool_poses={
            TOOL_REFS[0]: _pose(-0.3, 0.0, 0.2, "scene-r2"),
            TOOL_REFS[1]: _pose(0.3, 0.0, 0.2, "scene-r2"),
        },
    )
    result = ActionResult(status="succeeded", physical_effect="confirmed")

    assert _verification_achieved(skill_input, candidate, result, verification)
    verification.lift_height_m = 0.075
    assert not _verification_achieved(skill_input, candidate, result, verification)


def test_stable_load_closes_small_lift_tracking_gap() -> None:
    """稳定承载时按真实位移补足高度，不放宽物理完成标准。"""

    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    candidate = _candidate("candidate-lift", x=0.82, score=1.0)
    state = GraspObjectState(
        stage="lift_and_verify",
        active_strategy="direct_bilateral",
        target_revision="scene-7",
        target_pose=_pose(0.82, 0.0, 0.20, "scene-7"),
        target_extent_m=(0.42, 0.30, 0.24),
        candidates=[candidate],
        selected_candidate_id=candidate.candidate_id,
        object_held=True,
    )
    context = _context(skill_input)
    controller = DepalletizingGraspController()

    def queue_lift_and_verification(*, observed_height: float, held: bool) -> None:
        progress = _observation(
            "lift_progress",
            {
                "candidate_id": candidate.candidate_id,
                "lift_height_m": observed_height,
                "object_follows_tools": True,
                "stable_load": True,
                "slip_detected": False,
                "overloaded": False,
            },
        )
        context.queue_action(
            "motion.lift_held_object",
            ActionResult(status="succeeded", physical_effect="confirmed"),
            feedback=[
                ActionFeedback(
                    sequence=1,
                    status="lifting",
                    progress=1.0,
                    observations=[progress],
                )
            ],
        )
        context.queue_action(
            "perception.verify_grasp",
            ActionResult(
                status="succeeded",
                physical_effect="none",
                observations=[
                    _observation(
                        "grasp_verification",
                        {
                            "candidate_id": candidate.candidate_id,
                            "held": held,
                            "stable_bilateral_load": True,
                            "lift_height_m": observed_height,
                            "stable_duration_ms": 600,
                            "slipping": False,
                            "overloaded": False,
                            "object_pose": _pose(
                                0.82, 0.0, 0.20 + observed_height, "scene-8"
                            ).model_dump(mode="json"),
                            "tool_poses": {
                                ref: target.target_pose.model_dump(mode="json")
                                for ref, target in zip(
                                    TOOL_REFS,
                                    candidate.hook_insert_poses,
                                    strict=True,
                                )
                            },
                        },
                        revision="scene-8",
                    )
                ],
            ),
        )

    queue_lift_and_verification(observed_height=0.07698, held=False)
    asyncio.run(_lift_and_verify(context, controller, skill_input, state))
    first_lift = next(
        event
        for event in context.events
        if event["type"] == "action_started"
        and event["action"] == "motion.lift_held_object"
    )
    assert first_lift["parameters"]["distance_m"] == pytest.approx(0.04)

    assert state.stage == "lift_and_verify"
    assert not state.lift_completed
    assert state.object_held
    assert state.pending_lift_distance_m == pytest.approx(0.00402)

    queue_lift_and_verification(observed_height=0.0804, held=True)
    asyncio.run(_lift_and_verify(context, controller, skill_input, state))

    assert state.stage == "prepare_transport"
    assert state.lift_completed
    assert state.pending_lift_distance_m is None
    lift_actions = [
        event
        for event in context.events
        if event["type"] == "action_started"
        and event["action"] == "motion.lift_held_object"
    ]
    assert lift_actions[-1]["parameters"]["distance_m"] == pytest.approx(0.00402)


def test_preaction_lift_failure_keeps_verified_bilateral_load() -> None:
    """规划未下发物理命令时，不得清空前一Stage确认的双侧承载。"""

    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    candidate = _candidate("candidate-lift-plan-failed", x=0.82, score=1.0)
    state = GraspObjectState(
        stage="lift_and_verify",
        active_strategy="direct_bilateral",
        target_revision="scene-7",
        target_pose=_pose(0.82, 0.0, 0.20, "scene-7"),
        target_extent_m=(0.42, 0.30, 0.24),
        candidates=[candidate],
        selected_candidate_id=candidate.candidate_id,
        object_held=True,
    )
    context = _context(skill_input)
    context.queue_action(
        "motion.lift_held_object",
        ActionResult(status="failed", physical_effect="none"),
    )
    context.queue_agent_reply(
        {
            "expected_plan_revision": state.plan_revision,
            "action": "retry_verification",
            "reason": "抬升尚未下发，保持当前双侧承载并等待重试",
        }
    )

    asyncio.run(
        _lift_and_verify(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
        )
    )

    assert state.object_held is True
    assert state.stage == "lift_and_verify"
    assert state.lift_completed is False
    assert state.verification_only is True


@pytest.mark.parametrize("reason", ["hook_contact_missing", "hook_force_low", "clamp_force_low"])
def test_lift_preflight_load_loss_reseats_current_candidate_once(reason: str) -> None:
    """抖升尚未下发时丢失一侧接触，不能从贴箱位置重放接近路径。"""

    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    candidate = _candidate("candidate-lift-load-lost", x=0.82, score=1.0)
    state = GraspObjectState(
        stage="lift_and_verify",
        active_strategy="direct_bilateral",
        target_revision="scene-7",
        target_pose=_pose(0.82, 0.0, 0.20, "scene-7"),
        target_extent_m=(0.42, 0.30, 0.24),
        candidates=[candidate],
        selected_candidate_id=candidate.candidate_id,
        object_held=True,
        grasp_attempts=1,
    )
    context = _context(skill_input)
    context.queue_action(
        "motion.lift_held_object",
        ActionResult(
            status="failed",
            physical_effect="none",
            error_code="LOAD_NOT_STABLE",
            error_message=f"双侧夹具接触、受力或传感状态不完整，禁止开始抬升: component://tool/right:{reason}",
        ),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "gripper.close",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, list(TOOL_REFS))

    asyncio.run(
        _lift_and_verify(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
        )
    )

    assert state.stage == "lift_and_verify"
    assert state.object_held is True
    assert state.grasp_attempts == 2
    assert state.plan_revision == 1
    assert not any(event["type"] == "agent_requested" for event in context.events)
    started = [event for event in context.events if event["type"] == "action_started"]
    reseat = next(
        event
        for event in started
        if event["key"].endswith(":lift-preflight-reseat")
        and event["action"] == "motion.move_end_effector"
    )
    assert [item["tool_ref"] for item in reseat["parameters"]["targets"]] == [
        TOOL_REFS[1]
    ]
    assert reseat["parameters"]["required_contact_tools"] == [TOOL_REFS[0]]
    reclose = next(
        event
        for event in started
        if event["key"].endswith(":lift-preflight-reseat")
        and event["action"] == "gripper.close"
    )
    assert [item["tool_ref"] for item in reclose["parameters"]["tools"]] == [
        TOOL_REFS[1]
    ]


def test_lift_preflight_reseats_with_real_multi_reason_message() -> None:
    """LiftHeldObject 真实输出里一侧会同时缺接触、受力偏低，两条原因用 ", " 拼接。

    只按 "; " 拆分会把整串当成一个 token，白名单校验失败后单侧重入位永不触发，
    抓取直接升级给 Agent 并中止（BUG-002 的现场形状）。
    """

    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    candidate = _candidate("candidate-lift-multi-reason", x=0.82, score=1.0)
    state = GraspObjectState(
        stage="lift_and_verify",
        active_strategy="direct_bilateral",
        target_revision="scene-7",
        target_pose=_pose(0.82, 0.0, 0.20, "scene-7"),
        target_extent_m=(0.42, 0.30, 0.24),
        candidates=[candidate],
        selected_candidate_id=candidate.candidate_id,
        object_held=True,
        grasp_attempts=1,
    )
    context = _context(skill_input)
    context.queue_action(
        "motion.lift_held_object",
        ActionResult(
            status="failed",
            physical_effect="none",
            error_code="LOAD_NOT_STABLE",
            error_message=(
                "双侧夹具接触、受力或传感状态不完整，禁止开始抬升: "
                "component://tool/right:hook_contact_missing, "
                "component://tool/right:hook_force_low"
            ),
        ),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "gripper.close",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, list(TOOL_REFS))

    asyncio.run(
        _lift_and_verify(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
        )
    )

    assert not any(event["type"] == "agent_requested" for event in context.events)
    assert state.grasp_attempts == 2
    started = [event for event in context.events if event["type"] == "action_started"]
    reseat = next(
        event
        for event in started
        if event["key"].endswith(":lift-preflight-reseat")
        and event["action"] == "motion.move_end_effector"
    )
    assert [item["tool_ref"] for item in reseat["parameters"]["targets"]] == [
        TOOL_REFS[1]
    ]


def test_missing_preflight_contact_tools_rejects_unknown_reason() -> None:
    """不可恢复的原因必须保持整体拒绝，不能因为解析变宽就盲目补座。"""

    assert _missing_preflight_contact_tools(
        ActionResult(
            status="failed",
            error_code="LOAD_NOT_STABLE",
            error_message=(
                "双侧夹具接触、受力或传感状态不完整，禁止开始抬升: "
                "component://tool/right:sensor_fault"
            ),
        ),
        list(TOOL_REFS),
    ) == []
    assert _missing_preflight_contact_tools(
        ActionResult(status="failed", error_code="LOAD_NOT_STABLE", error_message=""),
        list(TOOL_REFS),
    ) == []


def _pose(x: float, y: float, z: float, revision: str) -> Pose3D:
    """构造带场景版本的测试位姿。"""

    return Pose3D(
        frame_id="map",
        position_m=(x, y, z),
        revision=revision,
    )


def _candidate(
    candidate_id: str,
    *,
    x: float,
    score: float,
    strategy: str = "direct_bilateral",
) -> GraspCandidate:
    """构造明确携带左右工具目标的 schema v2 候选。"""

    left_approach = _pose(x, 0.25, 0.24, "scene-7")
    left_transfer = _pose(x, 0.25, 0.44, "scene-7")
    right_transfer = _pose(x, -0.25, 0.44, "scene-7")
    right_approach = _pose(x, -0.25, 0.24, "scene-7")
    left_insert = _pose(x, 0.25, 0.12, "scene-7")
    right_insert = _pose(x, -0.25, 0.12, "scene-7")
    left_seat = _pose(x, 0.25, 0.14, "scene-7")
    right_seat = _pose(x, -0.25, 0.14, "scene-7")
    return GraspCandidate.model_validate(
        {
            "candidate_id": candidate_id,
            "object_ref": OBJECT_REF,
            "observation_revision": "scene-7",
            "planned_object_pose": _pose(x, 0.0, 0.25, "scene-7"),
            "strategy": strategy,
            "tool_targets": [
                {"tool_ref": TOOL_REFS[0], "side": "left"},
                {"tool_ref": TOOL_REFS[1], "side": "right"},
            ],
            "clearance_poses": [
                {
                    "tool_ref": TOOL_REFS[0],
                    "target_pose": _pose(x - 0.30, 0.15, 0.44, "scene-7"),
                },
                {
                    "tool_ref": TOOL_REFS[1],
                    "target_pose": _pose(x - 0.30, -0.15, 0.44, "scene-7"),
                },
            ],
            "transfer_poses": [
                {"tool_ref": TOOL_REFS[0], "target_pose": left_transfer},
                {"tool_ref": TOOL_REFS[1], "target_pose": right_transfer},
            ],
            "alignment_poses": [
                {"tool_ref": TOOL_REFS[0], "target_pose": left_transfer},
                {"tool_ref": TOOL_REFS[1], "target_pose": right_transfer},
            ],
            "pregrasp_poses": [
                {"tool_ref": TOOL_REFS[0], "target_pose": left_transfer},
                {"tool_ref": TOOL_REFS[1], "target_pose": right_transfer},
            ],
            "approach_poses": [
                {"tool_ref": TOOL_REFS[0], "target_pose": left_approach},
                {"tool_ref": TOOL_REFS[1], "target_pose": right_approach},
            ],
            "hook_insert_poses": [
                {"tool_ref": TOOL_REFS[0], "target_pose": left_insert},
                {"tool_ref": TOOL_REFS[1], "target_pose": right_insert},
            ],
            "hook_seat_poses": [
                {"tool_ref": TOOL_REFS[0], "target_pose": left_seat},
                {"tool_ref": TOOL_REFS[1], "target_pose": right_seat},
            ],
            "opening_setpoints": [
                {
                    "tool_ref": TOOL_REFS[0],
                    "target_position_m": 0.035,
                    "maximum_force_n": 45.0,
                },
                {
                    "tool_ref": TOOL_REFS[1],
                    "target_position_m": 0.035,
                    "maximum_force_n": 45.0,
                },
            ],
            "clamp_setpoints": [
                {
                    "tool_ref": TOOL_REFS[0],
                    "target_position_m": 0.015,
                    "maximum_force_n": 45.0,
                },
                {
                    "tool_ref": TOOL_REFS[1],
                    "target_position_m": 0.015,
                    "maximum_force_n": 45.0,
                },
            ],
            "pull_path": [],
            "clearance_m": 0.05,
            "score": score,
        }
    )


def _observation(
    kind: str,
    value: dict,
    *,
    revision: str = "scene-7",
    evidence: str | None = None,
) -> Observation:
    """构造带稳定版本和证据引用的业务观测。"""

    return Observation(
        kind=kind,
        subject_ref=OBJECT_REF,
        source="mock://pilot/demo",
        revision=revision,
        frame_id="map",
        confidence=0.95,
        value=value,
        evidence_refs=[evidence] if evidence else [],
    )


def _context(skill_input: GraspObjectInput | None = None) -> MockSkillContext:
    """建立注册了抓取 Controller 的测试 Runtime。"""

    context = MockSkillContext(
        skill_input or GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF)),
    )
    context.register_controller(CONTROLLER_NAME, DepalletizingGraspController())
    return context


def _queue_tool_load(
    context: MockSkillContext,
    tool_refs: tuple[str, ...] | list[str],
    *,
    condition_satisfied: bool = True,
    hook_contact: bool = True,
    clamp_contact: bool = True,
) -> None:
    """准备seat后由RobotState Ability返回的真实时间承载Observation。"""

    context.queue_action(
        "robot.verify_tool_load",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "robot.tool_load",
                    {
                        "condition_satisfied": condition_satisfied,
                        "observed_duration_ms": 160,
                        "tools": [
                            {
                                "tool_ref": ref,
                                "available": True,
                                "position": 0.015,
                                "velocity": 0.0,
                                "effort": 18.0,
                                "hook_contact": hook_contact,
                                "clamp_contact": clamp_contact,
                                "hook_force_n": 12.0,
                                "clamp_force_n": 18.0,
                                "hook_support_ratio": 0.8,
                                "relative_tangential_speed_mps": 0.0,
                                "sensor_fault": False,
                            }
                            for ref in tool_refs
                        ],
                        "slip_detected": False,
                        "overload_detected": False,
                        "sensor_fault": False,
                        "reasons": [],
                    },
                    revision="scene-9",
                )
            ],
        ),
    )


def _queue_reengagement_robot_states(context: MockSkillContext):
    """为第二侧四段运动准备逐段更新的首侧末端位姿。"""

    poses = [
        _pose(0.701 + index * 0.001, 0.111, 0.333, f"robot-state-{8 + index}")
        for index in range(4)
    ]
    for index, pose in enumerate(poses):
        context.queue_action(
            "robot.get_state",
            ActionResult(
                status="succeeded",
                physical_effect="none",
                observations=[
                    _observation(
                        "robot.state",
                        {
                            "robot_id": "robot://demo-1",
                            "generation": 8 + index,
                            "end_effectors": {
                                "left": {
                                    "frame_id": pose.frame_id,
                                    "position": list(pose.position_m),
                                    "quaternion_xyzw": list(pose.orientation_xyzw),
                                }
                            },
                            "tool_states": {
                                TOOL_REFS[0]: {
                                    "tool_ref": TOOL_REFS[0],
                                    "side": "left",
                                    "kind": "tote_clamp",
                                }
                            },
                        },
                        revision=f"robot-state-{8 + index}",
                    )
                ],
            ),
        )
    return poses


def _queue_observation_and_candidates(
    context: MockSkillContext,
    candidates: list[GraspCandidate],
    *,
    pose: Pose3D | None = None,
    revision: str = "scene-7",
) -> None:
    """准备目标观测和候选生成结果。"""

    target_pose = pose or _pose(0.82, 0.27, 0.12, revision)
    context.queue_action(
        "perception.locate_object",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "target_pose",
                    {
                        "object_ref": OBJECT_REF,
                        "pose": target_pose.model_dump(mode="json"),
                        "extent_m": [0.42, 0.30, 0.24],
                        "identity_confidence": 0.96,
                    },
                    evidence="artifact://target-image",
                )
            ],
        ),
    )
    context.queue_action(
        "grasp.generate_candidates",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "grasp_candidates",
                    {
                        "target_revision": revision,
                        "candidates": [
                            item.model_dump(mode="json") for item in candidates
                        ],
                    },
                )
            ],
        ),
    )


def _queue_successful_approach(
    context: MockSkillContext,
    candidate: GraspCandidate,
) -> None:
    """准备 Approach 短计划的成功结果。"""

    context.queue_action(
        "gripper.set_opening",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    for _ in range(4):
        context.queue_action(
            "motion.move_end_effector",
            ActionResult(status="succeeded", physical_effect="confirmed"),
        )
    context.queue_action(
        "perception.verify_pregrasp",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "pregrasp_state",
                    {
                        "candidate_id": candidate.candidate_id,
                        "target_revision": "scene-7",
                        "reached": True,
                        "position_errors_m": {TOOL_REFS[0]: 0.008, TOOL_REFS[1]: 0.009},
                        "object_relative_errors_m": {
                            TOOL_REFS[0]: 0.0,
                            TOOL_REFS[1]: 0.0,
                        },
                        "hook_contacts": {TOOL_REFS[0]: True, TOOL_REFS[1]: True},
                    },
                )
            ],
        ),
    )


def _queue_successful_grasp_and_verification(
    context: MockSkillContext,
    candidate: GraspCandidate,
) -> None:
    """准备双侧夹持、抬升反馈和独立双工具验证。"""

    # direct_bilateral 会由 EndEffector Ability 在一次 Execution 中聚合两条
    # Runtime 单工具命令；因此 Skill 收到的是一个双侧终态，而不是两个先后到达
    # 的单侧终态。Runtime 并未因此获得"原子双夹具命令"能力。
    contact = _observation(
        "grasp_contact",
        {
            "candidate_id": candidate.candidate_id,
            "object_ref": OBJECT_REF,
            "tools": {
                ref: {
                    "hook_contact": True,
                    "clamp_contact": True,
                    "slipping": False,
                    "overloaded": False,
                    "sensor_fault": False,
                }
                for ref in TOOL_REFS
            },
            "contact_confirmed": True,
            "stable_bilateral_load": True,
            "slipping": False,
            "overloaded": False,
        },
        evidence="artifact://contact-bilateral",
    )
    context.queue_action(
        "gripper.close",
        ActionResult(
            status="succeeded", physical_effect="confirmed", observations=[contact]
        ),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, TOOL_REFS)
    lift_progress = _observation(
        "lift_progress",
        {
            "candidate_id": candidate.candidate_id,
            "lift_height_m": 0.10,
            "object_follows_tools": True,
            "stable_load": True,
            "slip_detected": False,
            "overloaded": False,
        },
        evidence="artifact://lift-frame",
    )
    context.queue_action(
        "motion.lift_held_object",
        ActionResult(status="succeeded", physical_effect="confirmed"),
        feedback=[
            ActionFeedback(
                sequence=1, status="lifting", progress=1.0, observations=[lift_progress]
            )
        ],
    )
    object_pose = _pose(0.82, 0.0, 0.22, "scene-8")
    context.queue_action(
        "perception.verify_grasp",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "grasp_verification",
                    {
                        "candidate_id": candidate.candidate_id,
                        "held": True,
                        "stable_bilateral_load": True,
                        "lift_height_m": 0.10,
                        "stable_duration_ms": 700,
                        "slipping": False,
                        "overloaded": False,
                        "object_pose": object_pose.model_dump(mode="json"),
                        "tool_poses": {
                            ref: target.target_pose.model_dump(mode="json")
                            for ref, target in zip(
                                TOOL_REFS, candidate.hook_insert_poses, strict=True
                            )
                        },
                    },
                    revision="scene-8",
                    evidence="artifact://verification-image",
                )
            ],
        ),
    )
    transport_poses = [
        {
            "tool_ref": TOOL_REFS[0],
            "target_pose": _pose(0.82, 0.18, 0.42, "scene-8").model_dump(mode="json"),
        },
        {
            "tool_ref": TOOL_REFS[1],
            "target_pose": _pose(0.82, -0.18, 0.42, "scene-8").model_dump(mode="json"),
        },
    ]
    clearance_poses = [
        {
            "tool_ref": item["tool_ref"],
            "target_pose": {
                **item["target_pose"],
                "position_m": [
                    item["target_pose"]["position_m"][0] - 0.04,
                    item["target_pose"]["position_m"][1],
                    item["target_pose"]["position_m"][2],
                ],
            },
        }
        for item in transport_poses
    ]
    context.queue_action(
        "grasp.plan_transport_posture",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "transport_posture",
                    {
                        "object_ref": OBJECT_REF,
                        "target_revision": "scene-8",
                        "desired_object_pose": _pose(
                            0.82, 0.0, 0.42, "scene-8"
                        ).model_dump(mode="json"),
                        "clearance_poses": clearance_poses,
                        "transport_poses": transport_poses,
                    },
                    revision="scene-8",
                )
            ],
        ),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    tool_poses = {
        item["tool_ref"]: Pose3D.model_validate(item["target_pose"])
        for item in transport_poses
    }
    context.queue_action(
        "robot.get_state",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "robot.state",
                    {
                        "robot_id": "robot://demo-1",
                        "generation": 9,
                        "end_effectors": {
                            "left": {
                                "frame_id": tool_poses[TOOL_REFS[0]].frame_id,
                                "position": list(tool_poses[TOOL_REFS[0]].position_m),
                                "quaternion_xyzw": list(
                                    tool_poses[TOOL_REFS[0]].orientation_xyzw
                                ),
                            },
                            "right": {
                                "frame_id": tool_poses[TOOL_REFS[1]].frame_id,
                                "position": list(tool_poses[TOOL_REFS[1]].position_m),
                                "quaternion_xyzw": list(
                                    tool_poses[TOOL_REFS[1]].orientation_xyzw
                                ),
                            },
                        },
                        "tool_states": {
                            TOOL_REFS[0]: {
                                "tool_ref": TOOL_REFS[0],
                                "side": "left",
                                "kind": "tote_clamp",
                            },
                            TOOL_REFS[1]: {
                                "tool_ref": TOOL_REFS[1],
                                "side": "right",
                                "kind": "tote_clamp",
                            },
                        },
                    },
                    revision="scene-9",
                )
            ],
        ),
    )
    _queue_tool_load(context, TOOL_REFS)
    context.queue_action(
        "perception.locate_object",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "target_pose",
                    {
                        "object_ref": OBJECT_REF,
                        "pose": _pose(0.82, 0.0, 0.22, "scene-9").model_dump(
                            mode="json"
                        ),
                        "extent_m": [0.42, 0.30, 0.24],
                        "identity_confidence": 0.98,
                    },
                    revision="scene-9",
                )
            ],
        ),
    )


@pytest.mark.parametrize("capture_status", [None, "succeeded", "failed"])
def test_happy_path_outputs_grasp_local_held_object_state(capture_status) -> None:
    """完整执行后必须输出 Skill 业务 HeldObjectState，而非只看 Action 成功。"""

    candidate = _candidate("candidate-a", x=0.82, score=0.91)
    context = _context()
    frame_refs = [f"pilot-artifact://pilot-grasp/frame-{index}" for index in range(6)]
    if capture_status:
        context.execution_id = "execution-grasp-rgb"
        for ref in frame_refs:
            context.queue_action("sensor.capture_rgbd", ActionResult(
                status=capture_status, physical_effect="none",
                observations=[Observation(
                    kind="sensor.frame", source="robot-sdk://demo-1/sensor/camera.rgb",
                    value={"sensor_id": "camera.rgb", "media_type": "image/png"},
                    evidence_refs=[ref],
                )] if capture_status == "succeeded" else [],
                error_code="SENSOR_UNAVAILABLE" if capture_status == "failed" else None,
            ))
    _queue_observation_and_candidates(context, [candidate])
    _queue_successful_approach(context, candidate)
    _queue_successful_grasp_and_verification(context, candidate)

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    assert isinstance(context.result, GraspObjectResult)
    assert isinstance(context.result.held_object, HeldObjectState)
    assert context.result.held_object.object_ref == OBJECT_REF
    assert context.result.held_object.grasp_candidate_id == "candidate-a"
    assert context.result.held_object.robot_state_revision == "scene-9"
    assert set(context.result.held_object.tool_poses) == set(TOOL_REFS)
    assert "artifact://verification-image" in context.result.held_object.evidence_refs
    if capture_status:
        captures = [event for event in context.events if event.get("action") == "sensor.capture_rgbd"]
        assert len(captures) == len({event["key"] for event in captures}) == 6
        result_refs = set(context.result.held_object.evidence_refs)
        if capture_status == "succeeded":
            assert set(frame_refs) <= result_refs
        else:
            assert not set(frame_refs) & result_refs
    action_types = [
        event["action"] for event in context.events
        if event["type"] == "action_started" and event["action"] != "sensor.capture_rgbd"
    ]
    assert action_types[-6:] == [
        "grasp.plan_transport_posture",
        "motion.move_end_effector",
        "motion.move_end_effector",
        "robot.get_state",
        "robot.verify_tool_load",
        "perception.locate_object",
    ]


def test_extract_approach_keeps_secondary_before_primary_contact() -> None:
    """单侧接触前只移动首侧，另一侧保持当前安全姿态。"""

    candidate = _candidate(
        "candidate-left-extract",
        x=0.82,
        score=0.92,
        strategy="left_extract_first",
    )
    controller = DepalletizingGraspController()
    plan = controller.plan_approach(
        GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF)),
        candidate=candidate,
        target_revision="scene-7",
    )

    assert [action.parameters.get("purpose") for action in plan[1:5]] == [
        "clearance",
        "transfer",
        "pregrasp",
        "insert",
    ]
    for action in plan[1:5]:
        assert [target["tool_ref"] for target in action.parameters["targets"]] == [
            TOOL_REFS[0]
        ]
    assert sum(action.parameters.get("purpose") == "clearance" for action in plan) == 1


def test_approach_uses_observed_candidate_and_recovers_to_next_candidate() -> None:
    """首个候选不可达时应局部切换，且短计划使用动态观测位姿。"""

    first = _candidate("candidate-a", x=0.82, score=0.92)
    second = _candidate("candidate-b", x=0.91, score=0.83, strategy="direct_bilateral")
    controller = DepalletizingGraspController()
    plan = controller.plan_approach(
        GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF)),
        candidate=second,
        target_revision="scene-7",
    )
    assert plan[0].type == "gripper.set_opening"
    assert plan[0].parameters["tools"][0]["target_position_m"] == 0.035
    assert plan[1].parameters["targets"][0]["target_pose"][
        "position_m"
    ] == pytest.approx(
        [
            0.61,
            0.15,
            0.44,
        ]
    )
    assert plan[1].parameters["purpose"] == "clearance"
    assert plan[2].parameters["targets"][0]["target_pose"]["position_m"] == [
        0.91,
        0.25,
        0.44,
    ]
    assert plan[2].parameters["targets"][1]["target_pose"]["position_m"] == [
        0.91,
        -0.25,
        0.44,
    ]
    assert plan[2].parameters["purpose"] == "transfer"
    assert plan[3].parameters["targets"][0]["target_pose"]["position_m"] == [
        0.91,
        0.25,
        0.24,
    ]
    assert plan[4].parameters["targets"][0]["target_pose"]["position_m"] == [
        0.91,
        0.25,
        0.12,
    ]
    assert plan[4].parameters["purpose"] == "insert"
    assert plan[4].parameters["max_contact_force_n"] == 60.0
    assert plan[5].parameters["maximum_position_error_m"] == 0.015
    assert plan[2].parameters["targets"][0]["target_pose"]["position_m"] != [
        0.55,
        0.15,
        0.1,
    ]

    context = _context()
    _queue_observation_and_candidates(context, [first, second])
    context.queue_action(
        "gripper.set_opening",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(
            status="failed",
            physical_effect="none",
            error_code="CANDIDATE_UNREACHABLE",
        ),
    )
    _queue_successful_approach(context, second)
    _queue_successful_grasp_and_verification(context, second)

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    assert context.result.held_object.grasp_candidate_id == "candidate-b"
    assert any(
        event["type"] == "stage.recovering" and "candidate-b" in event["summary"]
        for event in context.events
    )


def test_approach_with_possible_physical_effect_does_not_switch_candidate() -> None:
    """插入失败但可能已接触箱沿时，必须保持现场并请求 Agent。"""

    first = _candidate("candidate-a", x=0.82, score=0.92)
    second = _candidate("candidate-b", x=0.91, score=0.83)
    context = _context()
    _queue_observation_and_candidates(context, [first, second])
    context.queue_action(
        "gripper.set_opening",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(
            status="failed",
            physical_effect="possible",
            error_code="CONTACT_LOST",
        ),
    )
    context.queue_agent_reply(
        {
            "expected_plan_revision": 2,
            "action": "abort_subtask",
            "reason": "保留 hold，等待重新检查箱沿接触",
        }
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "failed"
    requests = [event for event in context.events if event["type"] == "agent_requested"]
    assert len(requests) == 1
    assert requests[0]["context"]["selected_candidate_id"] == "candidate-a"
    started_moves = [
        event
        for event in context.events
        if event["type"] == "action_started"
        and event["action"] == "motion.move_end_effector"
    ]
    assert len(started_moves) == 1


def test_primary_contact_recovery_requires_new_physical_progress() -> None:
    state = GraspObjectState(primary_clamped=True)
    assert _allowed_agent_actions(
        state,
        {
            "failure_phase": "initial_extraction",
            "physical_effect": "possible",
            "error_code": "REQUIRED_TOOL_CONTACT_LOST",
        },
    ) == ["abort_subtask"]

    assert _allowed_agent_actions(state, {"failure_phase": "secondary_approach"}) == [
        "restart_observation",
        "abort_subtask",
    ]

    state.secondary_replan_attempted = True
    assert _allowed_agent_actions(state, {"failure_phase": "secondary_approach"}) == [
        "abort_subtask"
    ]
    assert _allowed_agent_actions(state, {"failure_phase": "primary_load_proof"}) == [
        "abort_subtask"
    ]
    assert _allowed_agent_actions(
        state,
        {
            "failure_phase": "extraction_adjustment",
            "physical_effect": "none",
            "error_code": "PLANNING_FAILED",
        },
    ) == ["abort_subtask"]


def test_agent_abort_after_extraction_confirms_hold_before_failure() -> None:
    """首侧已外拉时，Agent终止必须先取得真实hold证据。"""

    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    state = GraspObjectState(
        stage="grasp",
        primary_clamped=True,
        extraction_primary_side="right",
        plan_revision=3,
        decision_revision=1,
    )
    context = _context(skill_input)
    context.queue_action(
        "gripper.hold_object",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_agent_reply(
        {
            "expected_plan_revision": 3,
            "action": "abort_subtask",
            "reason": "第二侧无法安全接入，保持当前现场",
        }
    )

    asyncio.run(
        _request_agent_decision(
            context,
            skill_input,
            state,
            reason="第二侧插入规划失败",
            details={"failure_phase": "secondary_approach"},
        )
    )

    assert context.status == "failed"
    assert context.failure is not None
    assert context.failure.code == "AGENT_ABORTED_GRASP"
    started = [event for event in context.events if event["type"] == "action_started"]
    assert [event["action"] for event in started] == ["gripper.hold_object"]


def test_precontact_planning_failure_exposes_error_and_rejects_blind_restart() -> None:
    candidate = _candidate("candidate-a", x=0.82, score=0.92)
    context = _context()
    _queue_observation_and_candidates(context, [candidate])
    context.queue_action(
        "gripper.set_opening",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(
            status="failed",
            physical_effect="none",
            error_code="PLANNING_FAILED",
            error_message="torso_link3_0 与目标箱体碰撞",
        ),
    )
    context.queue_agent_reply(
        {
            "expected_plan_revision": 2,
            "action": "restart_observation",
            "reason": "没有新证据但再次重观测",
        }
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "failed"
    assert context.failure.code == "INVALID_AGENT_DECISION"
    requests = [event for event in context.events if event["type"] == "agent_requested"]
    assert len(requests) == 1
    request_context = requests[0]["context"]
    assert request_context["details"]["error_code"] == "PLANNING_FAILED"
    assert "torso_link3_0" in request_context["details"]["error_message"]
    assert "restart_observation" not in request_context["allowed_actions"]
    assert request_context["failed_precontact_strategies"] == ["direct_bilateral"]
    assert request_context["available_strategies"] == [
        "left_extract_first",
        "right_extract_first",
    ]


def test_precontact_planning_failure_stops_after_all_strategies_failed() -> None:
    state = GraspObjectState(
        failed_precontact_strategies=[
            "direct_bilateral",
            "left_extract_first",
            "right_extract_first",
        ]
    )
    assert _allowed_agent_actions(
        state,
        {
            "physical_effect": "none",
            "error_code": "PLANNING_FAILED",
        },
    ) == ["abort_subtask"]


def test_each_failed_candidate_is_remembered_before_switching() -> None:
    strategies = ["direct_bilateral", "left_extract_first", "right_extract_first"]
    candidates = [
        _candidate(f"candidate-{index}", x=0.82, score=0.9, strategy=strategy)
        for index, strategy in enumerate(strategies)
    ]
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    state = GraspObjectState(candidates=candidates, selected_candidate_id=candidates[0].candidate_id)
    context = _context(skill_input)
    details = {"physical_effect": "none", "error_code": "PLANNING_FAILED"}
    for index in range(3):
        if index == 2:
            context.queue_agent_reply({"expected_plan_revision": state.plan_revision,
                "action": "abort_subtask", "reason": "候选均已失败"})
        asyncio.run(_recover_approach_or_ask_agent(
            context, skill_input, state, reason="起点碰撞", details=details))
        assert state.failed_precontact_strategies == strategies[:index + 1]
    requests = [event for event in context.events if event["type"] == "agent_requested"]
    assert len(requests) == 1
    assert requests[0]["context"]["allowed_actions"] == ["abort_subtask"]
    assert requests[0]["context"]["available_strategies"] == []
    assert context.status == "failed"


def test_preflight_force_recovery_does_not_mask_sensor_fault() -> None:
    result = ActionResult(status="failed", error_code="LOAD_NOT_STABLE",
        error_message=f"{TOOL_REFS[0]}:hook_force_low; {TOOL_REFS[1]}:sensor_fault")
    assert _missing_preflight_contact_tools(result, TOOL_REFS) == []


def test_failed_load_verification_cannot_restart_from_contact() -> None:
    state = GraspObjectState(stage="lift_and_verify", object_held=False)
    details = {"failure_phase": "lift_verification"}
    assert _allowed_agent_actions(state, details) == ["abort_subtask"]
    context = _context()
    context.queue_action("gripper.hold_object",
        ActionResult(status="succeeded", physical_effect="confirmed"))
    context.queue_agent_reply({"expected_plan_revision": state.plan_revision,
        "action": "abort_subtask", "reason": "承载未恢复，保持现场"})
    asyncio.run(_request_agent_decision(context,
        GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF)), state,
        reason="承载验证失败", details=details))
    assert context.status == "failed"
    assert [event["action"] for event in context.events
            if event["type"] == "action_started"] == ["gripper.hold_object"]


def test_agent_verification_retry_preserves_budget_and_actual_lift_state() -> None:
    state = GraspObjectState(stage="lift_and_verify", object_held=True,
        lift_completed=False, verification_attempts=3)
    context = _context()
    context.queue_agent_reply({"expected_plan_revision": state.plan_revision,
        "action": "retry_verification", "reason": "只复核一次"})
    asyncio.run(_request_agent_decision(context,
        GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF)), state,
        reason="复核承载", details={}))
    assert state.verification_attempts == 3
    assert not state.lift_completed
    assert state.verification_only
    state.verification_attempts = 4
    assert _allowed_agent_actions(state, {}) == ["abort_subtask"]


def test_failed_lift_reverification_holds_and_exits_without_reapproaching() -> None:
    """回放本次低钩力链：决策只复核，未承载就保持现场结束，不重放接近。"""
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    candidate = _candidate("direct_bilateral-1", x=0.82, score=1.0)
    state = GraspObjectState(
        stage="lift_and_verify", target_revision="scene-7",
        target_pose=_pose(0.82, 0.0, 0.20, "scene-7"),
        target_extent_m=(0.42, 0.30, 0.24), candidates=[candidate],
        selected_candidate_id=candidate.candidate_id, object_held=True,
        grasp_attempts=2,
    )
    context = _context(skill_input)
    context.queue_action("motion.lift_held_object", ActionResult(
        status="failed", physical_effect="none", error_code="LOAD_NOT_STABLE",
        error_message=f"{TOOL_REFS[0]}:hook_force_low"))
    context.queue_agent_reply({"expected_plan_revision": 0,
        "action": "retry_verification", "reason": "复核当前承载"})
    asyncio.run(_lift_and_verify(context, DepalletizingGraspController(), skill_input, state))
    assert not state.lift_completed
    assert state.verification_only
    context.queue_action("perception.verify_grasp", ActionResult(
        status="succeeded", physical_effect="none", observations=[_observation(
            "grasp_verification", {
                "candidate_id": candidate.candidate_id, "held": False,
                "stable_bilateral_load": False, "lift_height_m": -0.00036069,
                "stable_duration_ms": 0,
                "object_pose": state.target_pose.model_dump(mode="json"),
                "tool_poses": {},
            })]))
    context.queue_action("gripper.hold_object",
        ActionResult(status="succeeded", physical_effect="confirmed"))
    context.queue_agent_reply({"expected_plan_revision": 1,
        "action": "abort_subtask", "reason": "未恢复承载，保持现场"})
    asyncio.run(_lift_and_verify(context, DepalletizingGraspController(), skill_input, state))
    assert context.status == "failed"
    assert context.failure.code == "AGENT_ABORTED_GRASP"
    assert not state.lift_completed
    assert state.verification_attempts == 1
    assert [event["action"] for event in context.events if event["type"] == "action_started"] == [
        "motion.lift_held_object", "perception.verify_grasp", "gripper.hold_object"]


def test_read_only_verification_can_schedule_actual_missing_lift() -> None:
    """复核确认稳定接合但未抬起时，按真实高度补抬，不伪造已完成状态。"""
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    candidate = _candidate("candidate-a", x=0.82, score=1.0)
    state = GraspObjectState(
        stage="lift_and_verify", target_revision="scene-7",
        target_pose=_pose(0.82, 0.0, 0.20, "scene-7"),
        target_extent_m=(0.42, 0.30, 0.24), candidates=[candidate],
        selected_candidate_id=candidate.candidate_id, object_held=True,
        verification_only=True,
    )
    context = _context(skill_input)
    context.queue_action("perception.verify_grasp", ActionResult(
        status="succeeded", physical_effect="none", observations=[_observation(
            "grasp_verification", {
                "candidate_id": candidate.candidate_id, "held": False,
                "stable_bilateral_load": True, "lift_height_m": 0.0,
                "stable_duration_ms": 700,
                "object_pose": state.target_pose.model_dump(mode="json"),
                "tool_poses": {},
            })]))
    asyncio.run(_lift_and_verify(context, DepalletizingGraspController(), skill_input, state))
    assert not state.lift_completed
    assert not state.verification_only
    assert state.pending_lift_distance_m == pytest.approx(0.04)
    assert [event["action"] for event in context.events if event["type"] == "action_started"] == [
        "perception.verify_grasp"]


def test_failed_pregrasp_after_insert_does_not_switch_candidate() -> None:
    """插入已执行但接触复核失败时，不得直接移动到下一候选。"""

    first = _candidate("candidate-a", x=0.82, score=0.92)
    second = _candidate("candidate-b", x=0.91, score=0.83)
    context = _context()
    _queue_observation_and_candidates(context, [first, second])
    context.queue_action(
        "gripper.set_opening",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    for _ in range(5):
        context.queue_action(
            "motion.move_end_effector",
            ActionResult(status="succeeded", physical_effect="confirmed"),
        )
    context.queue_action(
        "perception.verify_pregrasp",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "pregrasp_state",
                    {
                        "candidate_id": first.candidate_id,
                        "target_revision": "scene-7",
                        "reached": False,
                        "position_errors_m": {
                            TOOL_REFS[0]: 0.04,
                            TOOL_REFS[1]: 0.04,
                        },
                        "object_relative_errors_m": {
                            TOOL_REFS[0]: 0.0,
                            TOOL_REFS[1]: 0.0,
                        },
                        "hook_contacts": {
                            TOOL_REFS[0]: False,
                            TOOL_REFS[1]: False,
                        },
                    },
                )
            ],
        ),
    )
    context.queue_agent_reply(
        {
            "expected_plan_revision": 2,
            "action": "abort_subtask",
            "reason": "保持当前姿态并重新观测",
        }
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "failed"
    requests = [event for event in context.events if event["type"] == "agent_requested"]
    assert len(requests) == 1
    assert requests[0]["context"]["selected_candidate_id"] == first.candidate_id
    assert requests[0]["context"]["details"]["physical_effect"] == "possible"


def test_close_failure_holds_and_never_switches_candidate() -> None:
    """闭合命令已经改变工具位置，即使 effect=none 也必须 hold 后交给 Agent。"""

    first = _candidate("candidate-a", x=0.82, score=0.92)
    second = _candidate("candidate-b", x=0.91, score=0.83)
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    context = _context(skill_input)
    state = GraspObjectState(
        stage="grasp",
        active_strategy="direct_bilateral",
        candidates=[first, second],
        selected_candidate_id=first.candidate_id,
        grasp_attempts=1,
        plan_revision=2,
    )
    context.checkpoint(state)
    context.queue_action(
        "gripper.hold_object",
        ActionResult(
            status="succeeded",
            physical_effect="confirmed",
            evidence_refs=["artifact://hold-after-close-failure"],
        ),
    )
    context.queue_agent_reply(
        {
            "expected_plan_revision": 2,
            "action": "abort_subtask",
            "reason": "检查左右工具实际接触后再决定恢复",
        }
    )

    asyncio.run(
        _handle_partial_grasp_failure(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
            ActionResult(
                status="failed",
                physical_effect="none",
                error_code="CONTACT_NOT_CONFIRMED",
            ),
            reason="双侧接触尚未稳定",
        )
    )

    started = [event for event in context.events if event["type"] == "action_started"]
    assert [event["action"] for event in started] == ["gripper.hold_object"]
    requests = [event for event in context.events if event["type"] == "agent_requested"]
    assert len(requests) == 1
    assert requests[0]["context"]["selected_candidate_id"] == first.candidate_id
    assert requests[0]["context"]["details"]["hold_status"] == "succeeded"
    assert state.selected_candidate_id == first.candidate_id
    assert not any(event["type"] == "stage.recovering" for event in context.events)


def test_single_missing_hook_is_reseated_without_replaying_approach() -> None:
    """一侧下钩回弹时只补座该侧，并固定仍在承载的另一侧。"""

    candidate = _candidate("candidate-a", x=0.82, score=0.92)
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    context = _context(skill_input)
    state = GraspObjectState(
        stage="grasp",
        active_strategy="direct_bilateral",
        target_revision="scene-7",
        candidates=[candidate],
        selected_candidate_id=candidate.candidate_id,
        plan_revision=2,
    )
    load = ToolLoadObservationValue.model_validate(
        {
            "condition_satisfied": False,
            "observed_duration_ms": 0,
            "tools": [
                {
                    "tool_ref": TOOL_REFS[0],
                    "available": True,
                    "hook_contact": False,
                    "clamp_contact": True,
                    "sensor_fault": False,
                },
                {
                    "tool_ref": TOOL_REFS[1],
                    "available": True,
                    "hook_contact": True,
                    "clamp_contact": True,
                    "sensor_fault": False,
                },
            ],
            "slip_detected": False,
            "overload_detected": False,
            "sensor_fault": False,
            "reasons": [f"{TOOL_REFS[0]}:hook_contact_missing"],
        }
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "gripper.close",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, list(TOOL_REFS))

    recovered = asyncio.run(
        _reseat_single_missing_hook(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
            candidate,
            load,
        )
    )

    assert recovered is not None
    result, recovered_load = recovered
    assert result.status == "succeeded"
    assert recovered_load is not None and recovered_load.condition_satisfied
    started = [event for event in context.events if event["type"] == "action_started"]
    assert [event["action"] for event in started] == [
        "motion.move_end_effector",
        "gripper.close",
        "robot.verify_tool_load",
    ]
    reseat = started[0]
    assert [target["tool_ref"] for target in reseat["parameters"]["targets"]] == [
        TOOL_REFS[0]
    ]
    assert reseat["parameters"]["required_contact_tools"] == [TOOL_REFS[1]]


def test_close_failure_retry_opens_and_retracts_before_reobservation() -> None:
    """部分闭合后的重试必须先沿原候选退出，不能在箱沿内直接重新 approach。"""

    candidate = _candidate("candidate-a", x=0.82, score=0.92)
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    context = _context(skill_input)
    state = GraspObjectState(
        stage="grasp",
        active_strategy="direct_bilateral",
        target_revision="scene-7",
        candidates=[candidate],
        selected_candidate_id=candidate.candidate_id,
        grasp_attempts=1,
        plan_revision=2,
    )
    context.checkpoint(state)
    context.queue_action(
        "gripper.hold_object",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "gripper.set_opening",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    for _ in range(2):
        context.queue_action(
            "motion.move_end_effector",
            ActionResult(status="succeeded", physical_effect="confirmed"),
        )
    context.queue_agent_reply(
        {
            "expected_plan_revision": 2,
            "action": "restart_observation",
            "reason": "释放未承载工具后重新读取箱体位置",
        }
    )

    asyncio.run(
        _handle_partial_grasp_failure(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
            ActionResult(
                status="failed",
                physical_effect="possible",
                error_code="CONTACT_NOT_CONFIRMED",
            ),
            reason="双侧接触尚未稳定",
        )
    )

    started = [event for event in context.events if event["type"] == "action_started"]
    assert [event["action"] for event in started] == [
        "gripper.hold_object",
        "gripper.set_opening",
        "motion.move_end_effector",
        "motion.move_end_effector",
    ]
    recovery = DepalletizingGraspController().recover_after_partial_close(
        skill_input,
        candidate=candidate,
        target_revision="scene-7",
    )
    assert [action.parameters.get("purpose") for action in recovery] == [
        None,
        "extract",
        "extract",
    ]
    assert recovery[1].parameters["targets"] == [
        value.model_dump(mode="json") for value in candidate.hook_insert_poses
    ]
    assert recovery[2].parameters["targets"] == [
        value.model_dump(mode="json") for value in candidate.approach_poses
    ]
    assert state.stage == "observe_target"
    assert state.selected_candidate_id is None
    assert state.plan_revision == 3


def test_exhausted_local_recovery_requests_versioned_agent_decision() -> None:
    """观测恢复耗尽后请求 Agent，并允许 Agent 明确终止 SubTask。"""

    context = _context()
    for _ in range(2):
        context.queue_action(
            "perception.locate_object",
            ActionResult(
                status="failed",
                physical_effect="none",
                error_code="TARGET_NOT_FOUND",
            ),
        )
    context.queue_agent_reply(
        {
            "expected_plan_revision": 0,
            "action": "abort_subtask",
            "reason": "目标已不在当前托盘区域",
        }
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "failed"
    assert context.failure is not None
    assert context.failure.code == "AGENT_ABORTED_GRASP"
    requests = [event for event in context.events if event["type"] == "agent_requested"]
    assert len(requests) == 1
    assert requests[0]["context"]["plan_revision"] == 0


def test_on_stop_holds_verified_object_and_requests_intervention() -> None:
    """持物期间停止必须先执行专用保持动作，再返回物理状态。"""

    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    context = _context(skill_input)
    context.queue_action(
        "gripper.hold_object",
        ActionResult(
            status="succeeded",
            physical_effect="confirmed",
            evidence_refs=["artifact://hold-state"],
        ),
    )
    state = GraspObjectState(
        stage="lift_and_verify",
        active_strategy="direct_bilateral",
        object_held=True,
        evidence_refs=["artifact://grasp-contact"],
    )
    context.checkpoint(state)
    context.request_stop()
    outcome = asyncio.run(
        on_stop(
            context,
            StopRequest(source="user", reason="用户停止真机动作"),
        )
    )

    assert outcome.safe is True
    assert outcome.physical_state == "holding_object"
    assert outcome.requires_intervention is True
    assert "artifact://hold-state" in outcome.evidence_refs


def test_on_stop_without_object_executes_dedicated_safe_stop() -> None:
    """尚未持物时也必须由执行侧确认机械臂和夹爪已经停止。"""

    context = _context()
    context.checkpoint(
        GraspObjectState(
            stage="approach",
            active_strategy="direct_bilateral",
            object_held=False,
        )
    )
    context.request_stop()
    context.queue_action(
        "gripper.hold_object",
        ActionResult(
            status="succeeded",
            physical_effect="confirmed",
            evidence_refs=["artifact://grasp-motion-stopped"],
        ),
    )

    outcome = asyncio.run(
        on_stop(
            context,
            StopRequest(
                source="user",
                reason="用户停止预抓取运动",
                mode="immediate",
            ),
        )
    )

    assert outcome.safe is True
    assert outcome.physical_state == "stopped_without_object"
    assert outcome.requires_intervention is False


def test_extract_first_replans_pull_from_primary_contact() -> None:
    """首侧夹紧后必须丢弃抓取前生成的pull目标并读取实时几何。"""

    initial = _candidate(
        "candidate-before-contact",
        x=0.82,
        score=0.94,
        strategy="left_extract_first",
    )
    refreshed_payload = initial.model_dump(mode="json")
    refreshed_payload["candidate_id"] = "candidate-from-contact"
    refreshed_payload["observation_revision"] = "scene-contact"
    refreshed_payload["pull_path"] = [
        {
            "tool_ref": TOOL_REFS[0],
            "target_pose": _pose(0.70, 0.25, 0.14, "scene-contact").model_dump(
                mode="json"
            ),
        }
    ]
    refreshed_payload["pull_direction_world"] = [-1.0, 0.0, 0.0]
    refreshed_payload["pull_distance_m"] = 0.10
    refreshed = GraspCandidate.model_validate(refreshed_payload)
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    context = _context(skill_input)
    state = GraspObjectState(
        stage="grasp",
        active_strategy="left_extract_first",
        target_revision="scene-before-contact",
        target_pose=_pose(0.82, 0.0, 0.12, "scene-before-contact"),
        target_extent_m=(0.42, 0.30, 0.24),
        candidates=[initial],
        selected_candidate_id=initial.candidate_id,
        plan_revision=1,
        primary_clamped=True,
        extraction_primary_side="left",
    )
    context.checkpoint(state)
    contact_pose = _pose(0.80, 0.0, 0.12, "scene-contact")
    context.queue_action(
        "perception.locate_object",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "target_pose",
                    {
                        "object_ref": OBJECT_REF,
                        "pose": contact_pose.model_dump(mode="json"),
                        "extent_m": [0.42, 0.30, 0.24],
                        "identity_confidence": 0.98,
                    },
                    revision="scene-contact",
                )
            ],
        ),
    )
    context.queue_action(
        "grasp.generate_candidates",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "grasp_candidates",
                    {
                        "target_revision": "scene-contact",
                        "candidates": [refreshed.model_dump(mode="json")],
                    },
                    revision="scene-contact",
                )
            ],
        ),
    )

    assert asyncio.run(
        _refresh_extraction_plan_from_contact(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
            primary_side="left",
            primary_ref=TOOL_REFS[0],
        )
    )
    assert state.extraction_planned_from_contact is True
    assert state.extraction_completed is False
    assert state.selected_candidate_id == refreshed.candidate_id
    generated = next(
        event
        for event in context.events
        if event["type"] == "action_started"
        and event["action"] == "grasp.generate_candidates"
    )
    assert generated["parameters"]["engaged_tool_ref"] == TOOL_REFS[0]
    assert generated["parameters"]["preferred_strategy"] == "direct_bilateral"


def test_extract_first_keeps_primary_clamped_while_engaging_secondary() -> None:
    """外拉后首侧持续承载，并作为联合 IK 约束等待第二侧接管。"""

    extraction_payload = _candidate(
        "candidate-left-extract",
        x=0.82,
        score=0.94,
        strategy="left_extract_first",
    ).model_dump(mode="json")
    extraction_payload["pull_path"] = [
        {
            "tool_ref": TOOL_REFS[0],
            "target_pose": _pose(0.72, 0.25, 0.14, "scene-7").model_dump(mode="json"),
        }
    ]
    extraction_payload["pull_direction_world"] = [-1.0, 0.0, 0.0]
    extraction_payload["pull_distance_m"] = 0.10
    extraction = GraspCandidate.model_validate(extraction_payload)
    bilateral = _candidate("candidate-direct-after-pull", x=0.72, score=0.97)
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    context = _context(skill_input)
    state = GraspObjectState(
        stage="grasp",
        active_strategy="left_extract_first",
        target_revision="scene-7",
        target_pose=_pose(0.82, 0.0, 0.12, "scene-7"),
        target_extent_m=(0.42, 0.30, 0.24),
        candidates=[extraction],
        selected_candidate_id=extraction.candidate_id,
        plan_revision=1,
        extraction_planned_from_contact=True,
    )
    context.checkpoint(state)

    primary_contact = _observation(
        "grasp_contact",
        {
            "candidate_id": extraction.candidate_id,
            "object_ref": OBJECT_REF,
            "tools": {
                TOOL_REFS[0]: {
                    "hook_contact": True,
                    "clamp_contact": True,
                    "slipping": False,
                    "overloaded": False,
                    "sensor_fault": False,
                }
            },
            "contact_confirmed": True,
            "stable_bilateral_load": False,
            "slipping": False,
            "overloaded": False,
        },
    )
    context.queue_action(
        "gripper.close",
        ActionResult(
            status="succeeded",
            physical_effect="confirmed",
            observations=[primary_contact],
        ),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, [TOOL_REFS[0]])
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, [TOOL_REFS[0]])
    refreshed_pose = _pose(0.72, 0.0, 0.12, "scene-8")
    context.queue_action(
        "perception.locate_object",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "target_pose",
                    {
                        "object_ref": OBJECT_REF,
                        "pose": refreshed_pose.model_dump(mode="json"),
                        "extent_m": [0.42, 0.30, 0.24],
                        "identity_confidence": 0.98,
                    },
                    revision="scene-8",
                )
            ],
        ),
    )
    context.queue_action(
        "grasp.generate_candidates",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "grasp_candidates",
                    {
                        "target_revision": "scene-8",
                        "candidates": [bilateral.model_dump(mode="json")],
                    },
                    revision="scene-8",
                )
            ],
        ),
    )
    _queue_observation_and_candidates(
        context,
        [bilateral],
        pose=_pose(0.715, 0.0, 0.12, "scene-9"),
        revision="scene-9",
    )
    _queue_observation_and_candidates(
        context,
        [bilateral],
        pose=_pose(0.712, 0.0, 0.12, "scene-10"),
        revision="scene-10",
    )

    context.queue_action(
        "gripper.set_opening",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    # 第二侧依次执行clearance/transfer/斜向pregrasp/insert；
    # 首侧由Ability固定并持续监控，Skill不在段间重复close或VerifyToolLoad。
    for _ in range(4):
        context.queue_action(
            "motion.move_end_effector",
            ActionResult(status="succeeded", physical_effect="confirmed"),
        )
    context.queue_action(
        "perception.verify_pregrasp",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "pregrasp_state",
                    {
                        "candidate_id": bilateral.candidate_id,
                        "target_revision": "scene-8",
                        "reached": True,
                        "position_errors_m": {
                            TOOL_REFS[1]: 0.009,
                        },
                        "object_relative_errors_m": {
                            TOOL_REFS[1]: 0.009,
                        },
                        "hook_contacts": {
                            TOOL_REFS[1]: True,
                        },
                    },
                    revision="scene-8",
                )
            ],
        ),
    )
    bilateral_contact = _observation(
        "grasp_contact",
        {
            "candidate_id": bilateral.candidate_id,
            "object_ref": OBJECT_REF,
            "tools": {
                ref: {
                    "hook_contact": True,
                    "clamp_contact": True,
                    "slipping": False,
                    "overloaded": False,
                    "sensor_fault": False,
                }
                for ref in TOOL_REFS
            },
            "contact_confirmed": True,
            "stable_bilateral_load": True,
            "slipping": False,
            "overloaded": False,
        },
        revision="scene-8",
    )
    context.queue_action(
        "gripper.close",
        ActionResult(
            status="succeeded",
            physical_effect="confirmed",
            observations=[bilateral_contact],
        ),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, TOOL_REFS)

    asyncio.run(
        _grasp(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
        )
    )

    assert state.stage == "lift_and_verify"
    assert state.active_strategy == "direct_bilateral"
    assert state.selected_candidate_id == bilateral.candidate_id
    assert state.extraction_reobserved is True
    assert state.primary_clamped is False
    assert state.extraction_primary_side is None
    started = [event for event in context.events if event["type"] == "action_started"]
    assert not any(
        event["key"].startswith("grasp:release-primary") for event in started
    )
    secondary_open = next(
        event
        for event in started
        if event["key"].startswith("grasp:reengage")
        and event["action"] == "gripper.set_opening"
    )
    assert [item["tool_ref"] for item in secondary_open["parameters"]["tools"]] == [
        TOOL_REFS[1]
    ]
    secondary_motions = [
        event
        for event in started
        if event["key"].startswith("grasp:reengage")
        and event["action"] == "motion.move_end_effector"
    ]
    assert len(secondary_motions) == 4
    assert all(
        [item["tool_ref"] for item in motion["parameters"]["targets"]] == [TOOL_REFS[1]]
        for motion in secondary_motions
    )
    assert all(
        motion["parameters"]["required_contact_tools"] == [TOOL_REFS[0]]
        for motion in secondary_motions
    )
    assert not any(
        event["key"].startswith("grasp:reengage") and event["action"] == "gripper.close"
        for event in started
    )
    assert (
        sum(event["key"].startswith("grasp:reengage-candidates") for event in started)
        == 1
    )
    recandidates = next(
        event
        for event in started
        if event["key"].startswith("grasp:reengage-candidates")
    )
    assert recandidates["parameters"]["preferred_strategy"] == "direct_bilateral"
    assert recandidates["parameters"]["secondary_resume_phase"] == "insert"
    final_close = [event for event in started if event["action"] == "gripper.close"][-1]
    assert [item["tool_ref"] for item in final_close["parameters"]["tools"]] == [
        TOOL_REFS[1]
    ]
    assert not any(event["type"] == "agent_requested" for event in context.events)


def test_extract_first_reuses_fresh_geometry_without_repeating_extraction() -> None:
    """外拉后候选标签不阻止Skill复用新Pose接合第二侧。"""

    extraction_payload = _candidate(
        "candidate-left-extract",
        x=0.82,
        score=0.94,
        strategy="left_extract_first",
    ).model_dump(mode="json")
    extraction_payload["pull_path"] = [
        {
            "tool_ref": TOOL_REFS[0],
            "target_pose": _pose(0.72, 0.25, 0.14, "scene-7").model_dump(mode="json"),
        }
    ]
    extraction_payload["pull_direction_world"] = [-1.0, 0.0, 0.0]
    extraction_payload["pull_distance_m"] = 0.10
    extraction = GraspCandidate.model_validate(extraction_payload)
    alternate = _candidate(
        "candidate-right-after-pull",
        x=0.72,
        score=0.91,
        strategy="right_extract_first",
    )
    final_bilateral = _candidate(
        "candidate-direct-before-insert",
        x=0.715,
        score=0.97,
    )
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    context = _context(skill_input)
    state = GraspObjectState(
        stage="grasp",
        active_strategy="left_extract_first",
        target_revision="scene-7",
        target_pose=_pose(0.82, 0.0, 0.12, "scene-7"),
        target_extent_m=(0.42, 0.30, 0.24),
        candidates=[extraction],
        selected_candidate_id=extraction.candidate_id,
        plan_revision=1,
        extraction_planned_from_contact=True,
    )
    context.checkpoint(state)
    context.queue_action(
        "gripper.close",
        ActionResult(
            status="succeeded",
            physical_effect="confirmed",
            observations=[
                _observation(
                    "grasp_contact",
                    {
                        "candidate_id": extraction.candidate_id,
                        "object_ref": OBJECT_REF,
                        "tools": {
                            TOOL_REFS[0]: {
                                "hook_contact": True,
                                "clamp_contact": True,
                                "slipping": False,
                                "overloaded": False,
                                "sensor_fault": False,
                            }
                        },
                        "contact_confirmed": True,
                        "stable_bilateral_load": False,
                    },
                )
            ],
        ),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, [TOOL_REFS[0]])
    _queue_tool_load(context, [TOOL_REFS[0]])
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    refreshed_pose = _pose(0.72, 0.0, 0.12, "scene-8")
    context.queue_action(
        "perception.locate_object",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "target_pose",
                    {
                        "object_ref": OBJECT_REF,
                        "pose": refreshed_pose.model_dump(mode="json"),
                        "extent_m": [0.42, 0.30, 0.24],
                        "identity_confidence": 0.98,
                    },
                    revision="scene-8",
                )
            ],
        ),
    )
    context.queue_action(
        "grasp.generate_candidates",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "grasp_candidates",
                    {
                        "target_revision": "scene-8",
                        "candidates": [alternate.model_dump(mode="json")],
                    },
                    revision="scene-8",
                )
            ],
        ),
    )
    _queue_observation_and_candidates(
        context,
        [final_bilateral],
        pose=_pose(0.715, 0.0, 0.12, "scene-9"),
        revision="scene-9",
    )
    _queue_observation_and_candidates(
        context,
        [final_bilateral],
        pose=_pose(0.712, 0.0, 0.12, "scene-10"),
        revision="scene-10",
    )
    context.queue_action(
        "gripper.set_opening",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    for _ in range(5):
        context.queue_action(
            "motion.move_end_effector",
            ActionResult(status="succeeded", physical_effect="confirmed"),
        )
    context.queue_action(
        "perception.verify_pregrasp",
        ActionResult(
            status="succeeded",
            physical_effect="none",
            observations=[
                _observation(
                    "pregrasp_state",
                    {
                        "candidate_id": final_bilateral.candidate_id,
                        "target_revision": "scene-9",
                        "reached": True,
                        "position_errors_m": {
                            TOOL_REFS[0]: 0.004,
                            TOOL_REFS[1]: 0.003,
                        },
                        "object_relative_errors_m": {
                            TOOL_REFS[0]: 0.0,
                            TOOL_REFS[1]: 0.0,
                        },
                        "hook_contacts": {
                            TOOL_REFS[0]: True,
                            TOOL_REFS[1]: True,
                        },
                    },
                    revision="scene-9",
                )
            ],
        ),
    )
    context.queue_action(
        "gripper.close",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, TOOL_REFS)

    asyncio.run(_grasp(context, DepalletizingGraspController(), skill_input, state))

    assert not any(event["type"] == "agent_requested" for event in context.events)
    assert state.stage == "lift_and_verify"
    assert state.selected_candidate_id == final_bilateral.candidate_id
    assert state.primary_clamped is False
    assert state.extraction_primary_side is None
    # 双侧接合成功后临时外拉状态会被清理；事件中只能出现一次真实外拉。
    assert state.extraction_completed is False
    assert state.extraction_reobserved is True
    assert state.active_strategy == "direct_bilateral"
    assert state.grasp_attempts == 1
    started = [event for event in context.events if event["type"] == "action_started"]
    assert sum(event["key"].startswith("grasp:extract:") for event in started) == 1
    recandidates = next(
        event for event in started if event["action"] == "grasp.generate_candidates"
    )
    assert recandidates["parameters"]["engaged_tool_ref"] == TOOL_REFS[0]
    assert (
        sum(
            ":realign" in event["key"]
            for event in started
            if event["action"] == "motion.move_end_effector"
        )
        == 1
    )


def test_extraction_continues_while_live_clearance_is_improving() -> None:
    """不按固定次数中止仍有真实进展的凹槽外拉。"""

    initial_payload = _candidate(
        "candidate-left-before-adjustment",
        x=0.72,
        score=0.94,
        strategy="left_extract_first",
    ).model_dump(mode="json")
    initial_payload["pull_path"] = [
        {
            "tool_ref": TOOL_REFS[0],
            "target_pose": _pose(0.70, 0.25, 0.14, "scene-9").model_dump(mode="json"),
        }
    ]
    initial_payload["pull_direction_world"] = [-1.0, 0.0, 0.0]
    initial_payload["pull_distance_m"] = 0.02
    initial = GraspCandidate.model_validate(initial_payload)

    adjustment_payload = _candidate(
        "candidate-left-final-adjustment",
        x=0.70,
        score=0.95,
        strategy="left_extract_first",
    ).model_dump(mode="json")
    adjustment_payload["pull_path"] = [
        {
            "tool_ref": TOOL_REFS[0],
            "target_pose": _pose(0.69, 0.25, 0.14, "scene-10").model_dump(mode="json"),
        }
    ]
    adjustment_payload["pull_direction_world"] = [-1.0, 0.0, 0.0]
    adjustment_payload["pull_distance_m"] = 0.01
    adjustment = GraspCandidate.model_validate(adjustment_payload)
    direct = _candidate("candidate-direct-after-clearance", x=0.69, score=0.98)

    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    context = _context(skill_input)
    state = GraspObjectState(
        stage="grasp",
        active_strategy="left_extract_first",
        target_revision="scene-8",
        target_pose=_pose(0.72, 0.0, 0.12, "scene-8"),
        target_extent_m=(0.42, 0.30, 0.24),
        candidates=[initial],
        selected_candidate_id=initial.candidate_id,
        plan_revision=3,
        primary_clamped=True,
        extraction_primary_side="left",
        # 真实执行已经完成过一次补拉；旧实现会在这里无条件等待Agent。
        extraction_adjustments=1,
    )
    context.checkpoint(state)

    _queue_tool_load(context, [TOOL_REFS[0]])
    _queue_observation_and_candidates(
        context,
        [adjustment],
        pose=_pose(0.70, 0.0, 0.12, "scene-9"),
        revision="scene-9",
    )
    context.queue_action(
        "motion.move_end_effector",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, [TOOL_REFS[0]])
    _queue_observation_and_candidates(
        context,
        [direct],
        pose=_pose(0.69, 0.0, 0.12, "scene-10"),
        revision="scene-10",
    )

    assert asyncio.run(
        _reobserve_after_extraction(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
        )
    )
    assert state.extraction_adjustments == 2
    assert state.selected_candidate_id == direct.candidate_id
    assert not any(event["type"] == "agent_requested" for event in context.events)
    adjustments = [
        event
        for event in context.events
        if event["type"] == "action_started"
        and event["key"].startswith("grasp:extract-adjust:")
    ]
    assert len(adjustments) == 1


def test_primary_reclamp_is_conditional_and_can_only_run_once() -> None:
    skill_input = GraspObjectInput(target=TargetHint(object_ref=OBJECT_REF))
    candidate = _candidate(
        "candidate-left-extract",
        x=0.82,
        score=0.9,
        strategy="left_extract_first",
    )
    state = GraspObjectState(
        stage="grasp",
        plan_revision=3,
        candidates=[candidate],
        selected_candidate_id=candidate.candidate_id,
        primary_clamped=True,
        extraction_primary_side="left",
    )
    context = _context(skill_input)

    # 第一次仅压紧接触下降、钩爪和传感状态仍可信，允许一次恢复夹紧。
    _queue_tool_load(
        context,
        [TOOL_REFS[0]],
        condition_satisfied=False,
        hook_contact=True,
        clamp_contact=False,
    )
    context.queue_action(
        "gripper.close",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    _queue_tool_load(context, [TOOL_REFS[0]])
    first = asyncio.run(
        _ensure_primary_load(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
            candidate,
            TOOL_REFS[0],
            key_suffix="first",
        )
    )
    assert first is True
    assert state.primary_reclamp_attempted is True

    # 后续再次下降不能重放close；必须hold并交给Agent处理。
    _queue_tool_load(
        context,
        [TOOL_REFS[0]],
        condition_satisfied=False,
        hook_contact=True,
        clamp_contact=False,
    )
    context.queue_action(
        "gripper.hold_object",
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_agent_reply(
        {
            "expected_plan_revision": state.plan_revision,
            "action": "abort_subtask",
            "reason": "二次承载下降，保持现场",
        }
    )
    second = asyncio.run(
        _ensure_primary_load(
            context,
            DepalletizingGraspController(),
            skill_input,
            state,
            candidate,
            TOOL_REFS[0],
            key_suffix="second",
        )
    )
    assert second is False
    started = [event for event in context.events if event["type"] == "action_started"]
    assert sum(event["action"] == "gripper.close" for event in started) == 1
    assert sum(event["action"] == "gripper.hold_object" for event in started) == 1
    assert sum(event["type"] == "agent_requested" for event in context.events) == 1
