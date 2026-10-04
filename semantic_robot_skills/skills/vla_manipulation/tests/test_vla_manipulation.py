import asyncio
from types import SimpleNamespace

import pytest

from semantic_robot_skill_sdk import (
    ActionFeedback,
    ActionResult,
    MockSkillContext,
    StopRequest,
    run_skill,
)
from semantic_robot_skills.skills.vla_manipulation.scripts.models import (
    ManipulationInput,
    ManipulationState,
)
from semantic_robot_skills.skills.vla_manipulation.scripts.skill import on_stop, run
from semantic_robot_skills.skills.vla_manipulation.scripts import skill
from semantic_robot_skills.skills.vla_manipulation.scripts.verification import assess


def sample(sequence=1, lifted=False, contact=False, success=False):
    z = .85 if lifted else .8
    return {
        "generation": 1, "sequence": sequence, "sim_time": sequence * .1,
        "target": {"source_id": "bowl", "pose": {"position": [0, 0, z]},
                   "state": {"fixture": False}},
        "robot_state": {"robot_id": "franka", "end_effectors": {
            "hand": {"position": [0, 0, z + .05]}}, "grippers": {"hand": .03}},
        "contacts": {"contacts": [{"source_id": "bowl", "bilateral_contact": contact}]},
        "evaluation": {"success": success}, "artifact_refs": ["artifact://photo"],
    }


def success(output):
    return ActionResult(status="succeeded", output=output)


def context(objective="native_task"):
    ctx = MockSkillContext(ManipulationInput(objective=objective, target_source_id="bowl",
                                             instruction="pick up the bowl"))
    ctx.queue_action("vla.observe_environment", success(sample()))
    ctx.queue_action("vla.get_model_binding", success({
        "model_binding": {"ready": True, "robot_id": "franka"}}))
    return ctx


def test_native_task_checks_actual_evaluation_after_policy():
    ctx = context()
    ctx.queue_action("vla.execute_policy", success({"task_success": None}))
    ctx.queue_action("vla.observe_environment", success(sample(5, success=True)))
    asyncio.run(run_skill(run, ctx))
    assert ctx.result.native_task_success is True
    assert ctx.result.evidence_refs is not None


def test_policy_succeeded_does_not_mean_task_succeeded():
    ctx = context()
    ctx.queue_action("vla.execute_policy", success({}))
    ctx.queue_action("vla.observe_environment", success(sample(5, success=False)))
    asyncio.run(run_skill(run, ctx))
    assert ctx.failure.code == "OBJECTIVE_NOT_MET"
    assert ctx.status == "failed"
    assert ctx.result is None


def test_failed_initial_observation_preserves_error_without_starting_policy():
    ctx = MockSkillContext(ManipulationInput(objective="native_task", instruction="pick"))
    ctx.queue_action("vla.observe_environment", ActionResult(
        status="failed", error_code="OBSERVATION_FAILED", error_message="相机不可用"))
    asyncio.run(run_skill(run, ctx))
    assert ctx.status == "failed"
    assert ctx.failure.code == "OBSERVATION_FAILED"
    assert "vla:policy" not in ctx._records


@pytest.mark.parametrize("objective", ["native_task", "grasp"])
@pytest.mark.parametrize("lose_target", [False, True])
def test_both_objectives_continue_twenty_actions_and_recheck(monkeypatch, objective, lose_target):
    clock = iter(range(1000))
    monkeypatch.setattr(skill, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    ctx = context(objective)
    # 原生目标第一个采样成立；抓取在连续稳定观测后（第 10 步）成立。
    first = 0 if objective == "native_task" else 10
    counts = list(range(0, first + 21, 5))
    ctx.queue_action("vla.execute_policy", success({"executed_actions": first + 20}), feedback=[
        ActionFeedback(sequence=i + 1, status="running", measurements={
            "phase": "control", "executed_actions": count, "policy_invocation_id": "policy"})
        for i, count in enumerate(counts)
    ])
    ctx.queue_action("vla.set_execution_limit", success({"action_limit": first + 20}))
    for i, count in enumerate(counts + [first + 20]):
        reached = not lose_target or count <= first
        ctx.queue_action("vla.observe_environment", success(sample(
            i + 2, lifted=reached, contact=reached, success=reached)))
    asyncio.run(run_skill(run, ctx))
    assert ctx.state.first_success_actions == first
    assert ctx.state.tail_action_limit == first + 20
    assert ctx.state.tail_limit_applied
    assert ctx._records["vla:policy"].result.status == "succeeded"
    if lose_target:
        assert ctx.failure.code == "OBJECTIVE_NOT_MET"
    else:
        assert ctx.status == "completed"
        assert ctx.result.post_success_actions == 20
        assert ctx.result.post_success_complete


def test_relative_tail_uses_ability_cursor_instead_of_delayed_feedback(monkeypatch):
    clock = iter(range(1000))
    monkeypatch.setattr(skill, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    ctx = context()
    ctx.queue_action("vla.execute_policy", success({"executed_actions": 180}), feedback=[
        ActionFeedback(sequence=1, status="running", measurements={
            "phase": "control", "executed_actions": 50, "policy_invocation_id": "policy",
            "supports_relative_action_limit": True})])
    ctx.queue_action("vla.set_execution_limit", success({"start_actions": 160, "action_limit": 180}))
    ctx.queue_action("vla.observe_environment", success(sample(2, success=True)))
    ctx.queue_action("vla.observe_environment", success(sample(3, success=True)))
    asyncio.run(run_skill(run, ctx))
    assert ctx.status == "completed"
    assert ctx.state.first_success_actions == 160
    assert ctx.state.tail_action_limit == 180
    assert ctx.result.post_success_actions == 20


def test_tail_respects_total_budget_and_reports_short_tail(monkeypatch):
    clock = iter(range(1000))
    monkeypatch.setattr(skill, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    ctx = context()
    ctx._input.max_actions = 15
    ctx.queue_action("vla.execute_policy", success({"executed_actions": 15}), feedback=[
        ActionFeedback(sequence=1, status="running", measurements={
            "phase": "control", "executed_actions": 10, "policy_invocation_id": "policy"})])
    ctx.queue_action("vla.set_execution_limit", success({"action_limit": 15}))
    for i in (2, 3):
        ctx.queue_action("vla.observe_environment", success(sample(i, success=True)))
    asyncio.run(run_skill(run, ctx))
    assert ctx.state.tail_action_limit == 15
    assert ctx.result.post_success_actions == 5
    assert not ctx.result.post_success_complete


def test_rejected_tail_limit_stops_existing_policy():
    ctx = context()
    ctx.queue_action("vla.execute_policy", success({}), feedback=[
        ActionFeedback(sequence=1, status="running", measurements={
            "phase": "control", "executed_actions": 10, "policy_invocation_id": "policy"})])
    ctx.queue_action("vla.observe_environment", success(sample(2, success=True)))
    ctx.queue_action("vla.set_execution_limit", ActionResult(status="failed", error_message="预算已越过"))
    asyncio.run(run_skill(run, ctx))
    assert ctx.failure.code == "TAIL_LIMIT_FAILED"
    assert ctx._records["vla:policy"].result.status == "stopped"


def test_contact_without_lift_or_wrong_object_never_counts_as_grasp():
    inputs = ManipulationInput(objective="grasp", target_source_id="bowl", instruction="pick")
    state = ManipulationState(generation=1, initial_target_height=.8)
    assert not assess(sample(1, contact=True), inputs, state)
    wrong = sample(2, lifted=True, contact=True)
    wrong["contacts"]["contacts"][0]["source_id"] = "another-object"
    assert not assess(wrong, inputs, state)
    assert not assess(sample(3, lifted=True, contact=True), inputs, state)
    assert not assess(sample(3, lifted=True, contact=True), inputs, state)
    assert assess(sample(5, lifted=True, contact=True), inputs, state)


def test_generation_change_invalidates_verification():
    inputs = ManipulationInput(objective="native_task", instruction="native task")
    with pytest.raises(ValueError, match="重置"):
        assess(sample(success=True), inputs, ManipulationState(generation=2))


def test_stop_uses_dedicated_hold_without_observation_or_inference():
    ctx = context()
    ctx.queue_action("vla.hold_robot", success({"safe": True}))
    outcome = asyncio.run(on_stop(ctx, StopRequest(source="user", reason="stop")))
    assert outcome.safe and outcome.physical_state == "hold"
    assert "vla:policy" not in ctx._records
