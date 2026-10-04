"""语义导航 Skill 的 Stage、恢复、Agent 决策和停止验收测试。"""

from __future__ import annotations

import asyncio

from semantic_robot_skill_sdk import ActionFeedback, ActionResult, MockSkillContext, Observation, Pose3D, StopRequest, run_skill
from semantic_robot_skills.skills.semantic_navigation.scripts.models import (
    ResolvedNavigationTarget,
    SemanticNavigationInput,
    SemanticNavigationResult,
    SemanticNavigationState,
)
from semantic_robot_skills.skills.semantic_navigation.scripts.skill import on_stop, run


def _target(*, revision: str = "target-r1", x: float = 1.0) -> ResolvedNavigationTarget:
    return ResolvedNavigationTarget(
        target_ref="semantic://station/loading-a",
        pose=Pose3D(frame_id="map", position_m=(x, 2.0, 0.0), revision=revision),
    )


def _skill_input() -> SemanticNavigationInput:
    return SemanticNavigationInput(
        target=_target(),
        arrival_radius_m=0.4,
        maximum_speed_mps=0.6,
    )


def _plan_result(route_ref: str) -> ActionResult:
    return ActionResult(status="succeeded", output={"route_ref": route_ref}, physical_effect="none")


def _follow_result() -> ActionResult:
    return ActionResult(
        status="succeeded",
        output={"final_pose_ref": "pose://robot/final", "distance_to_target_m": 0.18},
        physical_effect="confirmed",
    )


def _verify_result() -> ActionResult:
    return ActionResult(
        status="succeeded",
        output={"verdict": "achieved", "final_pose_ref": "pose://robot/final", "distance_to_target_m": 0.18},
        evidence_refs=["artifact://navigation/arrival-proof"],
        physical_effect="none",
    )


def _run(context: MockSkillContext) -> None:
    asyncio.run(run_skill(run, context))


def _actions(context: MockSkillContext) -> list[str]:
    return [event["action"] for event in context.events if event.get("type") == "action_started"]


def _completed_stages(context: MockSkillContext) -> list[str]:
    return [
        event["stage"]
        for event in context.events
        if event.get("type") == "stage.completed"
    ]


def test_navigation_defaults_are_execution_safe_and_do_not_repeat_base_footprint() -> None:
    skill_input = SemanticNavigationInput(target=_target())

    assert skill_input.arrival_radius_m == 0.03
    assert skill_input.maximum_speed_mps == 0.15
    assert skill_input.minimum_clearance_m == 0.05


def test_normal_navigation_completes_after_independent_verification() -> None:
    context = MockSkillContext(_skill_input())
    context.queue_action("navigation.plan_route", _plan_result("route://loading-a/1"))
    context.queue_action("navigation.follow_route", _follow_result())
    context.queue_action("navigation.verify_arrival", _verify_result())

    _run(context)

    assert context.status == "completed"
    assert isinstance(context.result, SemanticNavigationResult)
    assert _actions(context) == [
        "navigation.plan_route",
        "navigation.follow_route",
        "navigation.verify_arrival",
    ]
    assert _completed_stages(context) == [
        "validate_target",
        "plan_route",
        "navigate",
        "verify_arrival",
    ]


def test_route_blocked_stops_current_action_and_replans_locally() -> None:
    context = MockSkillContext(_skill_input())
    context.queue_action("navigation.plan_route", _plan_result("route://loading-a/1"))
    context.queue_action(
        "navigation.follow_route",
        _follow_result(),
        feedback=[
            ActionFeedback(
                sequence=1,
                status="running",
                progress=0.35,
                severity="warning",
                observations=[Observation(kind="route_blocked", source="pilot://robot-1/navigation", value={"blocked": True})],
            )
        ],
        stop_result=ActionResult(status="stopped", physical_effect="confirmed"),
    )
    context.queue_action("navigation.plan_route", _plan_result("route://loading-a/2"))
    context.queue_action("navigation.follow_route", _follow_result())
    context.queue_action("navigation.verify_arrival", _verify_result())

    _run(context)

    assert context.status == "completed"
    assert isinstance(context.state, SemanticNavigationState)
    assert context.state.replan_count == 1
    assert _actions(context).count("navigation.plan_route") == 2
    assert _actions(context).count("navigation.follow_route") == 2


def test_agent_decision_can_reopen_route_planning_after_budget_exhaustion() -> None:
    context = MockSkillContext(_skill_input())
    for _ in range(2):
        context.queue_action(
            "navigation.plan_route",
            ActionResult(status="failed", error_code="NO_FEASIBLE_ROUTE", physical_effect="none"),
        )
    context.queue_agent_reply({"action": "replan_route", "reason": "更新局部代价图后重新规划"})
    context.queue_action("navigation.plan_route", _plan_result("route://loading-a/agent"))
    context.queue_action("navigation.follow_route", _follow_result())
    context.queue_action("navigation.verify_arrival", _verify_result())

    _run(context)

    assert context.status == "completed"
    requests = [event for event in context.events if event.get("type") == "agent_requested"]
    assert len(requests) == 1
    assert requests[0]["context"]["stage"] == "plan_route"


def test_invalid_target_does_not_repeat_the_same_route_request() -> None:
    context = MockSkillContext(_skill_input())
    context.queue_action(
        "navigation.plan_route",
        ActionResult(
            status="failed",
            error_code="PLANNING_FAILED",
            error_message="终点位于障碍物中",
            physical_effect="none",
        ),
    )
    context.queue_agent_reply({
        "action": "abort_subtask",
        "reason": "需要重新计算来源工位",
    })

    _run(context)

    assert context.status == "failed"
    assert _actions(context).count("navigation.plan_route") == 1
    requests = [event for event in context.events if event.get("type") == "agent_requested"]
    assert len(requests) == 1
    assert requests[0]["reason"] == "导航目标位姿不可用于路径规划，需要重新计算目标"
    assert requests[0]["context"]["last_action_error"]["code"] == "PLANNING_FAILED"


def test_on_stop_uses_declared_navigation_stop_action() -> None:
    context = MockSkillContext(_skill_input())
    context.request_stop()
    context.queue_action(
        "navigation.follow_route",
        ActionResult(
            status="succeeded",
            output={"safe": True, "physical_state": "base_stopped_and_braked"},
            evidence_refs=["artifact://navigation/stop-proof"],
            physical_effect="confirmed",
        ),
    )

    outcome = asyncio.run(
        on_stop(
            context,
            StopRequest(id="stop-navigation", source="user", reason="用户停止", mode="immediate"),
        )
    )

    assert outcome.safe is True
    assert outcome.physical_state == "base_stopped_and_braked"
    assert _actions(context) == ["navigation.follow_route"]
