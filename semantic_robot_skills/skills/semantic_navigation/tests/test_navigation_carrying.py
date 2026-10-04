"""语义导航Skill的实时携物约束与滑移终止验收测试。"""

from __future__ import annotations

import asyncio

from semantic_robot_skill_sdk import (
    ActionFeedback, ActionResult, MockSkillContext, Observation, Pose3D, run_skill,
)
from semantic_robot_skills.skills.semantic_navigation.scripts.controller import (
    build_follow_route_action, build_plan_route_action,
)
from semantic_robot_skills.skills.semantic_navigation.scripts.models import (
    CarryingObjectState, ResolvedNavigationTarget, SemanticNavigationInput,
    SemanticNavigationResult, SemanticNavigationState,
)
from semantic_robot_skills.skills.semantic_navigation.scripts.skill import run


OBJECT_REF = "semantic://box/17"
TOOL_REFS = ("component://tool/left", "component://tool/right")


def _pose(revision: str) -> Pose3D:
    return Pose3D(frame_id="map", position_m=(0.55, 0.0, 0.72), revision=revision)


def _carrying_state(*, generation: int) -> CarryingObjectState:
    pose = _pose(f"scene-{generation}")
    return CarryingObjectState(
        object_ref=OBJECT_REF,
        robot_ref="robot://demo-1",
        tool_refs=TOOL_REFS,
        tool_poses={ref: pose for ref in TOOL_REFS},
        object_pose=pose,
        object_size_m=(0.8, 0.4, 0.3),
        robot_state_generation=generation,
    )


def _carrying_input() -> SemanticNavigationInput:
    return SemanticNavigationInput(
        target=ResolvedNavigationTarget(
            target_ref="semantic://station/loading-a",
            pose=Pose3D(frame_id="map", position_m=(2.0, 1.0, 0.0), revision="target-r1"),
        ),
        navigation_purpose="carry_to_place",
        carried_object_ref=OBJECT_REF,
        arrival_radius_m=0.4,
        maximum_speed_mps=0.8,
        minimum_clearance_m=0.25,
    )


def _robot_state(generation: int) -> Observation:
    return Observation(
        kind="robot.state",
        subject_ref="robot://demo-1",
        source="pilot.robot-state",
        revision=f"robot-{generation}",
        value={
            "robot_id": "robot://demo-1",
            "generation": generation,
            "end_effectors": {
                "left": {"frame_id": "map", "position": [0.55, 0.22, 0.72], "quaternion_xyzw": [0, 0, 0, 1]},
                "right": {"frame_id": "map", "position": [0.55, -0.22, 0.72], "quaternion_xyzw": [0, 0, 0, 1]},
            },
            "tool_states": {
                TOOL_REFS[0]: {"tool_ref": TOOL_REFS[0], "side": "left", "kind": "tote_clamp", "hook_contact": True, "clamp_contact": True},
                TOOL_REFS[1]: {"tool_ref": TOOL_REFS[1], "side": "right", "kind": "tote_clamp", "hook_contact": True, "clamp_contact": True},
            },
        },
    )


def _tool_load(*, stable: bool = True) -> Observation:
    tools = [
        {
            "tool_ref": ref, "available": True, "position": 0.015, "velocity": 0.0,
            "effort": 18.0, "hook_contact": True, "clamp_contact": True,
            "hook_force_n": 12.0, "clamp_force_n": 18.0, "hook_support_ratio": 0.8,
            "relative_tangential_speed_mps": 0.0, "sensor_fault": False,
        }
        for ref in TOOL_REFS
    ]
    return Observation(
        kind="robot.tool_load", subject_ref="robot://demo-1", source="pilot.robot-state",
        value={
            "condition_satisfied": stable, "observed_duration_ms": 150, "tools": tools,
            "slip_detected": not stable, "overload_detected": False,
            "sensor_fault": False, "reasons": [] if stable else ["slipping"],
        },
    )


def _object_pose(revision: str) -> Observation:
    pose = _pose(revision)
    return Observation(
        kind="target_pose", subject_ref=OBJECT_REF, source="perception",
        revision=revision,
        value={
            "object_ref": OBJECT_REF,
            "pose": pose.model_dump(mode="json"),
            "extent_m": [0.8, 0.4, 0.3],
            "identity_confidence": 0.98,
        },
    )


def _queue_carrying_observation(context: MockSkillContext, generation: int) -> None:
    context.queue_action("robot.get_state", ActionResult(status="succeeded", observations=[_robot_state(generation)], physical_effect="none"))
    context.queue_action("robot.verify_tool_load", ActionResult(status="succeeded", observations=[_tool_load()], physical_effect="none"))
    context.queue_action("perception.locate_object", ActionResult(status="succeeded", observations=[_object_pose(f"scene-{generation}")], physical_effect="none"))


def _actions(context: MockSkillContext) -> list[str]:
    return [event["action"] for event in context.events if event.get("type") == "action_started"]


def test_carrying_navigation_limits_motion_and_reobserves_at_arrival() -> None:
    skill_input = _carrying_input()
    planned_state = SemanticNavigationState(
        target=skill_input.target,
        route_ref="route://loading-a/carrying-1",
        carrying_object=_carrying_state(generation=7),
    )
    for action in (build_plan_route_action(skill_input, planned_state), build_follow_route_action(skill_input, planned_state)):
        assert action.parameters["maximum_speed_mps"] == 0.05
        assert action.parameters["minimum_clearance_m"] == 0.5
    planned_state.replan_count = 1
    for action in (
        build_plan_route_action(skill_input, planned_state),
        build_follow_route_action(skill_input, planned_state),
    ):
        assert action.parameters["maximum_speed_mps"] == 0.025

    context = MockSkillContext(skill_input)
    _queue_carrying_observation(context, 7)
    context.queue_action("navigation.plan_route", ActionResult(status="succeeded", output={"route_ref": "route://loading-a/carrying-1"}, physical_effect="none"))
    context.queue_action("navigation.follow_route", ActionResult(status="succeeded", output={"final_pose_ref": "pose://robot/loading-a", "distance_to_target_m": 0.16}, physical_effect="confirmed"))
    context.queue_action("navigation.verify_arrival", ActionResult(status="succeeded", output={"verdict": "achieved", "final_pose_ref": "pose://robot/loading-a", "distance_to_target_m": 0.16, "carrying_object": None}, physical_effect="none"))
    _queue_carrying_observation(context, 8)

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    assert isinstance(context.result, SemanticNavigationResult)
    assert not hasattr(context.result, "carrying_object")
    assert context.state.carrying_object.robot_state_generation == 8
    assert _actions(context).count("robot.get_state") == 2
    assert _actions(context).count("robot.verify_tool_load") == 2
    assert _actions(context).count("perception.locate_object") == 2


def test_carrying_stop_rechecks_live_load_and_replans_once_at_lower_speed() -> None:
    context = MockSkillContext(_carrying_input())
    _queue_carrying_observation(context, 7)
    context.queue_action(
        "navigation.plan_route",
        ActionResult(status="succeeded", output={"route_ref": "route://loading-a/carrying-1"}, physical_effect="none"),
    )
    context.queue_action(
        "navigation.follow_route",
        ActionResult(
            status="stopped",
            output={"stop_evidence": {"command_status": "stopped", "holding": True, "reason": "carrying_load_unstable"}},
            physical_effect="confirmed",
        ),
    )
    _queue_carrying_observation(context, 8)
    context.queue_action(
        "navigation.plan_route",
        ActionResult(status="succeeded", output={"route_ref": "route://loading-a/carrying-2"}, physical_effect="none"),
    )
    context.queue_action(
        "navigation.follow_route",
        ActionResult(status="succeeded", output={"final_pose_ref": "pose://robot/loading-a", "distance_to_target_m": 0.12}, physical_effect="confirmed"),
    )
    context.queue_action(
        "navigation.verify_arrival",
        ActionResult(status="succeeded", output={"verdict": "achieved", "final_pose_ref": "pose://robot/loading-a", "distance_to_target_m": 0.12, "carrying_object": None}, physical_effect="none"),
    )
    _queue_carrying_observation(context, 9)

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    assert context.state.replan_count == 1
    assert _actions(context).count("navigation.plan_route") == 2
    follow_actions = [event for event in context.events if event.get("type") == "action_started" and event.get("action") == "navigation.follow_route"]
    assert len(follow_actions) == 2
    assert follow_actions[-1]["parameters"]["maximum_speed_mps"] == 0.025
    assert not any(event.get("type") == "agent_requested" for event in context.events)


def test_repeated_carrying_slip_fails_without_blind_agent_retry() -> None:
    context = MockSkillContext(_carrying_input())
    _queue_carrying_observation(context, 7)
    context.queue_action("navigation.plan_route", ActionResult(status="succeeded", output={"route_ref": "route://loading-a/carrying-1"}, physical_effect="none"))
    stopped_output = {"stop_evidence": {"command_status": "stopped", "holding": True, "reason": "carrying_load_unstable"}}
    context.queue_action(
        "navigation.follow_route",
        ActionResult(status="stopped", output=stopped_output, error_code="CARRYING_OBJECT_SLIPPED", physical_effect="confirmed"),
        feedback=[ActionFeedback(
            sequence=1, status="stopped", progress=0.42, severity="critical",
            observations=[Observation(
                kind="navigation.carrying_load", subject_ref=OBJECT_REF,
                source="navigation", value={
                    "stable": False, "slipping": True, "overloaded": False,
                    "sensor_fault": False,
                    "reasons": [f"{TOOL_REFS[1]}:relative_motion_high"],
                },
            )],
        )],
    )
    _queue_carrying_observation(context, 8)
    context.queue_action("navigation.plan_route", ActionResult(status="succeeded", output={"route_ref": "route://loading-a/carrying-2"}, physical_effect="none"))
    context.queue_action(
        "navigation.follow_route",
        ActionResult(status="stopped", output=stopped_output, error_code="CARRYING_OBJECT_SLIPPED", physical_effect="confirmed"),
        feedback=[ActionFeedback(
            sequence=2, status="stopped", progress=0.51, severity="critical",
            observations=[Observation(
                kind="navigation.carrying_load", subject_ref=OBJECT_REF,
                source="navigation", value={
                    "stable": False, "slipping": True, "overloaded": False,
                    "sensor_fault": False,
                    "reasons": [f"{TOOL_REFS[1]}:relative_motion_high"],
                },
            )],
        )],
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "failed"
    assert context.failure.code == "CARRYING_LOAD_UNSTABLE"
    assert "component://tool/right:relative_motion_high" in context.failure.message
    assert _actions(context).count("navigation.follow_route") == 2
    assert "navigation.verify_arrival" not in _actions(context)
    assert not any(event.get("type") == "agent_requested" for event in context.events)
