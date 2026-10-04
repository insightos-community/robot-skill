"""低频采图不改变动作判定、停止语义或 Stage 生命周期。"""

from __future__ import annotations

import asyncio
import importlib
import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, Field

from semantic_robot_skill_sdk import (
    ActionResult, MockSkillContext, Observation, SkillCancelled, run_skill,
)


class EvidenceState(BaseModel):
    stage: str = "verify"
    evidence_refs: list[str] = Field(default_factory=lambda: ["artifact://existing-proof"])
    physical_state: str = "unchanged"


@pytest.fixture(params=["grasp_object", "place_object", "semantic_navigation"])
def capture(request):
    return importlib.import_module(
        f"semantic_robot_skills.skills.{request.param}.scripts.stage_evidence"
    ).capture_stage_rgb


def _context() -> MockSkillContext:
    context = MockSkillContext(EvidenceState())
    context.execution_id = "execution-stage-rgb-1"
    return context


def _frame(ref: str = "pilot-artifact://pilot-1/frame-1") -> ActionResult:
    return ActionResult(
        status="succeeded", physical_effect="none",
        observations=[Observation(
            kind="sensor.frame", source="robot-sdk://demo-1/sensor/camera.rgb",
            observed_at=datetime(2026, 9, 7, 1, 2, 3, tzinfo=timezone.utc),
            value={
                "sensor_id": "camera.rgb", "media_type": "image/png",
                "observed_at": "2026-09-07T01:02:03+00:00",
            },
            evidence_refs=[ref, ref] if ref else [],
        )],
    )


def _started(context):
    return [event for event in context.events if event["type"] == "action_started"]


def test_one_frame_per_execution_stage_point_and_no_lifecycle_status(capture) -> None:
    context = _context()
    state = EvidenceState()
    context.queue_action("sensor.capture_rgbd", _frame())
    context.queue_action("sensor.capture_rgbd", _frame("pilot-artifact://pilot-1/frame-2"))
    context.queue_action("sensor.capture_rgbd", _frame("pilot-artifact://pilot-1/frame-3"))

    async def exercise():
        await capture(context, state, skill_name="test-skill")
        await capture(context, state, skill_name="test-skill")  # Resume: same frame/action.
        await capture(context, state, skill_name="test-skill", point="completed")
        state.stage = "next-stage"
        await capture(context, state, skill_name="test-skill")

    asyncio.run(exercise())
    started = _started(context)
    identities = [json.loads(event["key"].removeprefix("stage-rgb:")) for event in started]
    assert identities == [
        [context.execution_id, "test-skill", "verify", "entry"],
        [context.execution_id, "test-skill", "verify", "completed"],
        [context.execution_id, "test-skill", "next-stage", "entry"],
    ]
    assert all(event["parameters"] == {"sensor_ids": ["camera.rgb"]} for event in started)
    assert context.events[0]["type"] == "checkpoint"
    assert context.events[0]["stage"] == "verify"
    assert state.physical_state == "unchanged"
    assert state.evidence_refs == [
        "artifact://existing-proof", "pilot-artifact://pilot-1/frame-1",
        "pilot-artifact://pilot-1/frame-2", "pilot-artifact://pilot-1/frame-3",
    ]
    reports = [event for event in context.events if event["type"] == "stage.evidence"]
    assert len(reports) == 4
    assert all(event["stage_status"] is None for event in reports)
    assert all("2026-09-07T01:02:03+00:00" in event["observation_summary"] for event in reports)
    assert all("robot-sdk://demo-1/sensor/camera.rgb" in event["observation_summary"] for event in reports)
    assert context.status == "running" and context.result is None


@pytest.mark.parametrize("result", [
    ActionResult(status="failed", error_code="ARTIFACT_EXCHANGE_UNAVAILABLE", physical_effect="none"),
    ActionResult(status="interrupted", error_code="TIMEOUT", physical_effect="none"),
    _frame(""),
    _frame("file:///unpublished.png"),
    ActionResult(status="succeeded", output={"artifact_refs": ["artifact://without-frame"]}),
])
def test_missing_capture_or_publication_is_visible_but_not_motion_failure(capture, result) -> None:
    context = _context()
    state = EvidenceState()
    context.queue_action("sensor.capture_rgbd", result)
    asyncio.run(capture(context, state, skill_name="test-skill"))
    assert context.status == "running" and context.failure is None and context.result is None
    assert state.physical_state == "unchanged"
    assert state.evidence_refs == ["artifact://existing-proof"]
    report = context.events[-1]
    assert report["type"] == "stage.evidence_unavailable"
    assert report["stage"] == "verify" and report["stage_status"] is None
    assert report["deviation"] and not report["evidence_refs"]


def test_capture_transport_exception_is_nonfatal(capture) -> None:
    context = _context()

    async def unavailable(*args):
        raise TimeoutError("not serialized into user-facing telemetry")

    context.execute = unavailable
    asyncio.run(capture(context, EvidenceState(), skill_name="test-skill"))
    assert context.events[-1]["type"] == "stage.evidence_unavailable"
    assert "TimeoutError" in context.events[-1]["deviation"]
    assert context.status == "running"


@pytest.mark.parametrize("error", [SkillCancelled("stop requested"), asyncio.CancelledError()])
def test_capture_never_swallows_stop_or_cancellation(capture, error) -> None:
    context = _context()

    async def cancelled(*args):
        raise error

    context.execute = cancelled
    with pytest.raises(type(error)):
        asyncio.run(capture(context, EvidenceState(), skill_name="test-skill"))
    assert not any(event["type"].startswith("stage.evidence") for event in context.events)


def test_missing_execution_identity_is_reported_without_global_capture_key(capture) -> None:
    context = MockSkillContext(EvidenceState())
    asyncio.run(capture(context, EvidenceState(), skill_name="test-skill"))
    assert not _started(context)
    assert context.events[-1]["type"] == "stage.evidence_unavailable"


@pytest.mark.parametrize("capture_succeeds", [True, False])
def test_navigation_completion_keeps_motion_sequence_and_merges_real_rgb_refs(capture_succeeds) -> None:
    from semantic_robot_skill_sdk import Pose3D
    from semantic_robot_skills.skills.semantic_navigation.scripts.models import (
        ResolvedNavigationTarget, SemanticNavigationInput,
    )
    from semantic_robot_skills.skills.semantic_navigation.scripts.skill import run

    context = MockSkillContext(SemanticNavigationInput(
        target=ResolvedNavigationTarget(
            target_ref="region://target", pose=Pose3D(frame_id="world", position_m=(1, 2, 0)),
        ),
        arrival_radius_m=0.2,
    ))
    context.execution_id = "execution-navigation-capture"
    refs = [f"pilot-artifact://pilot-1/navigation-{number}" for number in range(4)]
    for ref in refs:
        context.queue_action("sensor.capture_rgbd", _frame(ref) if capture_succeeds else ActionResult(
            status="failed", error_code="SENSOR_UNAVAILABLE", physical_effect="none",
        ))
    context.queue_action("navigation.plan_route", ActionResult(
        status="succeeded", output={"route_ref": "route://navigation"},
    ))
    context.queue_action("navigation.follow_route", ActionResult(
        status="succeeded", output={"final_pose_ref": "pose://final", "distance_to_target_m": 0.1},
    ))
    context.queue_action("navigation.verify_arrival", ActionResult(
        status="succeeded", output={
            "verdict": "achieved", "final_pose_ref": "pose://final", "distance_to_target_m": 0.1,
        }, evidence_refs=["artifact://independent-arrival"],
    ))
    asyncio.run(run_skill(run, context))
    assert context.status == "completed"
    assert [event["action"] for event in _started(context) if event["action"] != "sensor.capture_rgbd"] == [
        "navigation.plan_route", "navigation.follow_route", "navigation.verify_arrival",
    ]
    captures = [event for event in _started(context) if event["action"] == "sensor.capture_rgbd"]
    assert len(captures) == 4
    if capture_succeeds:
        assert set(refs) <= set(context.result.evidence_refs)
    else:
        assert not set(refs) & set(context.result.evidence_refs)
        assert sum(event["type"] == "stage.evidence_unavailable" for event in context.events) == 4
