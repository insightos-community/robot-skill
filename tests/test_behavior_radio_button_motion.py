"""Button contact uses IK without changing preparation or checkpoint semantics."""
import asyncio

import pytest

from semantic_robot_skill_sdk import ActionResult
from semantic_robot_skill_sdk.mock import MockSkillContext
from semantic_robot_skills.skills.behavior_radio_button.scripts import skill
from semantic_robot_skills.skills.behavior_radio_button.scripts.models import (
    Input, LocatedButton, MotionPlan, Pose, State,
)


@pytest.mark.parametrize("status", ["succeeded", "failed", "stopped"])
def test_button_uses_ik_and_preserves_terminal_result(status, monkeypatch, tmp_path):
    pose = Pose(position_m=[.7, -.06, .9], orientation_xyzw=[0, 0, 0, 1], revision="button")
    button = LocatedButton(object_ref="radio-button", pose=pose,
                           extent_m=[.02, .02, .02], identity_confidence=.9)
    state = State(hand_selection={"holding_side": "right", "operating_side": "left"},
                  button=button, approach_pose=pose, recognition_attempts=1,
                  plans={key: MotionPlan(side=side, pose=pose) for key, side in [
                      ("radio:natural", "right"), ("radio:retract", "right"),
                      ("radio:approach-button", "left")]})
    ctx = MockSkillContext(Input(), workspace=tmp_path)
    ctx.checkpoint(state)
    reads = []

    async def read_pose(ctx, state, inputs, prefix, side):
        reads.append(prefix)
        return pose

    async def search_button(ctx, state, inputs):
        return button, pose

    monkeypatch.setattr(skill, "read_pose", read_pose)
    monkeypatch.setattr(skill, "search_button", search_button)
    for _ in range(2):
        ctx.queue_action("motion.move_end_effector", ActionResult(status="succeeded"))
    ctx.queue_action("gripper.set_opening", ActionResult(status="succeeded", output={"verified": True}))
    ctx.queue_action("motion.move_end_effector", ActionResult(
        status=status, error_code=None if status == "succeeded" else "TEST_TERMINAL"))
    sent = []
    original = ctx.execute

    async def record(key, action):
        sent.append((key, action))
        return await original(key=key, action=action)

    ctx.execute = record
    asyncio.run(skill.run(ctx))
    motions = [(key, action.parameters) for key, action in sent
               if action.type == "motion.move_end_effector"]
    assert [p["motion_mode"] for _, p in motions] == ["curobo", "curobo", "ik"]
    assert all(p["use_torso"] is False for _, p in motions[:2])
    key, final = motions[-1]
    assert key == "radio:approach-button:eef"
    assert final["axis"] == "left" and final["target"] == pose.model_dump(mode="json")
    assert final["position_tolerance_m"] == .003
    assert "use_torso" not in final and "contact_object_ref" not in final
    assert ctx.state.results[key].status == status
    if status == "succeeded":
        assert ctx.result.button == button
        assert "radio:approach-button:after" in reads
    else:
        assert ctx.failure.code == "TEST_TERMINAL"
        assert ctx.result is None
        assert "radio:approach-button:after" not in reads
    # A restored checkpoint must never replay a completed or failed movement.
    sent.clear()
    asyncio.run(skill.run(ctx))
    assert not sent
