"""Initialization must straighten the arms, preserve torso, and stop on failure."""
import asyncio

import numpy as np
import pytest

from semantic_robot_skill_sdk import ActionResult, Observation
from semantic_robot_skill_sdk.mock import MockSkillContext
from semantic_robot_skills.skills.behavior_init.scripts import skill
from semantic_robot_skills.skills.behavior_init.scripts.geometry import prepare_arms
from semantic_robot_skills.skills.behavior_init.scripts.kinematics import ArmModel
from semantic_robot_skills.skills.behavior_init.scripts.models import Input, JointSample


def sample():
    names = [f"torso_joint{i}" for i in range(1, 5)]
    names += [f"{side}_arm_joint{i}" for side in ("left", "right") for i in range(1, 8)]
    return JointSample(names=names, positions_rad=[1.025, -1.45, -.47, 0] + [.3] * 14)


def test_targets_are_straight_down_instead_of_raised():
    joints = sample()
    pitch, targets = prepare_arms(joints)
    assert pitch == pytest.approx(.045)
    for target in targets:
        assert target.positions_rad == [0] * 7
        transform = ArmModel(target.side, joints.positions_rad[:4]).forward(target.positions_rad)
        # Tool axis points down and the hand is below the waist, not raised ahead.
        assert np.dot(transform[:3, 2], [0, 0, -1]) > .99
        assert target.pose.position_m[2] < .5
        assert abs(target.pose.position_m[0]) < .2
        assert all(name.startswith(target.side + '_arm_joint') for name in target.joint_names)


@pytest.mark.parametrize('status', ['succeeded', 'failed', 'stopped'])
def test_executes_joint_targets_preserving_torso_and_terminal_checkpoints(status, tmp_path):
    ctx = MockSkillContext(Input(), workspace=tmp_path)
    ctx.queue_action('robot.get_state', ActionResult(status='succeeded', observations=[
        Observation(kind='robot.state', source='test', value={'joints': sample().model_dump()})]))
    ctx.queue_action('motion.move_arm_joint', ActionResult(
        status=status, error_code=None if status == 'succeeded' else 'TEST_TERMINAL'))
    if status == 'succeeded':
        ctx.queue_action('motion.move_arm_joint', ActionResult(status='succeeded'))
    sent = []
    original = ctx.execute

    async def record(key, action):
        sent.append((key, action))
        return await original(key=key, action=action)

    ctx.execute = record
    asyncio.run(skill.run(ctx))
    motions = [(key, action) for key, action in sent if action.type != 'robot.get_state']
    assert [key for key, _ in motions] == (['init:left', 'init:right'] if status == 'succeeded' else ['init:left'])
    for _, action in motions:
        assert action.type == 'motion.move_arm_joint'
        assert action.parameters['positions_rad'] == [0] * 7
        assert len(action.parameters['joint_names']) == 7
        assert not any('torso' in n or 'gripper' in n for n in action.parameters['joint_names'])
    if status == 'succeeded':
        assert ctx.result is not None
    else:
        assert ctx.failure.code == 'TEST_TERMINAL'
        assert ctx.result is None
    sent.clear()
    asyncio.run(skill.run(ctx))
    assert sent == []


def test_missing_joint_rejects_before_motion(tmp_path):
    joints = sample()
    joints.names.pop()
    joints.positions_rad.pop()
    ctx = MockSkillContext(Input(), workspace=tmp_path)
    ctx.queue_action('robot.get_state', ActionResult(status='succeeded', observations=[
        Observation(kind='robot.state', source='test', value={'joints': joints.model_dump()})]))
    asyncio.run(skill.run(ctx))
    assert ctx.failure.code == 'INIT_STATE_INVALID'
    assert ctx.result is None
