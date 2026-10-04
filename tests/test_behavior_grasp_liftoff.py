"""Configurable lift targets and durable height interpretation across resumes."""
import asyncio

import pytest
from pydantic import ValidationError

from semantic_robot_skill_sdk import ActionResult
from semantic_robot_skill_sdk.mock import MockSkillContext
from semantic_robot_skills.skills.behavior_grasp.scripts import liftoff
from semantic_robot_skills.skills.behavior_grasp.scripts.runtime import ActionFailed
from semantic_robot_skills.skills.behavior_grasp.scripts.models import Input, State, Pose, GraspPlan


def inputs(**extra):
    return Input(object_ref='radio', prompt='red radio handle', side='left',
                 grasp_offset_m=[0, 0, 0], opening_m=.05, maximum_force_n=50, **extra)


def initial_state():
    pose = Pose(frame_id='body', position_m=[.6, .2, .5],
                orientation_xyzw=[0, 0, 0, 1], revision='test')
    return State(plan=GraspPlan(clearance=pose, pregrasp=pose, grasp=pose,
                               lift=pose, candidate_id='test'))


def eef(z):
    return ActionResult(status='succeeded', output={'end_effector': {
        'axis': 'left', 'frame_id': 'body', 'position_m': [.6, .2, z],
        'orientation_xyzw': [0, 0, 0, 1]}})


async def read_state(key):
    assert key == 'grasp:liftoff-before-state', 'No post-lift state measurement'
    return {'base_pose': {'frame_id': 'odom', 'position': [0, 0, 0],
                          'orientation_xyzw': [0, 0, 0, 1]}}


@pytest.mark.parametrize('height', [None, .03, .02])
def test_height_reaches_action_and_runtime_success_needs_no_second_measurement(height, tmp_path):
    config = inputs(**({} if height is None else {'liftoff_height_m': height}))
    expected = .01 if height is None else height
    ctx = MockSkillContext(config, workspace=tmp_path)
    state = initial_state()
    ctx.queue_action('motion.get_end_effector_state', eef(.5))
    ctx.queue_action('motion.move_end_effector', ActionResult(status='succeeded'))
    original = ctx.execute

    async def check_target(key, action):
        assert 'after' not in key, 'No post-lift EEF measurement'
        if key == 'grasp:liftoff-ik':
            assert action.parameters['target']['position_m'] == pytest.approx([.6, .2, .5 + expected])
            assert action.parameters['position_tolerance_m'] == .003
            assert ctx.state.liftoff_height_m == expected
        return await original(key, action)

    ctx.execute = check_target
    asyncio.run(liftoff.lift(ctx, state, config, read_state))
    assert state.liftoff_verification['confirmed']
    assert state.liftoff_verification['requested_lift_m'] == expected
    assert state.liftoff_verification['source'] == 'runtime_action_result'
    assert state.liftoff_verification['action_status'] == 'succeeded'
    assert 'lift_world_z_m' not in state.liftoff_verification
    assert 'measured_pose' not in state.liftoff_verification


@pytest.mark.parametrize('height', [0, -.01, float('nan'), float('inf'), -float('inf')])
def test_invalid_height_rejected(height):
    with pytest.raises(ValidationError):
        inputs(liftoff_height_m=height)


@pytest.mark.parametrize('legacy', [False, True])
def test_resume_preserves_saved_target_and_height_even_if_input_changes(legacy, tmp_path):
    saved_height = .01 if legacy else .03
    state = initial_state()
    state.liftoff_target = dict(frame_id='odom', position_m=[.6, .2, .5 + saved_height],
                               orientation_xyzw=[0, 0, 0, 1], revision='saved-liftoff')
    state.liftoff_height_m = saved_height
    data = state.model_dump(mode='json')
    if legacy:
        del data['liftoff_height_m']
    state = State.model_validate(data)
    config = inputs(liftoff_height_m=.05)
    ctx = MockSkillContext(config, workspace=tmp_path)
    ctx.queue_action('motion.move_end_effector', ActionResult(status='succeeded'))
    original = ctx.execute

    async def check_replay(key, action):
        assert key == 'grasp:liftoff-ik', 'Resume must not remeasure poses'
        if key == 'grasp:liftoff-ik':
            assert action.parameters['target']['position_m'][2] == .5 + saved_height
        return await original(key, action)

    ctx.execute = check_replay
    asyncio.run(liftoff.lift(ctx, state, config, read_state))
    assert state.liftoff_verification['requested_lift_m'] == saved_height
    assert state.liftoff_verification['confirmed']


@pytest.mark.parametrize('status', ['failed', 'stopped', 'interrupted'])
def test_unsuccessful_runtime_action_still_fails_lift(status, tmp_path):
    config = inputs(liftoff_height_m=.03)
    ctx = MockSkillContext(config, workspace=tmp_path)
    state = initial_state()
    ctx.queue_action('motion.get_end_effector_state', eef(.5))
    ctx.queue_action('motion.move_end_effector', ActionResult(
        status=status, error_code='command_timeout', error_message='Runtime did not complete'))
    with pytest.raises(ActionFailed):
        asyncio.run(liftoff.lift(ctx, state, config, read_state))
    assert state.liftoff_verification is None


def test_saved_runtime_success_is_not_replayed_or_remeasured(tmp_path):
    config = inputs(liftoff_height_m=.005)
    ctx = MockSkillContext(config, workspace=tmp_path)
    state = initial_state()
    state.liftoff_target = dict(frame_id='odom', position_m=[.6, .2, .505],
                               orientation_xyzw=[0, 0, 0, 1], revision='saved-liftoff')
    state.liftoff_height_m = .005
    state.results['grasp:liftoff-ik'] = ActionResult(status='succeeded')
    # Old post-motion measurements must not veto an already successful action.
    state.liftoff_verification = dict(confirmed=False, target_error_m=.0036126)

    async def no_actions(*args, **kwargs):
        raise AssertionError('Completed lift must not be replayed or remeasured')

    ctx.execute = no_actions
    asyncio.run(liftoff.lift(ctx, state, config, no_actions))
    assert state.liftoff_verification['confirmed']
    assert state.liftoff_verification['source'] == 'runtime_action_result'
    assert 'target_error_m' not in state.liftoff_verification
