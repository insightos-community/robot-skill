"""Grasp-to-place geometry: frame changes, durable capture and result handoff."""
import asyncio
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from semantic_robot_skill_sdk import ActionResult, Observation
from semantic_robot_skill_sdk.mock import MockSkillContext
from semantic_robot_skills.skills.behavior_grasp.scripts import skill, curobo
from semantic_robot_skills.skills.behavior_grasp.scripts.enclosure import transform
from semantic_robot_skills.skills.behavior_grasp.scripts.held_geometry import observation_anchor, held_geometry, result_fields
from semantic_robot_skills.skills.behavior_grasp.scripts.liftoff import measured_pose
from semantic_robot_skills.skills.behavior_grasp.scripts.models import Input, State, Pose, GraspPlan, LocatedObject, Result
from semantic_robot_skills.skills.behavior_place.scripts.models import Input as PlaceInput
from semantic_robot_skills.skills.behavior_place.scripts import skill as place_skill
from semantic_robot_skills.skills.behavior_place.scripts.holding_side import SAMPLE_COUNT

TIME = datetime(2026, 9, 25, tzinfo=timezone.utc)


def pose(position=(.6, .2, .1), yaw=0.):
    return Pose(frame_id='body', position_m=list(position),
                orientation_xyzw=Rotation.from_euler('z', yaw).as_quat().tolist(),
                revision='rgbd-frame-1', observed_at=(TIME + timedelta(seconds=1)).isoformat())


def detection():
    return LocatedObject(object_ref='can', pose=pose(), extent_m=[.13, .07, .06], identity_confidence=.9)


def base_result(seconds, position=(0., 0., 0.), yaw=0., frame='odom'):
    value=dict(base_pose=dict(frame_id=frame, position=list(position),
                             orientation_xyzw=Rotation.from_euler('z', yaw).as_quat().tolist()),
               joints=dict(names=[f'torso_joint{i}' for i in range(1, 5)], positions_rad=[0., 0., 0., 0.]))
    return ActionResult(status='succeeded', observations=[Observation(kind='robot.state', source='test',
                       observed_at=TIME+timedelta(seconds=seconds), value=value)])


def anchor():
    return observation_anchor(detection(), base_result(0), base_result(2))


def test_anchor_interpolates_small_drift_at_rgbd_timestamp():
    value = observation_anchor(detection(), base_result(0), base_result(2, (.0004, 0., 0.), .0002))
    assert value.fixed_from_body.position_m == pytest.approx([.0002, 0, 0])
    assert Rotation.from_quat(value.fixed_from_body.orientation_xyzw).as_euler('xyz')[2] == pytest.approx(.0001)


@pytest.mark.parametrize('after', [base_result(2, (.02, 0, 0)), base_result(2, yaw=.01), base_result(2, frame='world'), base_result(.5)])
def test_anchor_rejects_motion_frame_change_or_stale_image(after):
    with pytest.raises(ValueError):
        observation_anchor(detection(), base_result(0), after)


def test_measured_closed_eef_and_moved_base_preserve_object_box_axes():
    located = detection()
    # Base has moved and rotated AFTER recognition; measured EEF is in torso frame.
    closed = base_result(3, (1., 2., 0.), np.pi/2).observations[0].value
    entry = dict(axis='left', frame_id='torso_link4', position_m=[.2, .3, .1],
                 orientation_xyzw=Rotation.from_euler('xyz', [.1, .2, .3]).as_quat().tolist())
    measured = measured_pose({'end_effector':entry}, closed, 'left')
    geometry = held_geometry(located, anchor(), measured, 'can')
    assert geometry.object_size_m == located.extent_m
    # The anchor is identity in odom; closing base motion must not move the old OBB.
    assert np.allclose(transform(measured) @ transform(geometry.eef_from_object.model_dump()), transform(located.pose.model_dump()))
    naive = np.linalg.inv(transform(entry)) @ transform(located.pose.model_dump())
    assert not np.allclose(naive, transform(geometry.eef_from_object.model_dump()))


@pytest.mark.parametrize('ref,frame', [('another-can', 'odom'), ('can', 'world')])
def test_geometry_rejects_wrong_object_or_fixed_frame(ref, frame):
    with pytest.raises(ValueError):
        held_geometry(detection(), anchor(), dict(frame_id=frame, position_m=[0., 0., 0.], orientation_xyzw=[0.,0.,0.,1.]), ref)


def test_legacy_checkpoint_does_not_invent_geometry():
    restored = State.model_validate({'stage':'completed', 'located':detection().model_dump()})
    assert result_fields(restored) == dict(object_size_m=None, eef_from_object=None, held_geometry=None)


def test_grasp_captures_before_lift_and_returns_geometry_on_resume(monkeypatch, tmp_path):
    inputs = Input(object_ref='can', prompt='can', side='left', orientation_xyzw=[0,0,0,1],
                   grasp_offset_m=[0,0,0], opening_m=.05, maximum_force_n=50)
    ctx = MockSkillContext(inputs, workspace=tmp_path)
    goal = pose()
    plan = GraspPlan(clearance=goal, pregrasp=goal, grasp=goal, lift=goal, candidate_id='test')
    ctx.checkpoint(State(reference_torso=[0,0,0,0], torso_targets=[], torso_durations=[],
                         plan=plan, carry_pose=goal))
    async def approach(ctx, state, inputs, read_state):
        state.selected_side='left'
        state.plan=plan
        return inputs
    monkeypatch.setattr(curobo, 'approach', approach)
    for seconds in (0, 2, 3):
        ctx.queue_action('robot.get_state', base_result(seconds))
    ctx.queue_action('perception.locate_object', ActionResult(status='succeeded', output={'verification':detection().model_dump()}))
    ctx.queue_action('gripper.close', ActionResult(status='succeeded', output={'contact':True,'object_ref':'can','candidate_id':'test'}))
    for z in (.1,):
        ctx.queue_action('motion.get_end_effector_state', ActionResult(status='succeeded', output={
            'end_effector':dict(axis='left', frame_id='body', position_m=[.6,.2,z], orientation_xyzw=[0,0,0,1], observed_at=(TIME+timedelta(seconds=3)).isoformat())}))
    for _ in range(2):
        ctx.queue_action('motion.move_end_effector', ActionResult(status='succeeded'))
    ctx.queue_action('gripper.get_state', ActionResult(status='succeeded', output={'tools':[{'held_object_ref':'can'}]}))
    execute=ctx.execute
    async def assert_saved(key, action):
        assert not key.startswith('grasp:liftoff-after')
        if key=='grasp:liftoff-ik':
            assert ctx.state.held_geometry is not None
            assert ctx.state.held_geometry.eef_from_object.position_m == pytest.approx([0,0,0])
        return await execute(key,action)
    ctx.execute=assert_saved
    asyncio.run(skill.run(ctx))
    assert ctx.status=='completed',ctx.failure
    assert ctx.result.object_size_m==[.13,.07,.06]
    assert ctx.result.eef_from_object.position_m==pytest.approx([0,0,0])
    saved=ctx.result.model_dump(mode='json')
    # Successful state is durable and completion replay must not execute actions.
    async def no_actions(*args,**kwargs):raise AssertionError('unexpected action on completed checkpoint')
    ctx.execute=no_actions
    asyncio.run(skill.run(ctx))
    assert ctx.result.model_dump(mode='json')==saved
    assert Result.model_validate_json(ctx.result.model_dump_json()).object_size_m==[.13,.07,.06]
    # Result is directly compatible with the existing placement contract.
    place=PlaceInput(object_ref='can',side=saved['grasp_side'],object_size_m=saved['object_size_m'],
        eef_from_object=saved['eef_from_object'],target={'object_ref':'bin','type':'top_open_container','prompt':'trash can'},
        release_mode='drop',release_opening_m=.05,maximum_force_n=50)
    place_ctx=MockSkillContext(place,workspace=tmp_path/'place')
    for _ in range(SAMPLE_COUNT):
        place_ctx.queue_action('gripper.get_state',ActionResult(status='succeeded',output={'tools':[
            {'side':'left','held_object_ref':'can'},{'side':'right','held_object_ref':None}]}))
    place_ctx.queue_action('motion.get_end_effector_state',ActionResult(status='succeeded',observations=[
        Observation(kind='manipulation.end_effector_state',source='test',value={
            'axis':'left','frame_id':'body','position_m':[.6,.2,.5],'orientation_xyzw':[0,0,0,1]})]))
    executed=[]
    original=place_ctx.execute
    async def check_target(key,action):
        executed.append(key)
        if key=='place:locate-target':
            assert action.parameters['object_ref']=='bin'
            return ActionResult(status='failed',error_code='TEST_TARGET_REACHED',error_message='Reached target recognition')
        assert key!='place:locate-held'
        return await original(key,action)
    place_ctx.execute=check_target
    monkeypatch.setattr(place_skill,'SAMPLE_INTERVAL_SECONDS',0)
    asyncio.run(place_skill.run(place_ctx))
    assert 'place:locate-target' in executed
    assert 'place:locate-held' not in executed
    assert place_ctx.state.object_geometry.source=='task_input'
