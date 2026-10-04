import asyncio
import pytest
from semantic_robot_skill_sdk import ActionResult
from semantic_robot_skill_sdk.mock import MockSkillContext
from semantic_robot_skills.skills.behavior_grasp.scripts import carry_motion
from semantic_robot_skills.skills.behavior_grasp.scripts.models import Input, State, Pose, GraspPlan
from semantic_robot_skills.skills.behavior_grasp.scripts.runtime import ActionFailed


def build_context(tmp_path):
    inputs = Input(object_ref='can', prompt='can', side='right', grasp_offset_m=[0,0,0],
                   opening_m=.05, maximum_force_n=50)
    pose = Pose(frame_id='body', position_m=[.6,-.2,.8], orientation_xyzw=[0,0,0,1], revision='test')
    state = State(carry_mode='direct', carry_pose=pose,
                  plan=GraspPlan(clearance=pose, pregrasp=pose, grasp=pose, lift=pose, candidate_id='test'),
                  liftoff_target=dict(frame_id='odom', position_m=[.6,-.2,.2],
                     orientation_xyzw=[0,0,0,1], revision='saved'))
    ctx=MockSkillContext(inputs, workspace=tmp_path)
    return ctx,state,inputs


async def no_measurement(*args):
    raise AssertionError('Saved geometry must not be measured again')


def run(ctx,state,inputs):
    return asyncio.run(carry_motion.carry(ctx,state,inputs,no_measurement))


def record(ctx):
    calls=[];original=ctx.execute
    async def execute(key,action):
        assert action.type == 'motion.move_end_effector', 'No holding-state query'
        calls.append((key,action.parameters))
        return await original(key,action)
    ctx.execute=execute
    return calls


def test_direct_success_has_no_lift_or_holding_query_and_resumes_without_replay(tmp_path):
    ctx,state,inputs=build_context(tmp_path);calls=record(ctx)
    for _ in range(2): ctx.queue_action('motion.move_end_effector',ActionResult(status='succeeded'))
    run(ctx,state,inputs)
    assert [k for k,_ in calls]==['grasp:carry-direct-plan','grasp:carry-direct']
    assert calls[0][1]['plan_only'] and not calls[1][1]['plan_only']
    assert all(p['allow_support_contact'] for _,p in calls)
    run(ctx,State.model_validate(state.model_dump()),inputs)
    assert len(calls)==2
    assert state.liftoff_verification is None


def test_only_preplan_failure_falls_back_once_and_persists_choice(tmp_path):
    ctx,state,inputs=build_context(tmp_path);calls=record(ctx)
    ctx.queue_action('motion.move_end_effector',ActionResult(status='failed',error_code='support_departure_unavailable'))
    for _ in range(2):ctx.queue_action('motion.move_end_effector',ActionResult(status='succeeded'))
    run(ctx,state,inputs)
    assert [k for k,_ in calls]==['grasp:carry-direct-plan','grasp:liftoff-ik','grasp:carry-curobo']
    assert state.carry_mode=='lift_first'
    assert state.carry_fallback_reason=='support_departure_unavailable'
    assert 'allow_support_contact' not in calls[-1][1]
    run(ctx,State.model_validate(state.model_dump()),inputs)
    assert len(calls)==3


@pytest.mark.parametrize('phase,status,code', [
    ('plan','stopped','cancelled'), ('plan','interrupted','scene_changed'),
    ('plan','failed','planning_timeout'), ('plan','failed','cuda_error'),
    ('execute','failed','collision_plan_failed'), ('execute','stopped','cancelled')])
def test_uncertain_planning_or_any_motion_failure_cannot_trigger_lift(tmp_path,phase,status,code):
    ctx,state,inputs=build_context(tmp_path);calls=record(ctx)
    if phase=='execute':ctx.queue_action('motion.move_end_effector',ActionResult(status='succeeded'))
    ctx.queue_action('motion.move_end_effector',ActionResult(status=status,error_code=code))
    with pytest.raises(ActionFailed):run(ctx,state,inputs)
    assert all('liftoff' not in key for key,_ in calls)
    assert state.carry_mode=='direct'


def test_old_lift_checkpoint_stays_on_original_route(tmp_path):
    ctx,state,inputs=build_context(tmp_path);state.carry_mode=None;calls=record(ctx)
    for _ in range(2):ctx.queue_action('motion.move_end_effector',ActionResult(status='succeeded'))
    run(ctx,state,inputs)
    assert [key for key,_ in calls]==['grasp:liftoff-ik','grasp:carry-curobo']
