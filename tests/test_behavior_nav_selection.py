import asyncio
import json
import pytest
from pydantic import ValidationError
from semantic_robot_skill_sdk import ActionResult
from semantic_robot_skill_sdk.mock import MockSkillContext
from semantic_robot_skills.skills.behavior_nav.scripts.models import Input
from semantic_robot_skills.skills.behavior_nav.scripts import skill


@pytest.mark.parametrize('value', [{}, {'object_name':'a','candidate_object_names':['b']},
    {'candidate_object_names':[]}, {'candidate_object_names':['a','a']}, {'candidate_object_names':[' ']}])
def test_exclusive_target_input(value):
    with pytest.raises(ValidationError): Input.model_validate(value)


@pytest.mark.parametrize('multiple', [False, True])
def test_selected_identity_and_diagnostics_survive_skill_result(tmp_path, multiple):
    inputs = Input(candidate_object_names=['blocked','chosen']) if multiple else Input(object_name='chosen')
    ctx = MockSkillContext(inputs, workspace=tmp_path)
    approach = dict(target=dict(frame_id='odom',position_m=[1.,2.,0.],yaw_rad=0.),
        object_name='chosen',map_id='map',instance_id='scene',run_id='run')
    selection = dict(policy='nearest_reachable',selected_object_name='chosen',map_id='map',
        candidates=[dict(object_name='blocked',status='failed',error_code='no_path'),
                    dict(object_name='chosen',status='succeeded',path_length_m=2.)])
    ctx.queue_action('navigation.follow_route', ActionResult(status='succeeded',
        output={'object_approach':approach, **({'selection':selection} if multiple else {})}))
    ctx.queue_action('navigation.verify_arrival', ActionResult(status='succeeded',
        output=dict(verdict='achieved',distance_to_target_m=.01,final_pose_ref='actual')))
    calls=[]; original=ctx.execute
    async def execute(key,action):
        calls.append(action)
        return await original(key=key,action=action)
    ctx.execute=execute
    asyncio.run(skill.run(ctx))
    assert ctx.failure is None
    result=ctx.result.model_dump() if hasattr(ctx.result,'model_dump') else ctx.result
    assert result['approach']['object_name']=='chosen'
    if multiple:
        assert result['selection']['selected_object_name']=='chosen'
        assert json.loads(calls[0].parameters['route_ref'])=={'candidate_object_names':['blocked','chosen']}
    else:
        assert json.loads(calls[0].parameters['route_ref'])=={'object_name':'chosen'}
    assert calls[1].parameters['target']['target_ref']=='chosen'
