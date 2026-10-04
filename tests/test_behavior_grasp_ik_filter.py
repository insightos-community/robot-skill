import pytest
from pydantic import ValidationError

from semantic_robot_skills.skills.behavior_grasp.scripts.curobo import action
from semantic_robot_skills.skills.behavior_grasp.scripts.models import Input, Pose


def test_online_ik_strategy_matches_for_check_and_execution_and_excludes_carry():
    inputs = Input(object_ref='can', prompt='soda can', side='left',
                   grasp_offset_m=[0, 0, 0], opening_m=.05, maximum_force_n=50)
    pose = Pose(frame_id='body', position_m=[.5, .2, .6], orientation_xyzw=[0, 0, 0, 1], revision='live')
    check = action(inputs, pose, 'alignment', plan_only=True, offset=[0, 0, -.1], contact_object_ref='can')
    execute = action(inputs, pose, 'alignment', offset=[0, 0, -.1], contact_object_ref='can')
    assert check.parameters['pregrasp_planner'] == 'ik_filter'
    assert {k:v for k,v in check.parameters.items() if k != 'plan_only'} == {
        k:v for k,v in execute.parameters.items() if k != 'plan_only'}
    assert 'pregrasp_planner' not in action(inputs, pose, 'transport', attached_object_ref='can').parameters
    original = inputs.model_copy(update={'pregrasp_planner':'curobo'})
    assert action(original, pose, 'alignment', offset=[0,0,-.1]).parameters['pregrasp_planner'] == 'curobo'


def test_unknown_planner_is_rejected():
    with pytest.raises(ValidationError):
        Input(object_ref='can', prompt='can', grasp_offset_m=[0,0,0], opening_m=.05,
              maximum_force_n=50, pregrasp_planner='unchecked')
