# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Upright whole-body goal and compact measured completion result."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field
from semantic_robot_skill_sdk import ActionResult
class Model(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
class Input(Model):
    timeout_seconds:float=Field(default=900,gt=0)
class State(Model):
    stage:str="read"
    results:dict[str,ActionResult]=Field(default_factory=dict)
    joint_goals:dict[str,float]|None=None
    holding_side:Literal["left","right"]|None=None
    held_object_ref:str|None=None
    skipped_motion:bool=False
    evidence_refs:list[str]=Field(default_factory=list)
class Result(Model):
    holding_side:Literal["left","right"]|None=None
    held_object_ref:str|None=None
    joint_goals:dict[str,float]
    measured_positions_rad:dict[str,float]
    maximum_error_rad:float
    skipped_motion:bool
    verification_level:Literal["upright_carry_posture_verified"]="upright_carry_posture_verified"
    evidence_refs:list[str]=Field(default_factory=list)
