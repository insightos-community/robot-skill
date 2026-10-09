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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semantic_robot_skill_sdk import ActionResult


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Target(Model):
    frame_id: Literal["odom"] = "odom"
    position_m: list[float] = Field(min_length=3, max_length=3)
    yaw_rad: float


class Input(Model):
    object_name: str | None = Field(default=None, min_length=1, pattern=r"\S")
    candidate_object_names: list[str] | None = Field(default=None, min_length=1, max_length=20)
    maximum_speed_mps: float = Field(default=0.15, gt=0, le=0.3)
    minimum_clearance_m: float = Field(default=0.05, gt=0)
    arrival_radius_m: float = Field(default=0.06, gt=0)
    timeout_seconds: float = Field(default=180, gt=0)

    @model_validator(mode='after')
    def target_mode(self):
        if (self.object_name is None) == (self.candidate_object_names is None):
            raise ValueError('Provide exactly one of object_name or candidate_object_names')
        names = self.candidate_object_names
        if names is not None and (any(not n.strip() for n in names) or len(set(names)) != len(names)):
            raise ValueError('Candidate object names must be nonempty and unique')
        return self


class CandidateEvaluation(Model):
    object_name: str
    status: Literal['succeeded', 'failed']
    path_length_m: float | None = Field(default=None, ge=0)
    error_code: str | None = None
    error_message: str | None = None


class Selection(Model):
    policy: Literal['nearest_reachable']
    selected_object_name: str
    map_id: str
    candidates: list[CandidateEvaluation]


class Approach(Model):
    target: Target
    object_name: str
    support_name: str | None = None
    map_id: str
    instance_id: str = Field(min_length=1, pattern=r"\S")
    run_id: str


class State(Model):
    stage: str = "resolve"
    approach: Approach | None = None
    selection: Selection | None = None
    results: dict[str, ActionResult] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class Arrival(Model):
    verdict: Literal["achieved", "not_achieved"]
    distance_to_target_m: float = Field(ge=0)
    final_pose_ref: str = Field(min_length=1)
    carrying_object: dict | None = None


class Result(Model):
    target: Target
    approach: Approach
    selection: Selection | None = None
    final_pose_ref: str
    distance_to_target_m: float
    yaw_verified: Literal[False] = False
    evidence_refs: list[str] = Field(default_factory=list)
