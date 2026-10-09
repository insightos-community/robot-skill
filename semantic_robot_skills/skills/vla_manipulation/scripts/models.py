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


class ManipulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: Literal["grasp", "native_task"]
    target_source_id: str | None = None
    instruction: str = Field(min_length=1)
    max_actions: int = Field(default=400, gt=0)
    timeout_seconds: float = Field(default=300, gt=0)

    @model_validator(mode="after")
    def require_grasp_target(self):
        if self.objective == "grasp" and not self.target_source_id:
            raise ValueError("抓取必须指定当前地图物品的 source_id")
        return self


class ManipulationState(BaseModel):
    stage: str = "validate_target"
    generation: int | None = None
    robot_id: str | None = None
    initial_target_height: float | None = None
    previous_sample: dict | None = None
    stable_since: float | None = None
    monitor_index: int = 0
    verified: bool = False
    # 首次成功采用该次反馈中已确认完成的控制样本数。它是观测到成功的边界，
    # 不是声称捕获了两次观测之间最早成立 BDDL 的物理帧。
    first_success_actions: int | None = None
    tail_action_limit: int | None = None
    tail_limit_applied: bool = False
    executed_actions: int = 0


class ManipulationResult(BaseModel):
    objective: str
    target_source_id: str | None
    generation: int
    native_task_success: bool
    evidence_refs: list[str]
    target_reached_once: bool = False
    first_success_actions: int | None = None
    post_success_actions: int = 0
    post_success_complete: bool = False
