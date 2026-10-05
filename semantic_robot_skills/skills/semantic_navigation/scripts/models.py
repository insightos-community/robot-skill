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

"""语义导航 Skill 使用的类型化输入、状态和结果。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semantic_robot_skill_sdk import Pose3D


StageName = Literal["validate_target", "plan_route", "navigate", "verify_arrival"]
NavigationPurpose = Literal["approach_grasp", "carry_to_place", "transit"]


class CarryingObjectState(BaseModel):
    """本次导航Execution由实时通用Observation组合出的携物状态。"""

    object_ref: str
    robot_ref: str
    tool_refs: tuple[str, ...] = Field(min_length=1, max_length=2)
    tool_poses: dict[str, Pose3D]
    object_pose: Pose3D
    object_size_m: tuple[float, float, float]
    robot_state_generation: int = Field(ge=0)
    evidence_refs: list[str] = Field(default_factory=list)


class ResolvedNavigationTarget(BaseModel):
    """Robot Agent 给出的导航意图；路线与到达状态由 Ability 实时确认。"""

    target_ref: str = Field(min_length=1)
    pose: Pose3D
    constraints: dict[str, str | float | int | bool] = Field(default_factory=dict)


class SemanticNavigationInput(BaseModel):
    """Robot Agent 已经解析完成的语义导航输入。"""

    target: ResolvedNavigationTarget
    navigation_purpose: NavigationPurpose = "transit"
    carried_object_ref: str | None = None
    arrival_radius_m: float = Field(default=0.03, gt=0)
    maximum_speed_mps: float = Field(default=0.15, gt=0)
    minimum_clearance_m: float = Field(default=0.05, gt=0)

    @model_validator(mode="after")
    def validate_navigation_purpose(self) -> "SemanticNavigationInput":
        if self.navigation_purpose == "carry_to_place" and not self.carried_object_ref:
            raise ValueError("carry_to_place 导航必须提供 carried_object_ref")
        return self


class SemanticNavigationState(BaseModel):
    """可以跨脚本进程恢复的轻量 Stage 状态。"""

    stage: StageName = "validate_target"
    target: ResolvedNavigationTarget | None = None
    route_ref: str | None = None
    route_revision: int = 0
    plan_attempt: int = 0
    replan_count: int = 0
    verification_count: int = 0
    final_pose_ref: str | None = None
    distance_to_target_m: float | None = None
    carrying_object: CarryingObjectState | None = None
    carrying_verified: bool = False
    decision_revision: int = 0
    pending_decision_key: str | None = None


class SemanticNavigationResult(BaseModel):
    reached: bool
    target_ref: str
    final_pose_ref: str
    distance_to_target_m: float
    route_ref: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class PlanRouteOutput(BaseModel):
    route_ref: str
    resolved_target: ResolvedNavigationTarget | None = None


class FollowRouteOutput(BaseModel):
    final_pose_ref: str
    distance_to_target_m: float = Field(ge=0.0)


class VerifyArrivalOutput(BaseModel):
    verdict: Literal["achieved", "not_achieved", "uncertain"]
    final_pose_ref: str
    distance_to_target_m: float = Field(ge=0.0)
    carrying_object: dict[str, Any] | None = None


class NavigationAgentDecision(BaseModel):
    """Agent 只能选择 Skill 明确支持的恢复方式。"""

    action: Literal[
        "replace_target",
        "replan_route",
        "retry_navigation",
        "recheck_arrival",
        "abort_subtask",
    ]
    target: ResolvedNavigationTarget | None = None
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)


class PlanRouteParameters(BaseModel):
    target: ResolvedNavigationTarget
    navigation_purpose: NavigationPurpose
    maximum_speed_mps: float = Field(gt=0)
    minimum_clearance_m: float = Field(gt=0)
    carrying_object: CarryingObjectState | None = None


class FollowRouteParameters(BaseModel):
    route_ref: str
    navigation_purpose: NavigationPurpose
    maximum_speed_mps: float = Field(gt=0)
    minimum_clearance_m: float = Field(gt=0)
    carrying_object: CarryingObjectState | None = None


class VerifyArrivalParameters(BaseModel):
    target: ResolvedNavigationTarget
    navigation_purpose: NavigationPurpose
    arrival_radius_m: float = Field(gt=0)
    require_visual_confirmation: bool
    carrying_object: CarryingObjectState | None = None


class SafeStopParameters(BaseModel):
    reason: str
    mode: Literal["safe", "immediate"]


class GetRobotStateParameters(BaseModel):
    pass


class VerifyToolLoadParameters(BaseModel):
    tool_refs: tuple[str, ...] = Field(min_length=1, max_length=2)


class LocateObjectParameters(BaseModel):
    object_ref: str
    minimum_confidence: float = Field(default=0.65, ge=0.0, le=1.0)


class EndEffectorStateValue(BaseModel):
    frame_id: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    model_config = ConfigDict(extra="ignore")


class RawToolStateValue(BaseModel):
    tool_ref: str
    side: str
    kind: str
    hook_contact: bool
    clamp_contact: bool
    sensor_fault: bool = False
    model_config = ConfigDict(extra="ignore")


class RobotStateValue(BaseModel):
    robot_id: str
    generation: int = Field(ge=0)
    end_effectors: dict[str, EndEffectorStateValue]
    tool_states: dict[str, RawToolStateValue]
    model_config = ConfigDict(extra="ignore")


class ToolLoadToolValue(BaseModel):
    tool_ref: str
    available: bool
    hook_contact: bool | None = None
    clamp_contact: bool | None = None
    hook_force_n: float | None = None
    clamp_force_n: float | None = None
    hook_support_ratio: float | None = None
    relative_tangential_speed_mps: float | None = None
    position: float | None = None
    velocity: float | None = None
    effort: float | None = None
    sensor_fault: bool | None = None
    model_config = ConfigDict(extra="ignore")


class ToolLoadObservationValue(BaseModel):
    condition_satisfied: bool
    observed_duration_ms: int = Field(ge=0)
    tools: list[ToolLoadToolValue] = Field(min_length=1, max_length=2)
    slip_detected: bool
    overload_detected: bool
    sensor_fault: bool
    reasons: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")


class TargetPoseValue(BaseModel):
    object_ref: str
    pose: Pose3D
    extent_m: tuple[float, float, float]
    identity_confidence: float = Field(ge=0.0, le=1.0)
