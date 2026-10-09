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

from math import isclose, sqrt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from semantic_robot_skill_sdk import ActionResult


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TorsoAngles(Model):
    positions_rad: list[float] = Field(min_length=4, max_length=4)


class TorsoOffset(Model):
    # [forward, up], measured in the R1Pro base's sagittal plane.
    offset_m: list[float] = Field(min_length=2, max_length=2)


class TorsoObservation(Model):
    mode: Literal["auto"] = "auto"


class ObjectOrientation(Model):
    object_ref: str = Field(min_length=1)
    frame_id: Literal["world"]
    source: Literal["simulation_geometry"]
    object_orientation_xyzw: list[float] = Field(min_length=4, max_length=4)
    body_orientation_xyzw: list[float] = Field(min_length=4, max_length=4)

    @field_validator("object_orientation_xyzw", "body_orientation_xyzw")
    @classmethod
    def unit_quaternion(cls, value):
        if not isclose(sqrt(sum(x * x for x in value)), 1.0, abs_tol=1e-5):
            raise ValueError("场景朝向必须为单位四元数")
        return value


class Input(Model):
    object_ref: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    side: Literal["left", "right", "auto"] = "auto"
    approach_preference: Literal["auto", "downward", "horizontal"] = "auto"
    pregrasp_planner: Literal["ik_filter", "curobo"] = "ik_filter"
    carry_mode: Literal["direct", "lift_first"] = "direct"
    torso: TorsoObservation | TorsoAngles | TorsoOffset | None = Field(default_factory=TorsoObservation)
    operation_torso: TorsoObservation | None = Field(default_factory=TorsoObservation)
    perception_profile: str | None = None
    grasp_offset_m: list[float] = Field(min_length=3, max_length=3)
    orientation_xyzw: list[float] | None = Field(default=None, min_length=4, max_length=4)
    opening_m: float = Field(gt=0)
    maximum_force_n: float = Field(gt=0)
    liftoff_height_m: float = Field(default=0.01, gt=0)
    minimum_confidence: float = Field(default=0.3, ge=0, le=1)
    timeout_seconds: float = Field(default=900, gt=0)

    @field_validator("orientation_xyzw")
    @classmethod
    def unit_quaternion(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not isclose(sqrt(sum(x * x for x in value)), 1.0, abs_tol=1e-5):
            raise ValueError("orientation_xyzw 必须为单位四元数，顺序 x/y/z/w")
        return value



class Pose(Model):
    frame_id: Literal["body"]
    position_m: list[float] = Field(min_length=3, max_length=3)
    orientation_xyzw: list[float] = Field(min_length=4, max_length=4)
    revision: str = Field(min_length=1)
    observed_at: str | None = None


class LocatedObject(Model):
    object_ref: str
    pose: Pose
    extent_m: list[float] = Field(min_length=3, max_length=3)
    identity_confidence: float = Field(ge=0, le=1)


class GraspPlan(Model):
    clearance: Pose
    pregrasp: Pose
    grasp: Pose
    lift: Pose
    candidate_id: str
    orientation_selection: dict | None = None


class PlannedMotion(Model):
    key: str
    purpose: str
    pose: Pose


class RigidTransform(Model):
    position_m: list[float] = Field(min_length=3, max_length=3)
    orientation_xyzw: list[float] = Field(min_length=4, max_length=4)

    @field_validator("orientation_xyzw")
    @classmethod
    def unit_quaternion(cls, value):
        if not isclose(sqrt(sum(x * x for x in value)), 1.0, abs_tol=1e-5):
            raise ValueError("抓持变换需要单位四元数")
        return value


class ObservationAnchor(Model):
    frame_id: Literal["world", "odom"]
    fixed_from_body: RigidTransform
    observed_at: str
    base_translation_drift_m: float = Field(ge=0)
    base_rotation_drift_rad: float = Field(ge=0)


class HeldGeometry(Model):
    object_size_m: list[float] = Field(min_length=3, max_length=3)
    eef_from_object: RigidTransform
    source: Literal["pregrasp_rgbd_and_measured_eef"] = "pregrasp_rgbd_and_measured_eef"
    fixed_frame_id: Literal["world", "odom"]
    recognition_observed_at: str
    eef_observed_at: str | None = None
    recognition_revision: str
    identity_confidence: float = Field(ge=0, le=1)

    @field_validator("object_size_m")
    @classmethod
    def positive_size(cls, value):
        if min(value) <= 0:
            raise ValueError("抓持物体尺寸必须为正")
        return value


class State(Model):
    stage: str = "torso"
    selected_side: Literal["left", "right"] | None = None
    side_selection: dict | None = None
    results: dict[str, ActionResult] = Field(default_factory=dict)
    torso_targets: list[list[float]] | None = None
    torso_durations: list[float] = Field(default_factory=list)
    object_orientation: ObjectOrientation | None = None
    plan: GraspPlan | None = None
    located: LocatedObject | None = None
    observation_anchor: ObservationAnchor | None = None
    held_geometry: HeldGeometry | None = None
    principal_axis: dict | None = None
    observation_source: dict | None = None
    observation_diagnostics: dict | None = None
    perception_hint: dict | None = None
    reference_torso: list[float] | None = None
    liftoff_target: dict | None = None
    # Legacy checkpoints used a fixed 1 cm target. Store the height with the
    # absolute target so resumed execution cannot reinterpret it from new input.
    liftoff_height_m: float = Field(default=0.01, gt=0)
    liftoff_verification: dict | None = None
    carry_pose: Pose | None = None
    carry_mode: Literal["direct", "lift_first"] | None = None
    carry_fallback_reason: str | None = None
    approach_offset: list[float] | None = None
    operation_verified: bool = False
    enclosure_checks: dict = Field(default_factory=dict)
    closing_started: bool = False
    evidence_refs: list[str] = Field(default_factory=list)


class Result(Model):
    object_ref: str
    grasp_side: Literal["left", "right"]
    side_selection: dict | None = None
    enclosure_checks: dict = Field(default_factory=dict)
    approach_selection: dict | None = None
    tool_ref: str
    grasp_pose: Pose
    carry_pose: Pose
    carry_mode: Literal["direct", "lift_first"] | None = None
    carry_fallback_reason: str | None = None
    liftoff_verification: dict | None = None
    object_size_m: list[float] | None = Field(default=None,
        description="同次识别 OBB 尺寸，米。与 eef_from_object 一起原样传给 behavior-place；null 时不可复用。")
    eef_from_object: RigidTransform | None = Field(default=None,
        description="闭合后实测夹爪到识别 OBB 的估计变换。仅同次持续稳定夹持有效；重抓、滑移或场景重置后失效。")
    held_geometry: HeldGeometry | None = None
    verification_level: Literal["contact_and_carry_motion"] = "contact_and_carry_motion"
    evidence_refs: list[str] = Field(default_factory=list)
