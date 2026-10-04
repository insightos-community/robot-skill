"""Inputs, persistent plans and measured poses for radio button approach."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from semantic_robot_skill_sdk import ActionResult


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Input(Model):
    operating_side: Literal["left", "right", "auto"] = "auto"
    wrist_sensor_id: str | None = Field(default=None, min_length=1)
    button_ref: str = Field(default="radio-button", min_length=1)
    button_prompt: str = Field(default="red button", min_length=1, max_length=512)
    wrist_model_profile: str | None = Field(default=None, min_length=1)
    minimum_confidence: float = Field(default=0.3, ge=0, le=1)
    maximum_force_n: float = Field(default=20, gt=0)
    timeout_seconds: float = Field(default=150, gt=0)

    @property
    def capture_sensor(self) -> str:
        if self.operating_side == "auto": raise ValueError("操作侧尚未由双手反馈确定")
        return self.wrist_sensor_id or f"{self.operating_side}_wrist_camera"

    @model_validator(mode="after")
    def consistent_wrist_side(self):
        if self.operating_side == "auto": return self
        other = "right" if self.operating_side == "left" else "left"
        sensor = self.wrist_sensor_id or ""
        if sensor in {f"{other}_wrist", f"{other}_wrist_camera"} or f"{other}_realsense_link" in sensor.split(":"):
            raise ValueError("wrist_sensor_id 须指向 operating_side 对应的腕部相机")
        if self.wrist_model_profile == f"sam31-{other}-wrist":
            raise ValueError("wrist_model_profile 须对应 operating_side")
        return self

    @property
    def holding_side(self) -> str:
        if self.operating_side == "auto": raise ValueError("持物侧尚未由双手反馈确定")
        return "right" if self.operating_side == "left" else "left"

    @property
    def perception_profile(self) -> str:
        if self.operating_side == "auto": raise ValueError("操作侧尚未由双手反馈确定")
        return self.wrist_model_profile or f"sam31-{self.operating_side}-wrist"


class Pose(Model):
    frame_id: Literal["body"] = "body"
    position_m: list[float] = Field(min_length=3, max_length=3)
    orientation_xyzw: list[float] = Field(min_length=4, max_length=4)
    revision: str = Field(min_length=1)
    observed_at: str | None = None

    @field_validator("orientation_xyzw")
    @classmethod
    def unit_quaternion(cls, value):
        if abs(sum(v * v for v in value) - 1) > 1e-5:
            raise ValueError("orientation_xyzw must be a unit quaternion")
        return value


class LocatedButton(Model):
    object_ref: str
    pose: Pose
    extent_m: list[float] = Field(min_length=3, max_length=3)
    identity_confidence: float = Field(ge=0, le=1)


class JointSample(Model):
    names: list[str]
    positions_rad: list[float]
    velocities_rad_s: list[float] = Field(default_factory=list)
    efforts: list[float] = Field(default_factory=list)


class ArmTarget(Model):
    arm_side: Literal["left", "right"]
    joint_names: list[str]
    positions_rad: list[float] = Field(min_length=7, max_length=7)


class MotionPlan(Model):
    pose: Pose
    side: Literal["left", "right"]
    diagnostics: dict = Field(default_factory=dict)


class State(Model):
    stage: str = "natural"
    hand_selection: dict | None = None
    results: dict[str, ActionResult] = Field(default_factory=dict)
    plans: dict[str, MotionPlan] = Field(default_factory=dict)
    holding_pose: Pose | None = None
    approach_pose: Pose | None = None
    button_plan_diagnostics: dict = Field(default_factory=dict)
    button: LocatedButton | None = None
    recognition_attempts: int = Field(default=0, ge=0, le=2)
    flipped: bool = False
    evidence_refs: list[str] = Field(default_factory=list)


class Result(Model):
    operating_side: Literal["left", "right"]
    holding_side: Literal["left", "right"]
    hand_selection: dict | None = None
    button: LocatedButton
    requested_pose: Pose
    flipped: bool = False
    recognition_attempts: int = Field(ge=1, le=2)
    verification_level: Literal["button_localized_and_approach_motion"] = (
        "button_localized_and_approach_motion"
    )
    evidence_refs: list[str] = Field(default_factory=list)
