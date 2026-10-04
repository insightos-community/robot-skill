"""Initialization state and per-arm joint targets."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from semantic_robot_skill_sdk import ActionResult


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Input(Model):
    """Select a preset; current joint angles come from robot.get_state."""

    arm_posture: Literal["down", "raised"] = Field(
        default="down", description="down：双臂伸直下垂；raised：双肩外展、屈肘抬起"
    )


class Pose(Model):
    frame_id: Literal["body"] = "body"
    position_m: list[float] = Field(min_length=3,max_length=3)
    orientation_xyzw: list[float] = Field(min_length=4,max_length=4)
    revision: str


class ArmTarget(Model):
    side: Literal["left","right"]
    joint_names: list[str] = Field(min_length=7, max_length=7)
    positions_rad: list[float] = Field(min_length=7, max_length=7)
    pose: Pose


class JointSample(Model):
    names: list[str]
    positions_rad: list[float]
    velocities_rad_s: list[float] = Field(default_factory=list)
    efforts: list[float] = Field(default_factory=list)


class State(Model):
    stage: str = "read_state"
    arm_posture: Literal["down", "raised"] = "down"
    results: dict[str, ActionResult] = Field(default_factory=dict)
    targets: list[ArmTarget] = Field(default_factory=list)
    torso_pitch_rad: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class Result(Model):
    arm_posture: Literal["down", "raised"]
    torso_pitch_rad: float
    targets: list[ArmTarget]
    evidence_refs: list[str] = Field(default_factory=list)
