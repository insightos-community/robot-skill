"""place-object 的 schema v2 双工具模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semantic_robot_skill_sdk import Pose3D


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HeldObjectState(StrictModel):
    """本次place-object Execution由实时Observation组合出的局部状态。"""

    object_ref: str
    robot_ref: str
    tool_refs: tuple[str, ...] = Field(min_length=2, max_length=2)
    tool_poses: dict[str, Pose3D]
    object_pose: Pose3D
    object_size_m: tuple[float, float, float]
    base_position_m: tuple[float, float, float] | None = None
    robot_state_generation: int = Field(ge=0)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tool_poses(self) -> "HeldObjectState":
        if set(self.tool_poses) != set(self.tool_refs):
            raise ValueError("tool_poses必须与tool_refs一一对应")
        return self


class PlacementTarget(StrictModel):
    """Robot Agent 给出的业务目标和可选提示，不是实时物理事实。"""

    target_ref: str = Field(min_length=1)
    pose_hint: Pose3D | None = None
    extent_hint_m: tuple[float, float, float] | None = None
    category_hint: str | None = None
    stability_duration_ms: int = Field(default=1000, gt=0)


class PlacedObjectState(StrictModel):
    """实时感知完成稳定性验证后形成的放置结果。"""

    object_ref: str
    target_ref: str
    final_pose: Pose3D
    support_surface_ref: str | None = None
    position_error_m: float = Field(ge=0.0)
    orientation_error_rad: float = Field(ge=0.0)
    stable: bool
    gripper_empty: bool
    verification_source: str
    observed_duration_ms: int = Field(ge=0)
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scene_revision: str
    evidence_refs: list[str] = Field(default_factory=list)


class EndEffectorStateValue(BaseModel):
    frame_id: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    model_config = ConfigDict(extra="ignore")


class RawToolStateValue(BaseModel):
    tool_ref: str
    side: str
    kind: str
    hook_contact: bool | None = None
    clamp_contact: bool | None = None
    model_config = ConfigDict(extra="ignore")


class RobotStateValue(BaseModel):
    """place-object实际消费的RobotState只读视图。

    RobotState是通用Ability输出，后续可以增加夹具开度等原始传感字段；
    本Skill只校验建立局部持物关系所需的字段，不能因无关字段扩展拒绝执行。
    """

    robot_id: str
    generation: int = Field(ge=0)
    observed_at: datetime
    base_pose: EndEffectorStateValue | None = None
    end_effectors: dict[str, EndEffectorStateValue]
    tool_states: dict[str, RawToolStateValue]
    model_config = ConfigDict(extra="ignore")


class ToolLoadToolValue(StrictModel):
    tool_ref: str
    available: bool
    position: float | None = None
    velocity: float | None = None
    effort: float | None = None
    hook_contact: bool | None = None
    clamp_contact: bool | None = None
    hook_force_n: float | None = None
    clamp_force_n: float | None = None
    hook_support_ratio: float | None = None
    relative_tangential_speed_mps: float | None = None
    sensor_fault: bool | None = None


class ToolLoadObservationValue(StrictModel):
    condition_satisfied: bool
    observed_duration_ms: int = Field(ge=0)
    tools: list[ToolLoadToolValue] = Field(min_length=1, max_length=2)
    slip_detected: bool
    overload_detected: bool
    sensor_fault: bool
    reasons: list[str] = Field(default_factory=list)


class TargetPoseValue(BaseModel):
    object_ref: str
    pose: Pose3D
    extent_m: tuple[float, float, float]
    identity_confidence: float = Field(ge=0.0, le=1.0)
    model_config = ConfigDict(extra="ignore")


class PlacementSlotState(BaseModel):
    schema_version: Literal[2] = 2
    target_ref: str = Field(min_length=1)
    region_ref: str = Field(min_length=1)
    placement_pose: Pose3D
    approach_vector: tuple[float, float, float]
    extent_m: tuple[float, float, float] | None = None
    free: bool
    reachable: bool
    occupants: list[str] = Field(default_factory=list)
    support_surface_ref: str = Field(min_length=1)
    support_center_pose: Pose3D | None = None
    support_occupied: bool = False
    lateral_clearance_m: dict[str, float | None] = Field(default_factory=dict)
    revision: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def validate_approach_vector(self) -> "PlacementSlotState":
        if sum(component * component for component in self.approach_vector) <= 1e-12:
            raise ValueError("approach_vector 不能为零向量")
        return self


class PlacementConstraints(StrictModel):
    approach_clearance_m: float = Field(default=0.12, gt=0.0, le=0.5)
    max_approach_speed_mps: float = Field(default=0.08, gt=0.0, le=0.3)
    max_contact_force_n: float = Field(default=60.0, gt=0.0)
    release_opening_m: float = Field(default=0.035, ge=0.0)
    retreat_disengage_m: float = Field(default=0.06, gt=0.0, le=0.2)
    target_observation_max_age_ms: int = Field(default=2_000, ge=100)
    held_observation_max_age_ms: int = Field(default=1_000, ge=100)
    max_target_refreshes: int = Field(default=1, ge=0, le=5)
    max_approach_recoveries: int = Field(default=1, ge=0, le=5)
    max_retreat_recoveries: int = Field(default=1, ge=0, le=3)
    max_stability_rechecks: int = Field(default=1, ge=0, le=3)


class PlaceObjectInput(StrictModel):
    object_ref: str = Field(min_length=1)
    target: PlacementTarget


class ToolPoseTarget(StrictModel):
    tool_ref: str
    target_pose: Pose3D


class ToolCommand(StrictModel):
    tool_ref: str
    target_position_m: float = Field(ge=0.0)
    maximum_force_n: float = Field(gt=0.0)
    hold: bool = False


class PlacementWaypoint(StrictModel):
    name: Literal[
        "preplace", "landing", "release", "unseat", "disengage", "retreat",
        "travel_clearance",
    ]
    object_pose: Pose3D
    targets: list[ToolPoseTarget] = Field(min_length=1, max_length=2)


class PlacementApproachPlan(StrictModel):
    target_ref: str
    target_revision: str
    waypoints: list[PlacementWaypoint] = Field(min_length=6, max_length=6)

    early_release_tool_ref: str | None = None
    early_release_waypoints: list[PlacementWaypoint] = Field(
        default_factory=list,
        max_length=5,
    )


class GetRobotStateParameters(StrictModel):
    pass


class VerifyToolLoadParameters(StrictModel):
    tool_refs: tuple[str, ...] = Field(min_length=1, max_length=2)


class LocateObjectParameters(StrictModel):
    object_ref: str
    minimum_confidence: float = Field(default=0.65, ge=0.0, le=1.0)


class ObservePlacementSlotParameters(StrictModel):
    target_ref: str
    object_ref: str
    require_free: bool = True
    require_reachable: bool = True
    pose_hint: Pose3D | None = None
    extent_hint_m: tuple[float, float, float] | None = None
    category_hint: str | None = None


class MoveToolTargetsParameters(StrictModel):
    targets: list[ToolPoseTarget] = Field(min_length=1, max_length=2)
    coordination: Literal["synchronized"] = "synchronized"
    purpose: Literal["place", "unseat", "disengage", "retreat", "clearance"]
    object_ref: str
    target_ref: str
    target_revision: str
    expected_object_pose: Pose3D | None = None
    maximum_speed_mps: float = Field(gt=0.0)
    max_contact_force_n: float = Field(gt=0.0)
    required_contact_tools: list[str] = Field(default_factory=list, max_length=1)


class ReleaseToolsParameters(StrictModel):
    object_ref: str
    tools: list[ToolCommand] = Field(min_length=1, max_length=1)
    target_ref: str
    target_revision: str


class VerifyPlacementParameters(StrictModel):
    object_ref: str
    target_ref: str
    target_revision: str
    target_pose_hint: Pose3D
    target_extent_hint_m: tuple[float, float, float] | None = None
    stability_duration_ms: int = Field(gt=0)
    require_independent_source: bool = True


class SafeStopPlacementParameters(StrictModel):
    object_ref: str
    tools: list[ToolCommand] = Field(min_length=2, max_length=2)
    preserve_tool_state: bool = True
    object_may_be_released: bool
    reason: str
    mode: Literal["safe", "immediate"]


class MoveToPostureParameters(StrictModel):
    posture: Literal["travel"] = "travel"


class PlacementVerificationObservation(StrictModel):
    state: PlacedObjectState
    within_target: bool
    stable: bool
    observed_displacement_m: float = Field(ge=0.0)
    observed_duration_ms: int = Field(ge=0)
    support_contact: bool
    gripper_empty: bool
    independent_verification: bool


PlaceStage = Literal[
    "verify_held_object", "observe_target_slot", "plan_approach", "approach",
    "release", "retreat", "verify_stability", "restore_travel_posture",
    "await_agent_decision", "completed",
]


class PlaceObjectRunState(StrictModel):
    stage: PlaceStage = "verify_held_object"
    verified_held_object: HeldObjectState | None = None
    target_observation_ref: str | None = None
    target_slot: PlacementSlotState | None = None
    approach_plan: PlacementApproachPlan | None = None
    target_refreshes: int = 0
    approach_recoveries: int = 0
    retreat_recoveries: int = 0
    retreat_cursor: int = Field(default=0, ge=0, le=4)
    stability_rechecks: int = 0
    release_cursor: int = Field(default=0, ge=0, le=2)
    released_tool_refs: list[str] = Field(default_factory=list)
    release_confirmed: bool = False
    support_transfer_confirmed: bool = False
    retreat_confirmed: bool = False
    placed_object: PlacedObjectState | None = None
    postplace_clearance_completed: bool = False
    travel_posture_completed: bool = False
    force_target_refresh: bool = False
    decision_reason: str | None = None
    decision_count: int = 0
    evidence_refs: list[str] = Field(default_factory=list)


class PlacementAgentDecision(StrictModel):
    decision: Literal[
        "refresh_target", "retry_approach", "retry_retreat",
        "recheck_stability", "retry_posture", "abort",
    ]
    reason: str = Field(min_length=1)
    replacement_target_observation_ref: str | None = None


class LocalRecoveryDecision(StrictModel):
    disposition: Literal["retry", "refresh_target", "request_agent", "fail"]
    reason: str
