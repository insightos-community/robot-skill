"""周转箱抓取 Skill 的 schema v2 类型。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semantic_robot_skill_sdk import Action, Pose3D

GraspStrategy = Literal[
    "auto", "direct_bilateral", "left_extract_first", "right_extract_first"
]
GraspStage = Literal[
    "observe_target",
    "approach",
    "grasp",
    "lift_and_verify",
    "prepare_transport",
]


class TargetHint(BaseModel):
    """Robot Agent 给出的目标提示；执行时必须由感知 Ability 重新观测。"""

    object_ref: str = Field(min_length=1)
    pose_hint: Pose3D | None = None
    extent_hint_m: tuple[float, float, float] | None = None
    category_hint: str | None = None


class HeldObjectState(BaseModel):
    """grasp-object 验证完成后输出的业务结果。

    这是 Skill 的业务契约，不属于通用 Runtime SDK。navigation/place 会按各自
    输入模型读取所需 JSON 字段，并在开始执行时重新确认实时持物状态。
    """

    object_ref: str
    robot_ref: str
    tool_refs: tuple[str, ...] = Field(min_length=1, max_length=2)
    tool_poses: dict[str, Pose3D]
    grasp_pose: Pose3D
    object_pose: Pose3D
    object_size_m: tuple[float, float, float]
    grasp_candidate_id: str
    grasp_confidence: float = Field(ge=0.0, le=1.0)
    estimated_mass_kg: float | None = Field(default=None, ge=0.0)
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    robot_state_revision: str
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tool_poses(self) -> "HeldObjectState":
        # 放置阶段需要真实的末端—物体几何关系；缺失一侧位姿时不能靠工具名称
        # 猜偏移，因此只在本业务结果边界校验一次，不在 SDK/Pilot 重复校验。
        if set(self.tool_poses) != set(self.tool_refs):
            raise ValueError("tool_poses 必须与 tool_refs 一一对应")
        return self


class ToolPoseTarget(BaseModel):
    tool_ref: str
    target_pose: Pose3D


class ToolSetpoint(BaseModel):
    tool_ref: str
    target_position_m: float = Field(ge=0)
    maximum_force_n: float = Field(gt=0)
    hold: bool = False


class GraspObjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: TargetHint
    tool_refs: tuple[str, ...] = Field(
        default=("component://tool/left", "component://tool/right"),
        min_length=2,
        max_length=2,
    )
    preferred_strategy: GraspStrategy = "auto"
    minimum_lift_height_m: float = Field(default=0.08, gt=0)

    @property
    def object_ref(self) -> str:
        return self.target.object_ref

    @model_validator(mode="after")
    def validate_tools(self) -> "GraspObjectInput":
        if len(set(self.tool_refs)) != 2:
            raise ValueError("周转箱抓取必须声明左右两个不同工具")
        return self


class ToolTargetDescriptor(BaseModel):
    tool_ref: str
    side: Literal["left", "right"]


class GraspCandidate(BaseModel):
    candidate_id: str
    object_ref: str
    observation_revision: str
    planned_object_pose: Pose3D
    strategy: Literal["direct_bilateral", "left_extract_first", "right_extract_first"]
    tool_targets: list[ToolTargetDescriptor] = Field(min_length=2, max_length=2)
    clearance_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)
    approach_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)
    transfer_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)
    alignment_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)
    pregrasp_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)
    hook_insert_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)
    hook_seat_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)
    opening_setpoints: list[ToolSetpoint] = Field(min_length=2, max_length=2)
    clamp_setpoints: list[ToolSetpoint] = Field(min_length=2, max_length=2)
    pull_path: list[ToolPoseTarget] = Field(default_factory=list, max_length=1)
    pull_direction_world: tuple[float, float, float] | None = None
    pull_distance_m: float = Field(default=0.0, ge=0)
    clearance_m: float = Field(gt=0)
    score: float = Field(ge=0, le=1)


class GraspObjectState(BaseModel):
    stage: GraspStage = "observe_target"
    active_strategy: GraspStrategy = "auto"
    plan_revision: int = Field(default=0, ge=0)
    decision_revision: int = Field(default=0, ge=0)
    pending_decision_key: str | None = None
    observation_attempts: int = Field(default=0, ge=0)
    grasp_attempts: int = Field(default=0, ge=0)
    verification_attempts: int = Field(default=0, ge=0)
    target_observation_ref: str | None = None
    target_revision: str | None = None
    target_pose: Pose3D | None = None
    target_extent_m: tuple[float, float, float] | None = None
    candidates: list[GraspCandidate] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    approach_plan: list[Action] = Field(default_factory=list)
    approach_cursor: int = Field(default=0, ge=0)
    pregrasp_observation_ref: str | None = None
    primary_clamped: bool = False
    extraction_planned_from_contact: bool = False
    primary_reclamp_attempted: bool = False
    extraction_primary_side: Literal["left", "right"] | None = None
    extraction_completed: bool = False
    extraction_reobserved: bool = False
    extraction_adjustments: int = Field(default=0, ge=0)
    secondary_replan_attempted: bool = False
    object_held: bool = False
    lift_completed: bool = False
    verification_only: bool = False
    pending_lift_distance_m: float | None = Field(default=None, gt=0)
    verified_held_object: HeldObjectState | None = None
    transport_clearance_poses: list[ToolPoseTarget] = Field(
        default_factory=list, max_length=2
    )
    transport_clearance_completed: bool = False
    transport_poses: list[ToolPoseTarget] = Field(default_factory=list, max_length=2)
    transport_completed: bool = False
    # 同一物理现场已经证明不可行的接触前策略只在本Execution内记忆。
    # 它阻止Robot Agent在left/direct/right之间无新证据地往返，不是新的领域状态。
    failed_precontact_strategies: list[
        Literal["direct_bilateral", "left_extract_first", "right_extract_first"]
    ] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class TargetPoseValue(BaseModel):
    object_ref: str
    pose: Pose3D
    extent_m: tuple[float, float, float]
    identity_confidence: float = Field(ge=0, le=1)


class GraspCandidatesValue(BaseModel):
    target_revision: str
    candidates: list[GraspCandidate]


class PregraspStateValue(BaseModel):
    candidate_id: str
    target_revision: str
    reached: bool
    position_errors_m: dict[str, float]
    object_relative_errors_m: dict[str, float]
    hook_contacts: dict[str, bool]


class GraspContactValue(BaseModel):
    candidate_id: str
    object_ref: str
    tools: dict[str, dict[str, Any] | None]
    contact_confirmed: bool
    stable_bilateral_load: bool
    slipping: bool = False
    overloaded: bool = False


class LiftProgressValue(BaseModel):
    candidate_id: str
    lift_height_m: float = Field(ge=0)
    object_follows_tools: bool
    stable_load: bool
    slip_detected: bool = False
    overloaded: bool = False


class GraspVerificationValue(BaseModel):
    candidate_id: str
    held: bool
    stable_bilateral_load: bool
    lift_height_m: float
    stable_duration_ms: int = Field(ge=0)
    slipping: bool = False
    overloaded: bool = False
    object_pose: Pose3D
    tool_poses: dict[str, Pose3D]


class GraspObjectResult(BaseModel):
    expected_state: Literal["held_object"] = "held_object"
    held_object: HeldObjectState


class GraspAgentDecision(BaseModel):
    expected_plan_revision: int = Field(ge=0)
    action: Literal[
        "restart_observation",
        "select_candidate",
        "change_strategy",
        "retry_verification",
        "abort_subtask",
    ]
    selected_candidate_id: str | None = None
    strategy: GraspStrategy | None = None
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)


class LocateObjectParameters(BaseModel):
    object_ref: str
    minimum_confidence: float = Field(ge=0, le=1)
    pose_hint: Pose3D | None = None
    extent_hint_m: tuple[float, float, float] | None = None
    category_hint: str | None = None


class GenerateCandidatesParameters(BaseModel):
    object_ref: str
    target_pose: Pose3D
    object_extent_m: tuple[float, float, float]
    target_revision: str
    preferred_strategy: GraspStrategy
    maximum_candidates: int = Field(default=3, ge=1, le=3)
    engaged_tool_ref: str | None = None
    secondary_resume_phase: Literal["insert"] | None = None


class PlanTransportPostureParameters(BaseModel):
    object_ref: str
    object_pose: Pose3D
    object_extent_m: tuple[float, float, float]
    target_revision: str
    tool_refs: tuple[str, ...] = Field(min_length=2, max_length=2)


class TransportPostureValue(BaseModel):
    object_ref: str
    target_revision: str
    desired_object_pose: Pose3D
    clearance_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)
    transport_poses: list[ToolPoseTarget] = Field(min_length=2, max_length=2)


class MoveTargetsParameters(BaseModel):
    targets: list[ToolPoseTarget] = Field(min_length=1, max_length=2)
    coordination: Literal["synchronized"] = "synchronized"
    purpose: Literal[
        "clearance",
        "transfer",
        "alignment",
        "pregrasp",
        "insert",
        "seat",
        "extract",
        "transport",
        "place",
        "retreat",
    ]
    object_ref: str
    candidate_id: str
    target_revision: str
    maximum_speed_mps: float = Field(gt=0)
    max_contact_force_n: float | None = Field(default=None, gt=0)
    required_contact_tools: tuple[str, ...] = Field(default_factory=tuple, max_length=2)


class VerifyPregraspParameters(BaseModel):
    object_ref: str
    candidate_id: str
    planned_object_pose: Pose3D
    expected_targets: list[ToolPoseTarget] = Field(min_length=1, max_length=2)
    target_revision: str
    maximum_position_error_m: float = Field(gt=0)


class SetOpeningParameters(BaseModel):
    tools: list[ToolSetpoint] = Field(min_length=1, max_length=2)


class CloseToolsParameters(BaseModel):
    object_ref: str
    tools: list[ToolSetpoint] = Field(min_length=1, max_length=2)
    candidate_id: str
    grasp_pose: Pose3D


class LiftObjectParameters(BaseModel):
    object_ref: str
    tools: tuple[str, ...] = Field(min_length=2, max_length=2)
    candidate_id: str
    distance_m: float = Field(gt=0)
    maximum_speed_mps: float = Field(gt=0)


class VerifyGraspParameters(BaseModel):
    object_ref: str
    tools: tuple[str, ...] = Field(min_length=2, max_length=2)
    candidate_id: str
    initial_object_pose: Pose3D
    minimum_lift_height_m: float = Field(gt=0)
    lift_height_tolerance_m: float = Field(ge=0, le=0.01)
    stable_duration_ms: int = Field(ge=100, le=5000)


class HoldObjectParameters(BaseModel):
    object_ref: str
    tools: list[ToolSetpoint] = Field(min_length=2, max_length=2)
    preserve_tool_state: bool = True
    reason: str


class SafeStopGraspParameters(HoldObjectParameters):
    mode: Literal["safe", "immediate"]


class GetRobotStateParameters(BaseModel):
    pass


class VerifyToolLoadParameters(BaseModel):
    tool_refs: tuple[str, ...] = Field(min_length=1, max_length=2)


class EndEffectorStateValue(BaseModel):
    frame_id: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    model_config = ConfigDict(extra="ignore")


class RawToolStateValue(BaseModel):
    tool_ref: str
    side: str
    kind: str
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
