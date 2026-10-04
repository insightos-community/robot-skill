"""Robot Skill SDK 的公共类型。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """返回带 UTC 时区的当前时间，避免产生无时区时间。"""

    return datetime.now(timezone.utc)


class Pose3D(BaseModel):
    """带坐标系和可选来源版本的三维位姿。"""

    frame_id: str
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    observed_at: datetime = Field(default_factory=utc_now)
    # 位姿既可能来自实时 Ability Observation，也可能只是 Robot Agent 给出的目标提示。
    # 只有前者天然拥有观测版本，因此公共协议不能强迫调用方伪造 revision。
    revision: str | None = None


class Observation(BaseModel):
    """Pilot 交给 Skill 的统一低频观测；大数据只保存 Artifact 引用。"""

    id: str = Field(default_factory=lambda: f"obs-{uuid4()}")
    kind: str
    schema_version: int = 2
    subject_ref: str | None = None
    source: str
    observed_at: datetime = Field(default_factory=utc_now)
    revision: str | None = None
    frame_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    value: dict[str, Any] | None = None
    data_ref: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class ResolvedArtifact(BaseModel):
    """Pilot 为当前 Worker 准备好的只读 Artifact。"""

    ref: str
    local_path: str
    media_type: str
    size_bytes: int | None = Field(default=None, ge=0)
    summary: str = ""


class PublishedArtifact(BaseModel):
    """Worker 发布文件后得到的 Pilot 与 Server 引用。"""

    local_ref: str
    server_ref: str | None = None
    sync_status: Literal["pending", "available", "failed"]


class FeedbackRequest(BaseModel):
    """当前 Action 希望执行侧返回的低频业务观测。"""

    observation_kinds: list[str] = Field(default_factory=list)
    interval_ms: int = Field(default=200, ge=50)


class Action(BaseModel):
    """Controller 或 Stage 脚本产生的类型化 Action 信封。"""

    type: str
    schema_version: int = 2
    parameters: dict[str, Any]
    timeout_seconds: float = Field(gt=0)
    feedback: FeedbackRequest | None = None
    label: str | None = None

    @classmethod
    def from_model(
        cls,
        *,
        action_type: str,
        parameters: BaseModel,
        timeout_seconds: float,
        feedback: FeedbackRequest | None = None,
        label: str | None = None,
    ) -> "Action":
        """先由 Pydantic 参数模型完成校验，再生成可持久化信封。"""

        return cls(
            type=action_type,
            parameters=parameters.model_dump(mode="json"),
            timeout_seconds=timeout_seconds,
            feedback=feedback,
            label=label,
        )


class ActionFeedback(BaseModel):
    """Action 执行过程中按序返回的进度和观测。"""

    sequence: int = Field(ge=1)
    status: str
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    message: str = ""
    severity: Literal["info", "warning", "critical"] = "info"
    measurements: dict[str, float | int | str | bool] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    safe_checkpoint: bool = False


class ActionResult(BaseModel):
    """Action 的唯一最终结果；超时由 error_code=TIMEOUT 表达。"""

    status: Literal["succeeded", "failed", "stopped", "interrupted"]
    output: dict[str, Any] | None = None
    observations: list[Observation] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    physical_effect: Literal["none", "possible", "confirmed", "unknown"] = "unknown"


class StopRequest(BaseModel):
    """用户、Agent 或 Runtime 锁存的停止请求。"""

    id: str = Field(default_factory=lambda: f"stop-{uuid4()}")
    source: Literal["user", "agent", "runtime"]
    reason: str
    mode: Literal["safe", "immediate"] = "safe"
    requested_at: datetime = Field(default_factory=utc_now)


class StopOutcome(BaseModel):
    """Skill 特有停止逻辑完成后的物理状态说明。"""

    safe: bool
    summary: str
    physical_state: str
    requires_intervention: bool = False
    evidence_refs: list[str] = Field(default_factory=list)


class SkillExecutionStatus(StrEnum):
    """Pilot 和 Worker 共同保存的 Skill Execution 状态。"""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    WAITING_AGENT = "waiting_agent"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"


class SkillFailure(RuntimeError):
    """表示 Skill 已经形成可解释的失败结果。"""

    def __init__(self, code: str, message: str, evidence: list[str] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.evidence = evidence or []


class SkillCancelled(RuntimeError):
    """表示 Runtime 已经锁存停止请求，脚本不得继续产生 Action。"""
