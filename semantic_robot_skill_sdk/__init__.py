"""正式 Robot Skill Runtime 的公共接口与测试实现。"""

from .context import ActionHandle, SkillContext
from .mock import MockSkillContext, ScriptedAction, run_skill
from .protocol import ConcurrentJsonRpcPeer, JsonRpcProtocolError, LineJsonRpcPeer
from .rpc_context import RpcSkillContext
from .models import (
    Action,
    ActionFeedback,
    ActionResult,
    FeedbackRequest,
    Observation,
    Pose3D,
    PublishedArtifact,
    ResolvedArtifact,
    SkillCancelled,
    SkillFailure,
    StopOutcome,
    StopRequest,
    SkillExecutionStatus,
)

__all__ = [
    "Action",
    "ActionFeedback",
    "ActionHandle",
    "ActionResult",
    "FeedbackRequest",
    "MockSkillContext",
    "Observation",
    "Pose3D",
    "PublishedArtifact",
    "ScriptedAction",
    "SkillCancelled",
    "SkillContext",
    "SkillFailure",
    "StopOutcome",
    "StopRequest",
    "SkillExecutionStatus",
    "ConcurrentJsonRpcPeer",
    "JsonRpcProtocolError",
    "LineJsonRpcPeer",
    "RpcSkillContext",
    "ResolvedArtifact",
    "run_skill",
]
