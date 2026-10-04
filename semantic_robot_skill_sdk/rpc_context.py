"""Python Worker 内使用的 SkillContext 实现。"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator, TypeVar

from pydantic import BaseModel

from .models import (
    Action,
    ActionFeedback,
    ActionResult,
    Observation,
    PublishedArtifact,
    ResolvedArtifact,
    SkillCancelled,
    StopOutcome,
)
from .protocol import LineJsonRpcPeer

ModelT = TypeVar("ModelT", bound=BaseModel)


class RpcActionHandle:
    def __init__(self, context: "RpcSkillContext", key: str, action_id: str, *, stop_action: bool = False) -> None:
        self._context = context
        self._key = key
        self._action_id = action_id
        self._stop_action = stop_action

    async def feedback(self) -> AsyncIterator[ActionFeedback]:
        while True:
            value = self._context._call(
                "action.feedback",
                {
                    "action_id": self._action_id,
                    "after_sequence": self._context._feedback.get(self._key, 0),
                },
            )
            items = value.get("feedback") or []
            for item in items:
                feedback = ActionFeedback.model_validate(item)
                self._context._feedback[self._key] = feedback.sequence
                self._context._remember(feedback.observations)
                yield feedback
            if bool(value.get("terminal")):
                return
            if not items:
                # 暂无反馈不等于动作结束，例如模型仍在推理。保持流开放，
                # 否则 Skill 会转入阻塞的 result()，丢失期间的验收与安全检查。
                # 等待让出事件循环；停止通知仍可通过原有 RPC 通道到达。
                self._context.check_cancelled()
                await asyncio.sleep(0.05)

    async def stop(self, reason: str) -> ActionResult:
        value = self._context._call(
            "action.stop", {"action_id": self._action_id, "reason": reason}
        )
        return self._context._result(value)

    async def result(self) -> ActionResult:
        value = self._context._call("action.result", {"action_id": self._action_id})
        result = self._context._result(value)
        if not self._stop_action:
            self._context.check_cancelled()
        return result


class RpcSkillContext:
    """把 SkillContext 方法翻译成 Pilot RPC，不包含 Robot 或 Ability 选择逻辑。"""

    def __init__(
        self,
        peer: LineJsonRpcPeer,
        *,
        execution_id: str,
        robot_ref: str,
        skill_input: dict[str, Any],
        checkpoint: dict[str, Any] | None,
        controllers: dict[str, Any],
        log_fields: dict[str, Any],
        feedback_cursors: dict[str, int] | None = None,
    ) -> None:
        self._peer = peer
        self.execution_id = execution_id
        self.robot_ref = robot_ref
        self._input = dict(skill_input)
        self._state = dict(checkpoint) if checkpoint else None
        self._controllers = dict(controllers)
        self._observations: dict[str, Observation] = {}
        self._feedback: dict[str, int] = dict(feedback_cursors or {})
        self._cancelled = False
        self._log_fields = dict(log_fields)
        self.status = "running"
        self.result: BaseModel | None = None
        self.failure: dict[str, Any] | None = None

    def input(self, model: type[ModelT]) -> ModelT:
        return model.model_validate(self._input)

    def load_state(self, model: type[ModelT], default: ModelT) -> ModelT:
        return (
            default.model_copy(deep=True)
            if self._state is None
            else model.model_validate(self._state)
        )

    def checkpoint(self, state: BaseModel) -> None:
        self._state = state.model_dump(mode="json")
        self._peer.send_notification(
            "checkpoint", {"execution_id": self.execution_id, "state": self._state}
        )

    def observation(self, observation_ref: str) -> Observation | None:
        existing = self._observations.get(observation_ref)
        if existing is not None:
            return existing.model_copy(deep=True)
        value = self._call("observation.get", {"observation_ref": observation_ref})
        if value is None:
            return None
        observation = Observation.model_validate(value)
        self._remember([observation])
        return observation

    def latest_observation(
        self,
        kind: str,
        *,
        subject_ref: str | None = None,
        max_age_ms: int | None = None,
    ) -> Observation | None:
        value = self._call(
            "observation.latest",
            {"kind": kind, "subject_ref": subject_ref, "max_age_ms": max_age_ms},
        )
        if value is None:
            return None
        observation = Observation.model_validate(value)
        self._remember([observation])
        return observation

    def controller(self, name: str) -> Any:
        if name not in self._controllers:
            raise KeyError(f"未注册 Controller：{name}")
        return self._controllers[name]

    async def execute(self, key: str, action: Action) -> ActionResult:
        return await (await self.start_action(key, action)).result()

    async def start_action(self, key: str, action: Action) -> RpcActionHandle:
        self.check_cancelled()
        value = self._call(
            "action.start",
            {
                "key": key,
                "action": action.model_dump(mode="json"),
                "stop_action": False,
            },
        )
        return RpcActionHandle(self, key, str(value["action_id"]))

    async def request_agent(
        self,
        key: str,
        reason: str,
        context: dict[str, Any],
        response_model: type[ModelT],
    ) -> ModelT:
        self.check_cancelled()
        value = self._call(
            "agent.request",
            {
                "key": key,
                "reason": reason,
                "context": context,
                "response_model": response_model.__name__,
                "response_schema": response_model.model_json_schema(),
            },
        )
        self.check_cancelled()
        return response_model.model_validate(value)

    def recent_evidence(self) -> list[str]:
        evidence: list[str] = []
        for item in self._observations.values():
            evidence.extend(item.evidence_refs)
            evidence.extend(item.artifact_refs)
        return list(dict.fromkeys(evidence))

    def resolve_artifact(self, ref: str) -> ResolvedArtifact:
        """让 Pilot 把 Server Artifact 下载到当前 Execution 的只读目录。"""

        value = self._call("artifact.resolve", {"ref": ref})
        return ResolvedArtifact.model_validate(value)

    def publish_artifact(
        self,
        path: str,
        media_type: str,
        summary: str,
    ) -> PublishedArtifact:
        """登记 Worker 工作区内的文件；文件内容不进入 JSON-RPC。"""

        value = self._call(
            "artifact.publish",
            {"path": path, "media_type": media_type, "summary": summary},
        )
        return PublishedArtifact.model_validate(value)

    def request_stop(self) -> None:
        self._cancelled = True

    def check_cancelled(self) -> None:
        if self._cancelled:
            raise SkillCancelled("Skill 已收到停止请求")

    def report(
        self,
        event: str,
        *,
        summary: str,
        evidence_refs: list[str] | None = None,
        stage: str | None = None,
        stage_status: str | None = None,
        expectation: str | None = None,
        observation_summary: str | None = None,
        deviation: str | None = None,
        progress: float | None = None,
        next_step: str | None = None,
    ) -> None:
        """上报由 Skill 自己判断的阶段状态，Pilot 不推测 Python 内部流程。"""

        self._peer.send_notification(
            "event.report",
            {
                "event": event,
                "summary": summary,
                "evidence_refs": evidence_refs or [],
                "stage": stage,
                "stage_status": stage_status,
                "expectation": expectation,
                "observation_summary": observation_summary,
                "deviation": deviation,
                "progress": progress,
                "next_step": next_step,
            },
        )

    async def execute_stop(self, key: str, action: Action) -> ActionResult:
        value = self._call(
            "action.start",
            {"key": key, "action": action.model_dump(mode="json"), "stop_action": True},
        )
        return await RpcActionHandle(self, key, str(value["action_id"]), stop_action=True).result()

    def stop_outcome(
        self,
        *,
        safe: bool,
        summary: str,
        physical_state: str,
        requires_intervention: bool = False,
        evidence_refs: list[str] | None = None,
    ) -> StopOutcome:
        return StopOutcome(
            safe=safe,
            summary=summary,
            physical_state=physical_state,
            requires_intervention=requires_intervention,
            evidence_refs=evidence_refs or [],
        )

    def complete(self, result: BaseModel) -> None:
        self.status = "completed"
        self.result = result.model_copy(deep=True)
        self._peer.send_notification(
            "complete", {"result": result.model_dump(mode="json")}
        )

    def fail(self, code: str, message: str, evidence: list[str] | None = None) -> None:
        self.status = "failed"
        self.failure = {
            "code": code,
            "message": message,
            "evidence_refs": evidence or [],
        }
        self._peer.send_notification("fail", self.failure)

    def log(self, level: str, message: str, **fields: Any) -> None:
        self._peer.send_notification(
            "log", {"level": level, "message": message, **self._log_fields, **fields}
        )

    def now(self) -> datetime:
        value = self._call("runtime.now", {})
        return datetime.fromisoformat(str(value))

    def _call(self, method: str, params: dict[str, Any]) -> Any:
        return self._peer.call(method, {"execution_id": self.execution_id, **params})

    def _result(self, value: Any) -> ActionResult:
        result = ActionResult.model_validate(value)
        self._remember(result.observations)
        return result

    def _remember(self, observations: list[Observation]) -> None:
        for item in observations:
            self._observations[item.id] = item.model_copy(deep=True)
