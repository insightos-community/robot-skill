"""用于文档示例和单元测试的内存 Mock Runtime。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from .context import ModelT
from .models import (
    Action,
    ActionFeedback,
    ActionResult,
    Observation,
    PublishedArtifact,
    ResolvedArtifact,
    SkillCancelled,
    SkillFailure,
    StopOutcome,
    utc_now,
)


@dataclass
class ScriptedAction:
    """一条预先编排的 Mock Action 反馈和结果。"""

    result: ActionResult
    feedback: list[ActionFeedback] = field(default_factory=list)
    stop_result: ActionResult | None = None


@dataclass
class _ActionRecord:
    """稳定 key 对应的 Action 记录。"""

    digest: str
    script: ScriptedAction
    feedback_cursor: int = 0
    result: ActionResult | None = None


class _MockActionHandle:
    """Mock Runtime 中可以恢复的 Action 句柄。"""

    def __init__(self, context: "MockSkillContext", key: str, record: _ActionRecord):
        self._context = context
        self._key = key
        self._record = record

    async def feedback(self) -> AsyncIterator[ActionFeedback]:
        """从已保存游标继续返回反馈，避免恢复后重复消费。"""

        while self._record.feedback_cursor < len(self._record.script.feedback):
            feedback = self._record.script.feedback[self._record.feedback_cursor]
            self._record.feedback_cursor += 1
            self._context._remember_observations(feedback.observations)
            self._context.events.append(
                {
                    "type": "action_feedback",
                    "key": self._key,
                    "sequence": feedback.sequence,
                }
            )
            await asyncio.sleep(0)
            yield feedback

    async def stop(self, reason: str) -> ActionResult:
        """返回脚本给定的停止结果；未指定时返回经过确认的 stopped。"""

        if self._record.result is not None:
            return self._record.result

        result = self._record.script.stop_result or ActionResult(
            status="stopped",
            error_code="STOPPED_BY_SKILL",
            error_message=reason,
            physical_effect="possible",
        )
        return self._context._finish_action(self._key, self._record, result)

    async def result(self) -> ActionResult:
        """等待或读取该 Action 的唯一最终结果。"""

        if self._record.result is not None:
            return self._record.result
        return self._context._finish_action(
            self._key,
            self._record,
            self._record.script.result,
        )


class MockSkillContext:
    """支持检查点、稳定 Action key、观测和 Agent 回复的测试 Context。"""

    def __init__(
        self,
        skill_input: BaseModel,
        *,
        robot_ref: str = "robot://demo-1",
        workspace: str | Path | None = None,
    ):
        self.robot_ref = robot_ref
        self._input = skill_input.model_copy(deep=True)
        self._state: BaseModel | None = None
        self._controllers: dict[str, Any] = {}
        self._scripts: dict[str, deque[ScriptedAction]] = defaultdict(deque)
        self._records: dict[str, _ActionRecord] = {}
        self._observations: dict[str, Observation] = {}
        self._agent_replies: deque[dict[str, Any] | BaseModel] = deque()
        self._cancelled = False
        self.status = "running"
        self.result: BaseModel | None = None
        self.failure: SkillFailure | None = None
        self.checkpoint_count = 0
        self.logs: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._temporary_workspace = (
            tempfile.TemporaryDirectory(prefix="semantic-skill-")
            if workspace is None
            else None
        )
        self.workspace = Path(
            self._temporary_workspace.name if self._temporary_workspace else workspace
        ).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, ResolvedArtifact] = {}

    def input(self, model: type[ModelT]) -> ModelT:
        """按 Skill 声明的模型重新校验输入。"""

        return model.model_validate(self._input.model_dump(mode="json"))

    def load_state(self, model: type[ModelT], default: ModelT) -> ModelT:
        """读取最近检查点；首次运行时使用默认状态。"""

        if self._state is None:
            return default.model_copy(deep=True)
        return model.model_validate(self._state.model_dump(mode="json"))

    def checkpoint(self, state: BaseModel) -> None:
        """深拷贝脚本状态，模拟跨进程持久化边界。"""

        self._state = state.model_copy(deep=True)
        self.checkpoint_count += 1
        self.events.append(
            {
                "type": "checkpoint",
                "stage": getattr(state, "stage", None),
                "revision": self.checkpoint_count,
            }
        )

    @property
    def state(self) -> BaseModel | None:
        """仅供测试断言读取最近检查点。"""

        return self._state.model_copy(deep=True) if self._state else None

    def register_controller(self, name: str, controller: Any) -> None:
        """注册当前 Skill 使用的 Controller。"""

        self._controllers[name] = controller

    def controller(self, name: str) -> Any:
        """获取已注册 Controller，不做隐式回退。"""

        if name not in self._controllers:
            raise KeyError(f"未注册 Controller：{name}")
        return self._controllers[name]

    def queue_action(
        self,
        action_type: str,
        result: ActionResult,
        *,
        feedback: list[ActionFeedback] | None = None,
        stop_result: ActionResult | None = None,
    ) -> None:
        """按 Action 类型压入下一次执行所需的 Mock 响应。"""

        self._scripts[action_type].append(
            ScriptedAction(
                result=result,
                feedback=feedback or [],
                stop_result=stop_result,
            )
        )

    def queue_agent_reply(self, reply: dict[str, Any] | BaseModel) -> None:
        """压入下一条类型化 Agent 回复。"""

        self._agent_replies.append(reply)

    def add_observation(self, observation: Observation) -> None:
        """直接向当前执行注入一条已知观测。"""

        self._observations[observation.id] = observation.model_copy(deep=True)

    def observation(self, observation_ref: str) -> Observation | None:
        """按稳定引用读取指定观测。"""

        observation = self._observations.get(observation_ref)
        return observation.model_copy(deep=True) if observation else None

    def latest_observation(
        self,
        kind: str,
        *,
        subject_ref: str | None = None,
        max_age_ms: int | None = None,
    ) -> Observation | None:
        """读取当前执行范围内满足过滤条件的最新观测。"""

        matches = [
            observation
            for observation in self._observations.values()
            if observation.kind == kind
            and (subject_ref is None or observation.subject_ref == subject_ref)
        ]
        if not matches:
            return None

        latest = max(matches, key=lambda item: item.observed_at)
        if max_age_ms is not None:
            age_ms = (self.now() - latest.observed_at).total_seconds() * 1000
            if age_ms > max_age_ms:
                return None
        return latest.model_copy(deep=True)

    async def execute(self, key: str, action: Action) -> ActionResult:
        """执行非流式 Action；相同 key 返回已保存的唯一结果。"""

        handle = await self.start_action(key, action)
        return await handle.result()

    async def start_action(self, key: str, action: Action) -> _MockActionHandle:
        """创建或恢复相同稳定 key 的 Action 句柄。"""

        self.check_cancelled()
        digest = self._digest(action)
        existing = self._records.get(key)
        if existing is not None:
            if existing.digest != digest:
                raise ValueError(f"相同 Action key 对应了不同内容：{key}")
            return _MockActionHandle(self, key, existing)

        scripts = self._scripts.get(action.type)
        if not scripts:
            raise AssertionError(f"没有为 Action 准备 Mock 响应：{action.type}")

        record = _ActionRecord(digest=digest, script=scripts.popleft())
        self._records[key] = record
        self.events.append(
            {
                "type": "action_started",
                "key": key,
                "action": action.type,
                # 参数进入 Trace 便于测试和前端解释 Controller 的确定性计算。
                "parameters": action.parameters,
            }
        )
        return _MockActionHandle(self, key, record)

    async def request_agent(
        self,
        key: str,
        reason: str,
        context: dict[str, Any],
        response_model: type[ModelT],
    ) -> ModelT:
        """返回测试预置的 Agent 回复，并保留请求摘要。"""

        self.events.append(
            {
                "type": "agent_requested",
                "key": key,
                "reason": reason,
                "context": context,
            }
        )
        if not self._agent_replies:
            raise AssertionError(f"没有为 Agent 请求准备回复：{key}")
        reply = self._agent_replies.popleft()
        if isinstance(reply, BaseModel):
            reply = reply.model_dump(mode="json")
        return response_model.model_validate(reply)

    def recent_evidence(self) -> list[str]:
        """汇总当前执行最近保存的证据引用。"""

        evidence: list[str] = []
        for observation in self._observations.values():
            evidence.extend(observation.evidence_refs)
        return list(dict.fromkeys(evidence))

    def add_artifact(
        self,
        ref: str,
        *,
        content: bytes,
        media_type: str,
        summary: str = "",
        filename: str = "input.bin",
    ) -> ResolvedArtifact:
        """把一份 Server Artifact 放入当前 Mock Execution 工作区。"""

        artifact_dir = self.workspace / "inputs" / str(len(self._artifacts) + 1)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / Path(filename).name
        path.write_bytes(content)
        resolved = ResolvedArtifact(
            ref=ref,
            local_path=str(path),
            media_type=media_type,
            size_bytes=len(content),
            summary=summary,
        )
        self._artifacts[ref] = resolved
        return resolved.model_copy(deep=True)

    def resolve_artifact(self, ref: str) -> ResolvedArtifact:
        """只解析显式授权给当前执行的 Artifact 引用。"""

        artifact = self._artifacts.get(ref)
        if artifact is None:
            raise PermissionError(f"当前 Skill Execution 无权读取 Artifact：{ref}")
        return artifact.model_copy(deep=True)

    def publish_artifact(
        self,
        path: str,
        media_type: str,
        summary: str,
    ) -> PublishedArtifact:
        """只允许发布当前 Execution 工作区中的普通文件。"""

        candidate = Path(path).resolve()
        if not candidate.is_relative_to(self.workspace) or not candidate.is_file():
            raise PermissionError("只能发布当前 Skill Execution 工作区中的文件")
        artifact_id = str(uuid4())
        published = PublishedArtifact(
            local_ref=f"pilot-artifact://mock-pilot/{artifact_id}",
            server_ref=f"artifact://{artifact_id}",
            sync_status="available",
        )
        self.events.append(
            {
                "type": "artifact_published",
                "path": str(candidate),
                "media_type": media_type,
                "summary": summary,
                **published.model_dump(mode="json"),
            }
        )
        return published

    def request_stop(self) -> None:
        """模拟 Runtime 锁存停止请求。"""

        self._cancelled = True

    def check_cancelled(self) -> None:
        """停止已锁存时禁止 Skill 继续产生普通 Action。"""

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
        """记录前端和 Agent 可见的阶段性通知。"""

        self.events.append(
            {
                "type": event,
                "summary": summary,
                "evidence_refs": evidence_refs or [],
                "stage": stage,
                "stage_status": stage_status,
                "expectation": expectation,
                "observation_summary": observation_summary,
                "deviation": deviation,
                "progress": progress,
                "next_step": next_step,
            }
        )

    async def execute_stop(self, key: str, action: Action) -> ActionResult:
        """测试环境允许在取消期间执行明确的安全停止 Action。"""

        cancelled = self._cancelled
        self._cancelled = False
        try:
            return await self.execute(key, action)
        finally:
            self._cancelled = cancelled

    def stop_outcome(
        self,
        *,
        safe: bool,
        summary: str,
        physical_state: str,
        requires_intervention: bool = False,
        evidence_refs: list[str] | None = None,
    ) -> StopOutcome:
        """构造结构化停止结果。"""

        return StopOutcome(
            safe=safe,
            summary=summary,
            physical_state=physical_state,
            requires_intervention=requires_intervention,
            evidence_refs=evidence_refs or [],
        )

    def complete(self, result: BaseModel) -> None:
        """保存 Skill 完成结果。"""

        self.status = "completed"
        self.result = result.model_copy(deep=True)
        self.events.append({"type": "skill_completed"})

    def fail(
        self,
        code: str,
        message: str,
        evidence: list[str] | None = None,
    ) -> None:
        """保存可解释失败，避免抛出没有语义的异常。"""

        self.status = "failed"
        self.failure = SkillFailure(code, message, evidence)
        self.events.append({"type": "skill_failed", "code": code})

    def log(self, level: str, message: str, **fields: Any) -> None:
        """保存结构化业务日志。"""

        self.logs.append({"level": level, "message": message, **fields})

    def now(self) -> datetime:
        """返回测试所用的当前时间。"""

        return utc_now()

    def _remember_observations(self, observations: list[Observation]) -> None:
        """把 Action 反馈或结果携带的观测纳入当前执行范围。"""

        for observation in observations:
            self._observations[observation.id] = observation.model_copy(deep=True)

    def _finish_action(
        self,
        key: str,
        record: _ActionRecord,
        result: ActionResult,
    ) -> ActionResult:
        """持久化 Action 唯一终态并保存观测。"""

        if record.result is None:
            record.result = result.model_copy(deep=True)
            self._remember_observations(record.result.observations)
            self.events.append(
                {"type": "action_finished", "key": key, "status": result.status}
            )
        return record.result.model_copy(deep=True)

    @staticmethod
    def _digest(action: Action) -> str:
        """计算稳定内容摘要，用于拒绝相同 key 的不同 Action。"""

        payload = json.dumps(
            action.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


RunFunction = Callable[[MockSkillContext], Awaitable[None]]


async def run_skill(
    run: RunFunction,
    context: MockSkillContext,
    *,
    max_turns: int = 64,
) -> None:
    """模拟 Runtime 每次从 ``run(ctx)`` 重新进入当前 Stage。"""

    for _ in range(max_turns):
        if context.status in {"completed", "failed"}:
            return
        await run(context)
    raise AssertionError(f"Skill 在 {max_turns} 轮内没有进入终态")
