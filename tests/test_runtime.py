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

"""共享 Mock Runtime 的关键恢复语义测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from semantic_robot_skill_sdk import (
    Action,
    ActionFeedback,
    ActionResult,
    MockSkillContext,
    Observation,
    RpcSkillContext,
)
from semantic_robot_skill_sdk.models import SkillCancelled
from semantic_robot_skill_sdk.rpc_context import RpcActionHandle


class DemoInput(BaseModel):
    """测试使用的最小 Skill 输入。"""

    target_ref: str


class DemoActionInput(BaseModel):
    """测试使用的类型化 Action 参数。"""

    target_ref: str


class DemoAgentDecision(BaseModel):
    """验证AgentRequest自动携带Pydantic派生Schema。"""

    action: str
    reason: str


class RecordingPeer:
    """记录 Worker 发给 Pilot 的同步 RPC，避免测试携带二进制内容。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.calls.append((method, params))
        if method == "artifact.resolve":
            return {
                "ref": params["ref"],
                "local_path": "/pilot/executions/rex-1/inputs/image.png",
                "media_type": "image/png",
                "size_bytes": 128,
                "summary": "目标图像",
            }
        if method == "artifact.publish":
            return {
                "local_ref": "pilot-artifact://pilot-1/report-1",
                "server_ref": None,
                "sync_status": "pending",
            }
        if method == "agent.request":
            return {"action": "abort_subtask", "reason": "测试结束"}
        raise AssertionError(f"未处理的测试 RPC：{method}")

    def send_notification(self, method: str, params: dict[str, object]) -> None:
        raise AssertionError(f"本测试不应发送 notification：{method} {params}")


def test_same_action_key_reuses_result() -> None:
    """相同key和相同内容必须复用结果，不能消费第二条能力调用。"""

    async def scenario() -> None:
        """在标准库事件循环中执行异步 Runtime 场景。"""

        context = MockSkillContext(DemoInput(target_ref="scene://box-17"))
        context.queue_action(
            "demo.observe",
            ActionResult(status="succeeded", output={"value": 1}),
        )

        action = Action.from_model(
            action_type="demo.observe",
            parameters=DemoActionInput(target_ref="scene://box-17"),
            timeout_seconds=1,
        )

        first = await context.execute("observe:box-17", action)
        second = await context.execute("observe:box-17", action)

        assert first == second
        started = [
            event for event in context.events if event["type"] == "action_started"
        ]
        assert len(started) == 1

    asyncio.run(scenario())


def test_same_key_rejects_different_action() -> None:
    """恢复后相同key出现不同参数时必须立即拒绝。"""

    async def scenario() -> None:
        """验证同一幂等键不能绑定不同物理动作。"""

        context = MockSkillContext(DemoInput(target_ref="scene://box-17"))
        context.queue_action("demo.observe", ActionResult(status="succeeded"))

        first = Action.from_model(
            action_type="demo.observe",
            parameters=DemoActionInput(target_ref="scene://box-17"),
            timeout_seconds=1,
        )
        second = Action.from_model(
            action_type="demo.observe",
            parameters=DemoActionInput(target_ref="scene://box-18"),
            timeout_seconds=1,
        )

        await context.execute("observe:stable-key", first)
        with pytest.raises(ValueError, match="相同 Action key"):
            await context.execute("observe:stable-key", second)

    asyncio.run(scenario())


def test_feedback_cursor_resumes_without_duplicates() -> None:
    """重新取得ActionHandle时应从上次反馈游标继续。"""

    async def scenario() -> None:
        """模拟 Runtime 重进后按游标续读反馈流。"""

        context = MockSkillContext(DemoInput(target_ref="scene://box-17"))
        context.queue_action(
            "demo.stream",
            ActionResult(status="succeeded"),
            feedback=[
                ActionFeedback(sequence=1, status="running", progress=0.5),
                ActionFeedback(sequence=2, status="running", progress=0.9),
            ],
        )
        action = Action.from_model(
            action_type="demo.stream",
            parameters=DemoActionInput(target_ref="scene://box-17"),
            timeout_seconds=1,
        )

        first_handle = await context.start_action("stream:box-17", action)
        first_feedback = []
        async for feedback in first_handle.feedback():
            first_feedback.append(feedback.sequence)
            break

        resumed_handle = await context.start_action("stream:box-17", action)
        resumed_feedback = [
            feedback.sequence async for feedback in resumed_handle.feedback()
        ]

        assert first_feedback == [1]
        assert resumed_feedback == [2]

    asyncio.run(scenario())


def test_latest_observation_filters_subject() -> None:
    """最新观测必须同时匹配kind和subject_ref。"""

    context = MockSkillContext(DemoInput(target_ref="scene://box-17"))
    box_17 = Observation(
        kind="object_pose",
        subject_ref="scene://box-17",
        source="mock-camera",
        revision="scene-1",
        value={"x": 1.0},
    )
    box_18 = Observation(
        kind="object_pose",
        subject_ref="scene://box-18",
        source="mock-camera",
        revision="scene-1",
        value={"x": 2.0},
    )
    context.add_observation(box_17)
    context.add_observation(box_18)

    selected = context.latest_observation(
        "object_pose",
        subject_ref="scene://box-17",
    )

    assert selected is not None
    assert selected.id == box_17.id


def test_artifact_resolve_and_publish_stay_inside_execution_workspace(
    tmp_path: Path,
) -> None:
    """Artifact 只通过引用和当前 Execution 工作区流转。"""

    context = MockSkillContext(
        DemoInput(target_ref="scene://box-17"),
        workspace=tmp_path / "execution",
    )
    source = context.add_artifact(
        "artifact://target-image",
        content=b"image-data",
        media_type="image/png",
        summary="目标图像",
        filename="target.png",
    )

    resolved = context.resolve_artifact("artifact://target-image")
    assert resolved == source
    assert Path(resolved.local_path).read_bytes() == b"image-data"

    report = context.workspace / "outputs" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"status":"ok"}', encoding="utf-8")
    published = context.publish_artifact(
        str(report),
        "application/json",
        "Skill 完成报告",
    )

    assert published.sync_status == "available"
    assert published.local_ref.startswith("pilot-artifact://")
    assert published.server_ref is not None


def test_artifact_rejects_unauthorized_reference_and_external_file(
    tmp_path: Path,
) -> None:
    """Skill 不能读取未授权引用，也不能发布工作区外的文件。"""

    context = MockSkillContext(
        DemoInput(target_ref="scene://box-17"),
        workspace=tmp_path / "execution",
    )
    outside = tmp_path / "outside.log"
    outside.write_text("private", encoding="utf-8")

    with pytest.raises(PermissionError, match="无权读取"):
        context.resolve_artifact("artifact://not-authorized")
    with pytest.raises(PermissionError, match="工作区"):
        context.publish_artifact(str(outside), "text/plain", "不应发布")


def test_rpc_agent_request_derives_response_schema_from_model() -> None:
    """Skill只声明Pydantic Model，SDK自动向Pilot提供本次回复契约。"""

    peer = RecordingPeer()
    context = RpcSkillContext(
        peer,  # type: ignore[arg-type]
        execution_id="rex-agent",
        robot_ref="robot://r1pro/1",
        skill_input={},
        checkpoint=None,
        controllers={},
        log_fields={},
    )

    decision = asyncio.run(
        context.request_agent(
            "decision-1",
            "需要Robot Agent决定",
            {"allowed_actions": ["abort_subtask"]},
            DemoAgentDecision,
        )
    )

    assert decision.action == "abort_subtask"
    method, params = peer.calls[-1]
    assert method == "agent.request"
    assert params["response_model"] == "DemoAgentDecision"
    schema = params["response_schema"]
    assert isinstance(schema, dict)
    assert schema["required"] == ["action", "reason"]
    assert set(schema["properties"]) == {"action", "reason"}


def test_rpc_stop_cancels_normal_results_and_decisions_but_allows_hold() -> None:
    class StopPeer(RecordingPeer):
        def call(self, method, params):
            self.calls.append((method, params))
            if method == "action.start":
                assert params["stop_action"] is True
                return {"action_id": "hold"}
            if method == "action.result":
                context.request_stop()
                return {"status": "succeeded", "physical_effect": "confirmed"}
            raise AssertionError(method)

    peer = StopPeer()
    context = RpcSkillContext(peer, execution_id="e", robot_ref="r", skill_input={},
                              checkpoint=None, controllers={}, log_fields={})
    with pytest.raises(SkillCancelled):
        asyncio.run(RpcActionHandle(context, "move", "move").result())
    with pytest.raises(SkillCancelled):
        asyncio.run(context.request_agent("decision", "retry", {}, DemoAgentDecision))
    action = Action(type="gripper.hold_object", parameters={}, timeout_seconds=3)
    result = asyncio.run(context.execute_stop("stop", action))
    assert result.status == "succeeded"
    assert result.physical_effect == "confirmed"
    assert all(method != "agent.request" for method, _ in peer.calls)


def test_rpc_artifact_methods_only_send_reference_path_and_metadata() -> None:
    """Worker 管道只传引用、路径和元数据，文件内容由 Pilot 单独同步。"""

    peer = RecordingPeer()
    context = RpcSkillContext(
        peer,  # type: ignore[arg-type]
        execution_id="rex-1",
        robot_ref="robot://r1pro/1",
        skill_input={},
        checkpoint=None,
        controllers={},
        log_fields={},
    )

    resolved = context.resolve_artifact("artifact://input-1")
    published = context.publish_artifact(
        "/pilot/executions/rex-1/outputs/report.json",
        "application/json",
        "完成报告",
    )

    assert resolved.size_bytes == 128
    assert published.sync_status == "pending"
    assert peer.calls == [
        (
            "artifact.resolve",
            {"execution_id": "rex-1", "ref": "artifact://input-1"},
        ),
        (
            "artifact.publish",
            {
                "execution_id": "rex-1",
                "path": "/pilot/executions/rex-1/outputs/report.json",
                "media_type": "application/json",
                "summary": "完成报告",
            },
        ),
    ]


def test_stage_report_keeps_skill_expectation_and_recovery_decision() -> None:
    """阶段摘要由 Skill 形成，Pilot 和 Studio 不应根据 Action 名猜测。"""

    context = MockSkillContext(DemoInput(target_ref="scene://box-17"))
    context.report(
        "stage.recovering",
        summary="路线被阻塞，准备重新规划",
        stage="navigate",
        stage_status="recovering",
        expectation="Robot 到达目标且所持物体保持稳定",
        observation_summary="前方 0.4 米检测到障碍",
        deviation="当前路线无法继续",
        progress=0.45,
        next_step="stop_and_replan",
        evidence_refs=["artifact://navigation/blocked-frame"],
    )

    report = context.events[-1]
    assert report["stage"] == "navigate"
    assert report["expectation"] == "Robot 到达目标且所持物体保持稳定"
    assert report["next_step"] == "stop_and_replan"
