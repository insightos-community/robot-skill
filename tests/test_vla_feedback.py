"""验证 VLA 证据限频和失败留证，不加载模型或改变原生验收。"""
import asyncio
import pytest
from types import SimpleNamespace

from semantic_robot_skills.skills.vla_manipulation.scripts import skill
from semantic_robot_skills.skills.vla_manipulation.scripts.models import ManipulationInput, ManipulationState


@pytest.mark.parametrize("phase", ["control", "inference"])
def test_policy_feedback_records_periodic_and_failure_images(monkeypatch, phase):
    clock = [0.0]
    observations, reports = [], []

    async def observe(_ctx, _inputs, key, _stage, capture=False):
        observations.append((key, capture))
        return {"artifact_refs": ["artifact://image"] if capture else []}

    class Handle:
        async def feedback(self):
            for index in range(3):
                clock[0] = [0, 1, 3][index]
                yield SimpleNamespace(measurements={"phase": phase, "executed_actions": index, "progress": index / 3})
            # 同批旧进度立即到达，应直接消费而不积压新的场景查询。
            for index in range(100):
                yield SimpleNamespace(measurements={"phase": phase, "executed_actions": index})

        async def result(self):
            return SimpleNamespace(status="failed", output=None, error_code="MODEL_FAILED", error_message="模型失败", evidence_refs=[])

    class Context:
        def input(self, _):
            return ManipulationInput(objective="native_task", instruction="pick bowl")

        def load_state(self, *_, **__):
            return ManipulationState(stage="execute_policy", generation=1, robot_id="franka-0")

        def check_cancelled(self): pass
        def checkpoint(self, _): pass
        def report(self, event, **fields): reports.append((event, fields))
        def recent_evidence(self): return ["artifact://image"]
        def fail(self, code, message, refs): assert (code, refs) == ("MODEL_FAILED", ["artifact://image"])
        async def start_action(self, *_): return Handle()

    # asyncio 调度也调用 time.monotonic；只替换 Skill 本地使用的时钟对象。
    monkeypatch.setattr(skill, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(skill, "observe", observe)
    monkeypatch.setattr(skill, "assess", lambda *_: False)
    asyncio.run(skill.run(Context()))
    assert [capture for _, capture in observations] == [True, False, True, True]
    assert len([r for _, r in reports if "progress" in r]) == 2
    if phase == "inference":
        assert any("等待模型输出" in r.get("summary", "") for _, r in reports)
