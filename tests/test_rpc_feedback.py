"""真实 RPC ActionHandle 的空反馈与终态契约，不依赖 MockHandle 的队列行为。"""
import asyncio

import pytest

from semantic_robot_skill_sdk.models import SkillCancelled
from semantic_robot_skill_sdk.rpc_context import RpcActionHandle


class Context:
    def __init__(self, pages):
        self.pages = iter(pages)
        self._feedback = {}
        self.calls = []
        self.cancelled = False

    def _call(self, method, params):
        self.calls.append((method, params))
        return next(self.pages)

    def _remember(self, observations):
        pass

    def check_cancelled(self):
        if self.cancelled:
            raise SkillCancelled("stop")


def test_empty_running_page_does_not_end_feedback_stream():
    ctx = Context([
        {"feedback": [], "terminal": False},
        {"feedback": [{"sequence": 1, "status": "running",
                       "measurements": {"phase": "control"}}], "terminal": False},
        {"feedback": [], "terminal": True},
    ])

    async def collect():
        return [item async for item in RpcActionHandle(ctx, "policy", "action").feedback()]

    items = asyncio.run(collect())
    assert [item.measurements for item in items] == [{"phase": "control"}]
    assert ctx.calls[-1][1]["after_sequence"] == 1


def test_cancel_is_observed_while_waiting_for_first_feedback():
    ctx = Context([{"feedback": [], "terminal": False}])
    ctx.cancelled = True

    async def collect():
        return [item async for item in RpcActionHandle(ctx, "policy", "action").feedback()]

    with pytest.raises(SkillCancelled):
        asyncio.run(collect())
