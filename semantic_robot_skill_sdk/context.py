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

"""Robot Skill Python 脚本所依赖的最小 Context 协议。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterator, Protocol, TypeVar

from pydantic import BaseModel

from .models import (
    Action,
    ActionFeedback,
    ActionResult,
    Observation,
    PublishedArtifact,
    ResolvedArtifact,
    StopOutcome,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


class ActionHandle(Protocol):
    """流式 Action 的轻量句柄，不是新的流程或领域对象。"""

    def feedback(self) -> AsyncIterator[ActionFeedback]: ...

    async def stop(self, reason: str) -> ActionResult: ...

    async def result(self) -> ActionResult: ...


class SkillContext(Protocol):
    """三个示例实际使用的 Runtime SDK。"""

    robot_ref: str

    def input(self, model: type[ModelT]) -> ModelT: ...

    def load_state(self, model: type[ModelT], default: ModelT) -> ModelT: ...

    def checkpoint(self, state: BaseModel) -> None: ...

    def observation(self, observation_ref: str) -> Observation | None: ...

    def latest_observation(
        self,
        kind: str,
        *,
        subject_ref: str | None = None,
        max_age_ms: int | None = None,
    ) -> Observation | None: ...

    def controller(self, name: str) -> Any: ...

    async def execute(self, key: str, action: Action) -> ActionResult: ...

    async def start_action(self, key: str, action: Action) -> ActionHandle: ...

    async def request_agent(
        self,
        key: str,
        reason: str,
        context: dict[str, Any],
        response_model: type[ModelT],
    ) -> ModelT: ...

    def recent_evidence(self) -> list[str]: ...

    def resolve_artifact(self, ref: str) -> ResolvedArtifact: ...

    def publish_artifact(
        self,
        path: str,
        media_type: str,
        summary: str,
    ) -> PublishedArtifact: ...

    def check_cancelled(self) -> None: ...

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
    ) -> None: ...

    async def execute_stop(self, key: str, action: Action) -> ActionResult: ...

    def stop_outcome(
        self,
        *,
        safe: bool,
        summary: str,
        physical_state: str,
        requires_intervention: bool = False,
        evidence_refs: list[str] | None = None,
    ) -> StopOutcome: ...

    def complete(self, result: BaseModel) -> None: ...

    def fail(
        self,
        code: str,
        message: str,
        evidence: list[str] | None = None,
    ) -> None: ...

    def log(self, level: str, message: str, **fields: Any) -> None: ...

    def now(self) -> datetime: ...
