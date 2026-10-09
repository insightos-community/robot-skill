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

"""Small execution helper embedded in each standalone Skill archive."""

from semantic_robot_skill_sdk import Action, ActionResult, SkillContext

from .models import State


class ActionFailed(Exception):
    """The failure has already been reported through the Skill protocol."""


async def execute(
    ctx: SkillContext, state: State, key: str, action: Action
) -> ActionResult:
    ctx.check_cancelled()
    result = state.results.get(key)
    if result is None:
        state.stage = key
        ctx.checkpoint(state)
        ctx.report(
            "stage.running",
            stage=key,
            stage_status="running",
            summary=action.label or action.type,
        )
        result = await ctx.execute(key=key, action=action)
        ctx.check_cancelled()
        state.results[key] = result
        state.evidence_refs = list(
            dict.fromkeys(state.evidence_refs + result.evidence_refs)
        )
        ctx.checkpoint(state)
    if result.status != "succeeded":
        ctx.fail(
            result.error_code or "ACTION_NOT_COMPLETED",
            result.error_message or f"{action.type}: {result.status}",
            state.evidence_refs,
        )
        raise ActionFailed
    ctx.report(
        "stage.completed",
        stage=key,
        stage_status="completed",
        summary=action.label or action.type,
        evidence_refs=result.evidence_refs,
    )
    return result
