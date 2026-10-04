"""Stable action keys for ordered placement motions."""
from semantic_robot_skill_sdk import Action, ActionResult, SkillContext
from .models import State


class ActionFailed(Exception):
    pass


async def execute(ctx: SkillContext, state: State, key: str, action: Action) -> ActionResult:
    ctx.check_cancelled()
    result = state.results.get(key)
    if result is None:
        state.stage = key
        ctx.checkpoint(state)
        ctx.report("stage.running", stage=key, stage_status="running", summary=action.label or action.type)
        result = await ctx.execute(key=key, action=action)
        ctx.check_cancelled()
        state.results[key] = result
        state.evidence_refs = list(dict.fromkeys(state.evidence_refs + result.evidence_refs))
        ctx.checkpoint(state)
    if result.status != "succeeded":
        ctx.fail(result.error_code or "ACTION_NOT_COMPLETED",
                 result.error_message or f"{action.type}: {result.status}", state.evidence_refs)
        raise ActionFailed
    ctx.report("stage.completed", stage=key, stage_status="completed", summary=action.type,
               evidence_refs=result.evidence_refs)
    return result
