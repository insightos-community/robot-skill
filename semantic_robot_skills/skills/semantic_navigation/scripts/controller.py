"""语义导航 Stage 使用的纯 Action 构造和反馈判断函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from semantic_robot_skill_sdk import Action, ActionFeedback, FeedbackRequest

from .models import (
    CarryingObjectState,
    FollowRouteParameters,
    GetRobotStateParameters,
    LocateObjectParameters,
    PlanRouteParameters,
    SafeStopParameters,
    SemanticNavigationInput,
    SemanticNavigationState,
    VerifyToolLoadParameters,
    VerifyArrivalParameters,
)


LOCAL_REPLAN_LIMIT = 1


@dataclass(frozen=True)
class FeedbackAssessment:
    decision: Literal["continue", "replan", "critical_stop"]
    reason: str | None = None


def build_plan_route_action(skill_input: SemanticNavigationInput, state: SemanticNavigationState) -> Action:
    if state.target is None:
        raise ValueError("路线规划前必须先复核目标")
    carrying_object = state.carrying_object
    return Action.from_model(
        action_type="navigation.plan_route",
        parameters=PlanRouteParameters(
            target=state.target,
            navigation_purpose=skill_input.navigation_purpose,
            maximum_speed_mps=_effective_speed(
                skill_input, carrying_object, replan_count=state.replan_count
            ),
            minimum_clearance_m=_effective_clearance(skill_input, carrying_object),
            carrying_object=carrying_object,
        ),
        timeout_seconds=15,
        label="规划语义导航路线",
    )


def build_follow_route_action(skill_input: SemanticNavigationInput, state: SemanticNavigationState) -> Action:
    if state.route_ref is None:
        raise ValueError("路线执行前必须先获得路线引用")
    carrying_object = state.carrying_object
    return Action.from_model(
        action_type="navigation.follow_route",
        parameters=FollowRouteParameters(
            route_ref=state.route_ref,
            navigation_purpose=skill_input.navigation_purpose,
            maximum_speed_mps=_effective_speed(
                skill_input, carrying_object, replan_count=state.replan_count
            ),
            minimum_clearance_m=_effective_clearance(skill_input, carrying_object),
            carrying_object=carrying_object,
        ),
        timeout_seconds=180,
        feedback=FeedbackRequest(
            observation_kinds=[
                "navigation_progress",
                "route_blocked",
                "localization_state",
                "navigation.carrying_load",
            ],
            interval_ms=250,
        ),
        label="沿规划路线导航",
    )


def build_verify_arrival_action(skill_input: SemanticNavigationInput, state: SemanticNavigationState) -> Action:
    if state.target is None:
        raise ValueError("到达验证前必须保留目标")
    return Action.from_model(
        action_type="navigation.verify_arrival",
        parameters=VerifyArrivalParameters(
            target=state.target,
            navigation_purpose=skill_input.navigation_purpose,
            arrival_radius_m=skill_input.arrival_radius_m,
            require_visual_confirmation=False,
            carrying_object=state.carrying_object,
        ),
        timeout_seconds=15,
        label="验证语义导航到达状态",
    )


def build_get_robot_state_action() -> Action:
    return Action.from_model(
        action_type="robot.get_state",
        parameters=GetRobotStateParameters(),
        timeout_seconds=8,
        label="读取当前Robot与工具状态",
    )


def build_verify_tool_load_action(tool_refs: tuple[str, ...]) -> Action:
    return Action.from_model(
        action_type="robot.verify_tool_load",
        parameters=VerifyToolLoadParameters(tool_refs=tool_refs),
        timeout_seconds=8,
        label="连续验证当前工具承载",
    )


def build_locate_object_action(object_ref: str) -> Action:
    return Action.from_model(
        action_type="perception.locate_object",
        parameters=LocateObjectParameters(object_ref=object_ref),
        timeout_seconds=8,
        label="重新观测携带物体",
    )


def build_safe_stop_action(*, reason: str, mode: str) -> Action:
    return Action.from_model(
        action_type="navigation.follow_route",
        parameters=SafeStopParameters(reason=reason, mode=mode),
        timeout_seconds=10,
        label="停止语义导航",
    )


def assess_follow_feedback(feedback: ActionFeedback, state: SemanticNavigationState) -> FeedbackAssessment:
    if feedback.severity == "critical":
        return FeedbackAssessment("critical_stop", feedback.message or "执行侧检测到严重异常")
    for observation in feedback.observations:
        value = observation.value or {}
        # 两类正式 Observation 有不同 schema，必须按 kind 解释，不能用一组
        # 兼容别名兜底。否则字段缺失会被误当成“没有异常”，掩盖携物风险。
        if observation.kind == "navigation.carrying_load":
            if (
                value.get("stable") is not True
                or value.get("slipping") is True
                or value.get("overloaded") is True
                or value.get("sensor_fault") is True
            ):
                return FeedbackAssessment("critical_stop", "持物状态异常，执行侧已经停止底盘")
        if observation.kind == "localization_state" and not bool(value.get("valid")):
            return FeedbackAssessment("critical_stop", "定位状态失效，无法安全继续")
        if observation.kind == "route_blocked" and bool(value.get("blocked")):
            return FeedbackAssessment("replan", "当前路线被阻塞")
    return FeedbackAssessment("continue")


def _effective_speed(
    skill_input: SemanticNavigationInput,
    carrying_object: CarryingObjectState | None,
    *,
    replan_count: int,
) -> float:
    if carrying_object is None:
        return skill_input.maximum_speed_mps
    # Agent 给出的速度是业务期望上限；携物动作还必须服从本 Skill 已验证的
    # 安全上限。Robot Profile/SDK 会在此基础上继续应用设备级运动缩放。
    speed = min(skill_input.maximum_speed_mps, 0.05)
    # 携物路线已经因真实相对滑动停止时，下一次动作不能原速重放。
    # replan_count 是现有 Skill 局部恢复状态；这里只降低一次底盘速度，
    # 不新增仿真参数，也不改变接触、过载或传感故障的停止边界。
    return speed * 0.5 if replan_count > 0 else speed


def _effective_clearance(skill_input: SemanticNavigationInput, carrying_object: CarryingObjectState | None) -> float:
    if carrying_object is None:
        return skill_input.minimum_clearance_m
    object_radius = max(carrying_object.object_size_m[:2]) / 2
    return max(skill_input.minimum_clearance_m, object_radius + 0.1)
