"""放置 Robot Skill 的可执行契约测试。"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


DEPALLETIZING_ROOT = Path(__file__).resolve().parents[2]
if str(DEPALLETIZING_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPALLETIZING_ROOT))

from semantic_robot_skills.skills.place_object.scripts.controller import PlacementController  # noqa: E402
from semantic_robot_skills.skills.place_object.scripts.models import (  # noqa: E402
    PlaceObjectInput,
    HeldObjectState,
    PlaceObjectRunState,
    PlacementConstraints,
    PlacementSlotState,
    PlacementVerificationObservation,
    PlacementTarget,
    PlacedObjectState,
)
from semantic_robot_skills.skills.place_object.scripts.skill import (  # noqa: E402
    GET_ROBOT_STATE_ACTION,
    MOVE_ACTION,
    MOVE_TO_POSTURE_ACTION,
    OBSERVE_SLOT_ACTION,
    RELEASE_ACTION,
    SAFE_STOP_ACTION,
    LOCATE_OBJECT_ACTION,
    VERIFY_STABILITY_ACTION,
    VERIFY_TOOL_LOAD_ACTION,
    on_stop,
    run,
)
from semantic_robot_skill_sdk import (  # noqa: E402
    ActionResult,
    MockSkillContext,
    Observation,
    Pose3D,
    SkillFailure,
    StopRequest,
    run_skill,
)


def _now() -> datetime:
    """构造带时区时间，避免测试产生无时区观测。"""

    return datetime.now(timezone.utc)


def _pose(revision: str, *, z: float = 0.6) -> Pose3D:
    """创建测试位姿；数值只存在于 Mock 观测，不是 Skill 默认参数。"""

    return Pose3D(
        frame_id="map",
        position_m=(1.2, 0.8, z),
        revision=revision,
        observed_at=_now(),
    )


def _held_state() -> HeldObjectState:
    """模拟place-object在本Execution中组合出的局部HeldObjectState。"""

    return HeldObjectState(
        object_ref="object://box-17",
        robot_ref="robot://demo-1",
        tool_refs=("component://tool/left", "component://tool/right"),
        tool_poses={
            "component://tool/left": _pose("held-tools", z=0.65).model_copy(update={"position_m": (1.2, 1.05, 0.65)}),
            "component://tool/right": _pose("held-tools", z=0.65).model_copy(update={"position_m": (1.2, 0.55, 0.65)}),
        },
        object_pose=_pose("object-7"),
        object_size_m=(0.4, 0.3, 0.2),
        base_position_m=(1.2, -0.2, 0.0),
        robot_state_generation=20,
        evidence_refs=["artifact://grasp-proof"],
    )


def _target() -> PlacementTarget:
    """创建 Robot Agent 给出的语义放置目标。"""

    return PlacementTarget(
        target_ref="slot://pallet-b/cell-07",
        stability_duration_ms=800,
    )


def _slot_observation(*, revision: str, free: bool = True) -> Observation:
    """创建由感知或码垛规划器形成的槽位观测。"""

    slot = PlacementSlotState(
        target_ref=_target().target_ref,
        region_ref="region://pallet-b",
        schema_version=2,
        placement_pose=_pose(revision, z=0.7),
        approach_vector=(0.0, 0.0, 1.0),
        free=free,
        reachable=True,
        occupants=[],
        support_surface_ref="surface://pallet-b",
        revision=revision,
        confidence=0.97,
        evidence_refs=[f"artifact://slot-{revision}"],
    )
    return Observation(
        kind="placement.target_slot",
        subject_ref=slot.target_ref,
        source="pallet-planner",
        revision=revision,
        frame_id="map",
        confidence=slot.confidence,
        value=slot.model_dump(mode="json"),
        evidence_refs=slot.evidence_refs,
    )


def test_place_local_alignment_limit_comes_from_live_object_geometry() -> None:
    """局部接近范围由箱体足迹和既有接近余量共同确定。"""

    controller = PlacementController()
    held = _held_state().model_copy(
        update={"object_size_m": (0.6, 0.4, 0.34)}
    )
    slot = PlacementSlotState(
        target_ref=_target().target_ref,
        region_ref="region://pallet-b",
        placement_pose=_pose("far-slot", z=0.7).model_copy(
            update={"position_m": (1.55, 0.8, 0.7)}
        ),
        approach_vector=(0.0, 0.0, 1.0),
        free=True,
        reachable=True,
        support_surface_ref="surface://pallet-b",
        revision="far-slot",
        confidence=1.0,
    )

    error = controller.horizontal_alignment_error_m(held, slot)
    limit = controller.local_alignment_limit_m(held, PlacementConstraints())
    assert abs(error - 0.35) < 1e-9
    assert limit == max(held.object_size_m[:2]) / 2 + PlacementConstraints().approach_clearance_m
    assert limit == 0.42
    assert error < limit


def test_preplace_aligns_horizontally_before_descending() -> None:
    """密集槽位先在当前持物高度对齐，再沿接近方向下降。"""

    held = _held_state().model_copy(update={
        "tool_poses": {
            "component://tool/left": _pose("held-tools", z=1.15).model_copy(
                update={"position_m": (0.95, 0.8, 1.15)}
            ),
            "component://tool/right": _pose("held-tools", z=1.15).model_copy(
                update={"position_m": (0.95, 0.3, 1.15)}
            ),
        },
        "object_pose": _pose("held-high", z=1.1).model_copy(
            update={"position_m": (0.95, 0.55, 1.1)}
        ),
    })
    slot = PlacementSlotState.model_validate(
        _slot_observation(revision="scene-dense").value
    )
    plan = PlacementController().build_approach_plan(
        held,
        slot,
        PlacementConstraints(approach_clearance_m=0.12),
    )

    preplace = plan.waypoints[0].object_pose
    release = plan.waypoints[1].object_pose
    assert preplace.position_m[:2] == slot.placement_pose.position_m[:2]
    assert preplace.position_m[2] == held.object_pose.position_m[2]
    assert release.position_m[:2] == slot.placement_pose.position_m[:2]
    assert release.position_m[2] == slot.placement_pose.position_m[2]
    retreat = plan.waypoints[-2]
    # 放置后的空工具抬回本次已实际到达过的preplace高度，再折叠travel；
    # 不能只抬半个箱高，否则会扫到目标列附近仍堆叠的较高物体。
    for target in retreat.targets:
        release_target = next(
            item for item in plan.waypoints[1].targets
            if item.tool_ref == target.tool_ref
        )
        assert target.target_pose.position_m[2] == pytest.approx(
            release_target.target_pose.position_m[2] + held.object_size_m[2] - 0.008
        )


def test_long_descent_keeps_only_object_scaled_final_landing_slow() -> None:
    """长距离下降增加快速中点，进入同层邻箱高度前切回低速动作。"""

    held = _held_state().model_copy(update={
        "object_pose": _pose("held-high", z=1.08),
        "object_size_m": (0.6, 0.4, 0.34),
        "tool_poses": {
            "component://tool/left": _pose("tools-high", z=1.13).model_copy(
                update={"position_m": (1.2, 1.093, 1.13)}
            ),
            "component://tool/right": _pose("tools-high", z=1.13).model_copy(
                update={"position_m": (1.2, 0.507, 1.13)}
            ),
        },
    })
    slot = PlacementSlotState.model_validate(
        _slot_observation(revision="scene-long-descent").value
    ).model_copy(update={
        "placement_pose": _pose("scene-long-descent", z=0.32),
    })
    controller = PlacementController()
    plan = controller.build_approach_plan(held, slot, PlacementConstraints())

    landing = controller.build_free_space_descent_waypoint(
        plan.waypoints[0],
        plan.waypoints[1],
        slot.approach_vector,
        object_height_m=held.object_size_m[2],
    )

    assert landing is not None
    assert landing.name == "landing"
    assert abs(landing.object_pose.position_m[2] - 0.685) < 1e-9
    release_targets = {item.tool_ref: item for item in plan.waypoints[1].targets}
    for target in landing.targets:
        release = release_targets[target.tool_ref]
        assert abs(
            target.target_pose.position_m[2]
            - release.target_pose.position_m[2]
            - 0.365
        ) < 1e-9

    short_plan = controller.build_approach_plan(
        _held_state(),
        PlacementSlotState.model_validate(
            _slot_observation(revision="scene-short-descent").value
        ),
        PlacementConstraints(),
    )
    assert controller.build_free_space_descent_waypoint(
        short_plan.waypoints[0],
        short_plan.waypoints[1],
        (0.0, 0.0, 1.0),
        object_height_m=_held_state().object_size_m[2],
    ) is None


def test_retreat_disengages_hooks_before_vertical_clearance() -> None:
    """工具沿槽长偏置时仍沿双工具横轴退出，不走错误的中心径向。"""

    held = _held_state().model_copy(update={
        "object_size_m": (0.6, 0.4, 0.34),
        "tool_poses": {
            "component://tool/left": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (0.907, 0.72, 0.65)}
            ),
            "component://tool/right": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (1.493, 0.72, 0.65)}
            ),
        },
    })
    slot = PlacementSlotState.model_validate(
        _slot_observation(revision="scene-42").value
    )
    constraints = PlacementConstraints(retreat_disengage_m=0.06)
    plan = PlacementController().build_approach_plan(held, slot, constraints)

    assert [waypoint.name for waypoint in plan.waypoints] == [
        "preplace", "release", "unseat", "disengage", "retreat",
        "travel_clearance",
    ]
    release = {target.tool_ref: target.target_pose for target in plan.waypoints[1].targets}
    unseat = {target.tool_ref: target.target_pose for target in plan.waypoints[2].targets}
    disengage = {target.tool_ref: target.target_pose for target in plan.waypoints[3].targets}
    retreat = {target.tool_ref: target.target_pose for target in plan.waypoints[4].targets}
    travel_clearance = {
        target.tool_ref: target.target_pose for target in plan.waypoints[5].targets
    }
    left_release = release[held.tool_refs[0]].position_m
    right_release = release[held.tool_refs[1]].position_m
    across = tuple(right_release[index] - left_release[index] for index in range(3))
    across_length = sum(value * value for value in across) ** 0.5
    across_axis = tuple(value / across_length for value in across)
    for tool_ref in held.tool_refs:
        relative = tuple(
            release[tool_ref].position_m[index] - slot.placement_pose.position_m[index]
            for index in range(3)
        )
        sign = 1.0 if sum(
            value * axis for value, axis in zip(relative, across_axis, strict=True)
        ) >= 0.0 else -1.0
        direction = tuple(sign * axis for axis in across_axis)
        displacement = sum(
            (disengage[tool_ref].position_m[index] - release[tool_ref].position_m[index])
            * direction[index]
            for index in range(3)
        )
        assert abs(unseat[tool_ref].position_m[2] - release[tool_ref].position_m[2] + 0.008) < 1e-9
        assert disengage[tool_ref].position_m[2] == unseat[tool_ref].position_m[2]
        # load frame原本深入边界7mm；退出再覆盖17mm下钩前伸和
        # 8mm安全余量，因此横向位移为32mm。
        assert abs(displacement - 0.032) < 1e-9
        assert displacement < constraints.retreat_disengage_m
        assert disengage[tool_ref].position_m[1] == release[tool_ref].position_m[1]
        assert retreat[tool_ref].position_m[:2] == disengage[tool_ref].position_m[:2]
        assert abs(
            retreat[tool_ref].position_m[2]
            - disengage[tool_ref].position_m[2] - 0.34
        ) < 1e-9

        assert travel_clearance[tool_ref].position_m[2] == retreat[tool_ref].position_m[2]
        # 箱体在Robot方向的半尺寸为20cm；空夹具移出该投影后还覆盖
        # 25mm退钩净空、52mm夹具包络和80mm折叠通道余量。
        assert abs(
            slot.placement_pose.position_m[1]
            - travel_clearance[tool_ref].position_m[1]
            - 0.357
        ) < 1e-9


def test_dense_column_uses_supported_slide_before_releasing_inner_tool() -> None:
    """邻箱净空不足时，先偏置落座并撤内侧手，再由外侧手推回中心。"""

    held = _held_state().model_copy(update={
        "object_size_m": (0.6, 0.4, 0.34),
        "tool_poses": {
            "component://tool/left": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (0.907, 0.8, 0.65)}
            ),
            "component://tool/right": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (1.493, 0.8, 0.65)}
            ),
        },
    })
    slot = PlacementSlotState.model_validate(
        _slot_observation(revision="scene-supported-slide").value
    ).model_copy(update={
        "lateral_clearance_m": {"negative": 0.02, "positive": None},
    })
    plan = PlacementController().build_approach_plan(
        held, slot, PlacementConstraints()
    )

    assert plan.early_release_tool_ref == "component://tool/left"
    assert len(plan.early_release_waypoints) == 4
    staged = plan.early_release_waypoints[0]
    assert staged.object_pose.position_m[0] > slot.placement_pose.position_m[0]
    # 临时偏置只补足实时净空相对52mm夹具包络的缺口，不再叠加固定3cm；
    # 既让打开后的压片避开邻箱，也避免箱体落座后依赖空行程推回中心。
    assert abs(
        staged.object_pose.position_m[0]
        - slot.placement_pose.position_m[0]
        - 0.039
    ) < 1e-9
    # 理论支撑面只负责几何落点；相邻列逐侧释放前再做5mm低速寻底，
    # 覆盖接触与伺服误差，同时避免把已经落座的箱体继续向托盘内压入。
    assert staged.object_pose.position_m[2] == pytest.approx(
        slot.placement_pose.position_m[2] - 0.005
    )
    assert plan.waypoints[0].object_pose.position_m[0] == staged.object_pose.position_m[0]
    assert plan.waypoints[1].object_pose.position_m == slot.placement_pose.position_m
    staged_targets = {target.tool_ref: target for target in staged.targets}
    for waypoint in plan.early_release_waypoints[1:]:
        targets = {target.tool_ref: target for target in waypoint.targets}
        assert set(targets) == {"component://tool/left"}
    assert "component://tool/right" in staged_targets
    # 外侧工具推箱前，内侧空钩先反向落座、退出凹槽，再从临时偏置位
    # 直接抬过箱沿；不能沿凹槽方向朝基座滑动或挤入相邻箱体。
    early_disengage = plan.early_release_waypoints[-2].targets[0].target_pose
    early_clearance = plan.early_release_waypoints[-1].targets[0].target_pose
    assert early_clearance.position_m[:2] == early_disengage.position_m[:2]
    assert early_clearance.position_m[2] == pytest.approx(
            early_disengage.position_m[2] + 0.178
    )


def test_early_withdrawal_rebases_on_live_post_support_pose() -> None:
    """落座误差不能吃掉下钩反向退出凹槽的实际行程。"""

    held = _held_state().model_copy(update={
        "object_size_m": (0.6, 0.4, 0.34),
        "tool_poses": {
            "component://tool/left": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (0.907, 0.8, 0.65)}
            ),
            "component://tool/right": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (1.493, 0.8, 0.65)}
            ),
        },
    })
    slot = PlacementSlotState.model_validate(
        _slot_observation(revision="scene-live-rebase").value
    ).model_copy(update={
        "lateral_clearance_m": {"negative": 0.02, "positive": None},
    })
    controller = PlacementController()
    plan = controller.build_approach_plan(held, slot, PlacementConstraints())
    support, *withdrawal = plan.early_release_waypoints
    planned_left = next(
        target.target_pose for target in support.targets
        if target.tool_ref == plan.early_release_tool_ref
    )
    actual_left = planned_left.model_copy(update={
        "position_m": (
            planned_left.position_m[0] + 0.003,
            planned_left.position_m[1],
            planned_left.position_m[2] + 0.0045,
        )
    })

    rebased = controller.rebase_early_withdrawal(
        support,
        withdrawal,
        tool_ref=plan.early_release_tool_ref,
        current_pose=actual_left,
    )

    unseat = rebased[0].targets[0].target_pose
    planned_unseat = withdrawal[0].targets[0].target_pose
    planned_delta = tuple(
        target - source
        for target, source in zip(
            planned_unseat.position_m, planned_left.position_m, strict=True
        )
    )
    actual_delta = tuple(
        target - source
        for target, source in zip(
            unseat.position_m, actual_left.position_m, strict=True
        )
    )
    assert actual_delta == pytest.approx(planned_delta)
    assert all(len(waypoint.targets) == 1 for waypoint in rebased)


def test_early_withdrawal_skips_unseat_when_open_hook_is_already_clear() -> None:
    """开夹后下钩已无接触时，不应再向箱沿执行反向落座。"""

    held = _held_state().model_copy(update={
        "object_size_m": (0.6, 0.4, 0.34),
        "tool_poses": {
            "component://tool/left": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (0.907, 0.8, 0.65)}
            ),
            "component://tool/right": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (1.493, 0.8, 0.65)}
            ),
        },
    })
    slot = PlacementSlotState.model_validate(
        _slot_observation(revision="scene-skip-unseat").value
    ).model_copy(update={
        "lateral_clearance_m": {"negative": 0.02, "positive": None},
    })
    controller = PlacementController()
    plan = controller.build_approach_plan(held, slot, PlacementConstraints())
    support, *withdrawal = plan.early_release_waypoints
    early_tool = plan.early_release_tool_ref
    assert early_tool is not None
    current_pose = next(
        target.target_pose for target in support.targets
        if target.tool_ref == early_tool
    )

    rebased = controller.rebase_early_withdrawal(
        support,
        withdrawal,
        tool_ref=early_tool,
        current_pose=current_pose,
        skip_unseat=True,
    )

    assert [waypoint.name for waypoint in rebased] == [
        "disengage", "travel_clearance",
    ]
    planned_horizontal_delta = tuple(
        target - source
        for target, source in zip(
            withdrawal[1].targets[0].target_pose.position_m,
            withdrawal[0].targets[0].target_pose.position_m,
            strict=True,
        )
    )
    actual_horizontal_delta = tuple(
        target - source
        for target, source in zip(
            rebased[0].targets[0].target_pose.position_m,
            current_pose.position_m,
            strict=True,
        )
    )
    assert actual_horizontal_delta == pytest.approx(planned_horizontal_delta)
    assert actual_horizontal_delta[2] == pytest.approx(0.0)


def test_supported_slide_target_applies_live_object_error_once() -> None:
    """外侧手仍贴箱时，把箱体水平误差转换成同方向的一次短推。"""

    held = _held_state().model_copy(update={
        "object_size_m": (0.6, 0.4, 0.34),
        "tool_poses": {
            "component://tool/left": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (0.907, 0.8, 0.65)}
            ),
            "component://tool/right": _pose("held-tools", z=0.65).model_copy(
                update={"position_m": (1.493, 0.8, 0.65)}
            ),
        },
    })
    slot = PlacementSlotState.model_validate(
        _slot_observation(revision="scene-slide-proof").value
    ).model_copy(update={
        "lateral_clearance_m": {"negative": 0.02, "positive": None},
    })
    controller = PlacementController()
    plan = controller.build_approach_plan(held, slot, PlacementConstraints())
    staged = plan.early_release_waypoints[0].object_pose

    def observed(pose: Pose3D) -> Observation:
        return Observation(
            kind="target_pose",
            source="mujoco-ground-truth",
            value={
                "object_ref": held.object_ref,
                "pose": pose.model_dump(mode="json"),
                "extent_m": list(held.object_size_m),
                "identity_confidence": 1.0,
            },
        )

    planned_target = next(
        target
        for target in plan.waypoints[1].targets
        if target.tool_ref != plan.early_release_tool_ref
    )
    current_target = planned_target.model_copy(update={
        "target_pose": planned_target.target_pose.model_copy(update={
            "position_m": (0.5, 0.6, 1.0),
        }),
    })
    corrected = controller.supported_slide_correction_target(
        observed(staged),
        plan,
        current_target,
    )
    expected = plan.waypoints[1].object_pose
    assert corrected.target_pose.position_m == pytest.approx((
        current_target.target_pose.position_m[0]
        + expected.position_m[0] - staged.position_m[0],
        current_target.target_pose.position_m[1]
        + expected.position_m[1] - staged.position_m[1],
        current_target.target_pose.position_m[2],
    ))
    assert corrected.target_pose.orientation_xyzw == (
        current_target.target_pose.orientation_xyzw
    )


def test_dense_column_keeps_bilateral_load_until_support_transfer() -> None:
    """邻列占用不能在箱体尚未获得支撑时触发单侧释放。"""

    held = _held_state()
    slot = PlacementSlotState.model_validate(
        _slot_observation(revision="scene-dense").value
    ).model_copy(update={
        "support_center_pose": _pose("pallet-b").model_copy(
            update={"position_m": (1.2, 1.5, 0.075)}
        ),
        "support_occupied": True,
    })
    plan = PlacementController().build_approach_plan(
        held,
        slot,
        PlacementConstraints(),
    )

    assert plan.early_release_tool_ref is None
    assert plan.early_release_waypoints == []


def _release_observation(tool_ref: str, *, gripper_empty: bool) -> Observation:
    return Observation(
        kind="manipulation.object_released",
        subject_ref="object://box-17",
        source="pilot.end-effector",
        revision=f"released-{tool_ref}",
        value={
            "object_ref": "object://box-17",
            "tools": {tool_ref: {"hook_contact": False, "clamp_contact": False}},
            "released": True,
            "released_tool_refs": [tool_ref],
            "gripper_empty": gripper_empty,
        },
    )


def _support_transfer_observation(held: HeldObjectState) -> Observation:
    return Observation(
        kind="placement.carrying_load",
        subject_ref=held.object_ref,
        source="robot-sdk://state",
        value={
            "support_transfer_confirmed": True,
            "support_transfer": {
                "object_ref": held.object_ref,
                "target_ref": _target().target_ref,
                "contact": True,
                "within_target_xy": True,
            },
        },
    )


def _robot_state(
    held: HeldObjectState,
    *,
    tool_contact: bool = True,
    tool_contacts: dict[str, bool] | None = None,
) -> Observation:
    tool_states = {}
    end_effectors = {}
    for index, ref in enumerate(held.tool_refs):
        side = "left" if index == 0 else "right"
        pose = held.tool_poses[ref]
        has_contact = (
            tool_contacts.get(ref, tool_contact)
            if tool_contacts is not None
            else tool_contact
        )
        end_effectors[side] = {
            "frame_id": pose.frame_id,
            "position": list(pose.position_m),
            "quaternion_xyzw": list(pose.orientation_xyzw),
        }
        tool_states[ref] = {
            "tool_ref": ref, "side": side, "kind": "tote_clamp",
            "position": 0.015, "velocity": 0.0, "effort": 18.0,
            "target_position": 0.015, "reached_target": True,
            "hook_contact": has_contact, "clamp_contact": has_contact,
            "sensor_fault": False,
            "hook_force_n": 12.0 if has_contact else 0.0,
            "clamp_force_n": 18.0 if has_contact else 0.0,
            "hook_support_ratio": 0.8 if has_contact else 0.0,
            "hook_tangential_speed_m_s": 0.0,
        }
    return Observation(
        kind="robot.state", source="pilot.robot-state", revision="robot-state-20",
        value={
            "robot_id": held.robot_ref, "generation": 20,
            "observed_at": _now().isoformat(),
            "base_pose": {
                "frame_id": "map", "position": [1.2, -0.2, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "joint_positions": {}, "end_effectors": end_effectors,
            "tool_states": tool_states, "in_hold": False,
            "backend": "mujoco", "firmware_profile": "simulation",
        },
    )


def _tool_load(tool_refs, *, stable: bool = True) -> Observation:
    return Observation(
        kind="robot.tool_load", source="pilot.robot-state",
        value={
            "condition_satisfied": stable,
            "observed_duration_ms": 150,
            "tools": [
                {
                    "tool_ref": ref, "available": True, "position": 0.015,
                    "velocity": 0.0, "effort": 18.0,
                    "hook_contact": stable, "clamp_contact": stable,
                    "hook_force_n": 12.0 if stable else 0.0,
                    "clamp_force_n": 18.0 if stable else 0.0,
                    "hook_support_ratio": 0.8 if stable else 0.0,
                    "relative_tangential_speed_mps": 0.0,
                    "sensor_fault": False,
                }
                for ref in tool_refs
            ],
            "slip_detected": False, "overload_detected": False,
            "sensor_fault": False, "reasons": [] if stable else ["not_engaged"],
        },
    )


def _object_pose(held: HeldObjectState) -> Observation:
    return Observation(
        kind="target_pose", subject_ref=held.object_ref, source="perception",
        revision="object-7",
        value={
            "object_ref": held.object_ref,
            "pose": held.object_pose.model_dump(mode="json"),
            "extent_m": list(held.object_size_m),
            "identity_confidence": 0.98,
        },
    )


def _queue_verify_held(context: MockSkillContext, held: HeldObjectState) -> None:
    context.queue_action(GET_ROBOT_STATE_ACTION, ActionResult(status="succeeded", observations=[_robot_state(held)], physical_effect="none"))
    context.queue_action(VERIFY_TOOL_LOAD_ACTION, ActionResult(status="succeeded", observations=[_tool_load(held.tool_refs)], physical_effect="none"))
    context.queue_action(LOCATE_OBJECT_ACTION, ActionResult(status="succeeded", observations=[_object_pose(held)], physical_effect="none"))


def test_robot_state_is_only_raw_input_for_local_held_state() -> None:
    held = _held_state()
    controller = PlacementController()
    state = controller.parse_robot_state(_robot_state(held))
    assert not hasattr(state, "stable_load")
    assert not hasattr(state, "tool_loads")
    assert controller.discover_tool_refs(state) == held.tool_refs
    assert controller.tool_pose_from_robot_state(
        state, held.tool_refs[0], revision="scene-replay"
    ).observed_at == state.observed_at


def _stability_observation(
    *,
    stable: bool = True,
    position_error_m: float = 0.01,
    orientation_error_rad: float = 0.04,
) -> Observation:
    """创建撤离后的独立稳定性观测。"""

    placed = PlacedObjectState(
        object_ref="object://box-17",
        target_ref=_target().target_ref,
        final_pose=_pose("scene-43", z=0.7),
        support_surface_ref="surface://pallet-b",
        position_error_m=position_error_m,
        orientation_error_rad=orientation_error_rad,
        stable=stable,
        gripper_empty=True,
        verification_source="perception.release-zone-verifier",
        observed_duration_ms=800,
        verified_at=_now(),
        scene_revision="scene-43",
        evidence_refs=["artifact://stability-proof"],
    )
    verified = PlacementVerificationObservation(
        state=placed,
        within_target=True,
        stable=stable,
        observed_displacement_m=0.002,
        observed_duration_ms=800,
        support_contact=True,
        gripper_empty=True,
        independent_verification=True,
    )
    return Observation(
        kind="placement.object_stability",
        subject_ref=placed.object_ref,
        source="perception.release-zone-verifier",
        revision=placed.scene_revision,
        value=verified.model_dump(mode="json"),
        evidence_refs=placed.evidence_refs,
    )


def test_support_contact_can_be_confirmed_before_remaining_tool_release() -> None:
    """释放前只确认目标支撑，不能错误要求保留侧工具已经为空。"""

    observation = _stability_observation(stable=False)
    value = dict(observation.value or {})
    state = dict(value["state"])
    state["gripper_empty"] = False
    state["stable"] = False
    value.update({"state": state, "gripper_empty": False, "stable": False})
    observation = observation.model_copy(update={"value": value})

    controller = PlacementController()
    assert controller.support_contact_confirmed(
        observation,
        object_ref="object://box-17",
        target=_target(),
    )
    supported = controller.supported_placed_state(
        observation,
        object_ref="object://box-17",
        target=_target(),
    )
    assert supported is not None
    assert supported.stable is False
    with pytest.raises(SkillFailure, match="尚未满足稳定放置完成条件"):
        controller.parse_placed_state(
            observation,
            object_ref="object://box-17",
            target=_target(),
        )


def test_supported_slide_requires_centimeter_level_column_alignment() -> None:
    """箱体只是在目标区域内静止时，不能跳过尚未完成的支撑面推进。"""

    controller = PlacementController()
    assert not controller.supported_slide_completed(
        _stability_observation(stable=False, position_error_m=0.041),
        object_ref="object://box-17",
        target=_target(),
    )
    assert controller.supported_slide_completed(
        _stability_observation(stable=False, position_error_m=0.039),
        object_ref="object://box-17",
        target=_target(),
    )
    # 回归真实边界：25.0033mm 不应因为浮点/伺服尾差被当作物理失败。
    assert controller.supported_slide_completed(
        _stability_observation(stable=False, position_error_m=0.0250328),
        object_ref="object://box-17",
        target=_target(),
    )


def test_tilted_supported_object_is_not_ready_for_final_release() -> None:
    """邻箱或工具侧面临时支住的倾斜箱体，不能只凭接触就释放。"""

    observation = _stability_observation(
        stable=False,
        orientation_error_rad=0.39,
    )
    value = dict(observation.value or {})
    state = dict(value["state"])
    state.update({"stable": False, "gripper_empty": False})
    value.update({"state": state, "stable": False, "gripper_empty": False})
    observation = observation.model_copy(update={"value": value})

    assert not PlacementController().support_contact_confirmed(
        observation,
        object_ref="object://box-17",
        target=_target(),
    )
    with pytest.raises(SkillFailure, match="尚未满足稳定放置完成条件"):
        PlacementController().parse_placed_state(
            _stability_observation(orientation_error_rad=0.39),
            object_ref="object://box-17",
            target=_target(),
        )


def test_verified_target_membership_is_not_rejected_by_duplicate_center_threshold() -> None:
    observation = _stability_observation(position_error_m=0.0312)

    placed = PlacementController().parse_placed_state(
        observation,
        object_ref="object://box-17",
        target=_target(),
    )

    assert placed.position_error_m == 0.0312


def test_target_region_membership_is_not_rejected_by_fixed_center_threshold() -> None:
    """正式Region、支撑和稳定证据成立时，不再叠加固定中心距离。"""

    observation = _stability_observation(position_error_m=0.0425)

    assert PlacementController().support_contact_confirmed(
        observation,
        object_ref="object://box-17",
        target=_target(),
    )
    assert PlacementController().parse_placed_state(
        observation,
        object_ref="object://box-17",
        target=_target(),
    ).position_error_m == 0.0425


def _new_context(
    *,
    slot: Observation,
) -> tuple[MockSkillContext, HeldObjectState]:
    """创建带 Controller 和初始槽位观测的 Mock Runtime。"""

    held = _held_state()
    skill_input = PlaceObjectInput(
        object_ref=held.object_ref,
        target=_target(),
    )
    context = MockSkillContext(skill_input)
    context.register_controller("placement", PlacementController())
    context.add_observation(slot)
    return context, held


def _queue_successful_tail(
    context: MockSkillContext,
    held: HeldObjectState,
    *,
    stability: bool = True,
    include_verify: bool = True,
    stability_attempts: int = 1,
    support_transferred: bool = False,
) -> None:
    """编排双工具接近检查、逐侧释放、撤离和独立验证。"""

    if include_verify:
        _queue_verify_held(context, held)
    for suffix in ("preplace", "release"):
        observations = (
            [_support_transfer_observation(held)]
            if suffix == "release" and support_transferred
            else []
        )
        context.queue_action(
            MOVE_ACTION,
            ActionResult(status="succeeded", observations=observations, physical_effect="confirmed"),
        )
        context.queue_action(
            VERIFY_TOOL_LOAD_ACTION,
            ActionResult(
                status="succeeded",
                observations=[_tool_load(held.tool_refs, stable=not (support_transferred and suffix == "release"))],
                physical_effect="none",
            ),
        )
    for released, tool_ref in enumerate(held.tool_refs, start=1):
        context.queue_action(
            RELEASE_ACTION,
            ActionResult(
                status="succeeded",
                observations=[_release_observation(tool_ref, gripper_empty=released == 2)],
                physical_effect="confirmed",
            ),
        )
        if released == 1:
            context.queue_action(
                VERIFY_TOOL_LOAD_ACTION,
                ActionResult(
                    status="succeeded",
                    observations=[_tool_load((held.tool_refs[1],), stable=not support_transferred)],
                    physical_effect="none",
                ),
            )
    for _ in range(4):
        context.queue_action(MOVE_ACTION, ActionResult(status="succeeded", physical_effect="confirmed"))
    for _ in range(stability_attempts):
        context.queue_action(
            VERIFY_STABILITY_ACTION,
            ActionResult(
                status="succeeded", observations=[_stability_observation(stable=stability)],
                physical_effect="none",
            ),
        )
    if stability:
        context.queue_action(MOVE_TO_POSTURE_ACTION, ActionResult(status="succeeded", physical_effect="confirmed"))


@pytest.mark.parametrize("capture_status", [None, "succeeded", "failed"])
def test_place_object_completes_only_after_independent_verification(capture_status) -> None:
    """release 成功后仍必须撤离并通过独立验证才生成共享输出。"""

    slot = _slot_observation(revision="scene-42")
    context, held = _new_context(slot=slot)
    frame_refs = [f"pilot-artifact://pilot-place/frame-{index}" for index in range(7)]
    if capture_status:
        context.execution_id = "execution-place-rgb"
        for ref in frame_refs:
            context.queue_action("sensor.capture_rgbd", ActionResult(
                status=capture_status, physical_effect="none",
                observations=[Observation(
                    kind="sensor.frame", source="robot-sdk://demo-1/sensor/camera.rgb",
                    value={"sensor_id": "camera.rgb", "media_type": "image/png"},
                    evidence_refs=[ref],
                )] if capture_status == "succeeded" else [],
                error_code="SENSOR_UNAVAILABLE" if capture_status == "failed" else None,
            ))
    _queue_successful_tail(context, held)

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    assert isinstance(context.result, PlacedObjectState)
    assert context.result.target_ref == _target().target_ref
    assert context.result.stable is True
    if capture_status:
        captures = [event for event in context.events if event.get("action") == "sensor.capture_rgbd"]
        assert len(captures) == len({event["key"] for event in captures}) == 7
        assert not any('"retreat"' in event["key"] for event in captures)
        result_refs = set(context.result.evidence_refs)
        if capture_status == "succeeded":
            assert set(frame_refs) <= result_refs
        else:
            assert not set(frame_refs) & result_refs
    started = [
        event["action"]
        for event in context.events
        if event["type"] == "action_started" and event["action"] != "sensor.capture_rgbd"
    ]
    assert started == [
        GET_ROBOT_STATE_ACTION,
        VERIFY_TOOL_LOAD_ACTION,
        LOCATE_OBJECT_ACTION,
        MOVE_ACTION,
        VERIFY_TOOL_LOAD_ACTION,
        MOVE_ACTION,
        VERIFY_TOOL_LOAD_ACTION,
        RELEASE_ACTION,
        VERIFY_TOOL_LOAD_ACTION,
        RELEASE_ACTION,
        MOVE_ACTION,
        MOVE_ACTION,
        MOVE_ACTION,
        MOVE_ACTION,
        VERIFY_STABILITY_ACTION,
        MOVE_TO_POSTURE_ACTION,
    ]
    retreat_purposes = [
        event["parameters"]["purpose"]
        for event in context.events
        if event.get("type") == "action_started"
        and event.get("key", "").startswith("place.retreat.")
    ]
    assert retreat_purposes == ["unseat", "disengage", "retreat", "retreat"]
    assert context.state.stage == "completed"
    assert context.state.release_cursor == 2
    assert context.state.released_tool_refs == list(held.tool_refs)
    completed_stages = [
        event["stage"]
        for event in context.events
        if event.get("type") == "stage.completed"
    ]
    assert completed_stages == [
        "verify_held_object",
        "observe_target_slot",
        "plan_approach",
        "approach",
        "release",
        "retreat",
        "verify_stability",
        "restore_travel_posture",
    ]


def test_supported_object_does_not_require_remaining_tool_to_carry_after_release() -> None:
    """载荷已实时转移到托盘后，逐侧释放不能要求另一侧继续悬空承载。"""

    context, held = _new_context(slot=_slot_observation(revision="scene-supported"))
    _queue_successful_tail(context, held, support_transferred=True)

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    assert context.state.support_transfer_confirmed is True


def test_dense_side_clearance_runs_supported_slide_flow() -> None:
    """托盘承重、内侧撤离、外侧推入和最终撤离按真实动作顺序执行。"""

    base_slot = _slot_observation(revision="scene-supported-slide-flow")
    slot_state = PlacementSlotState.model_validate(base_slot.value).model_copy(update={
        "lateral_clearance_m": {"negative": 0.02, "positive": None},
    })
    slot = base_slot.model_copy(update={"value": slot_state.model_dump(mode="json")})
    context, held = _new_context(slot=slot)
    held = held.model_copy(update={
        "object_size_m": (0.6, 0.4, 0.34),
        "tool_poses": {
            "component://tool/left": held.tool_poses[held.tool_refs[0]].model_copy(
                update={"position_m": (0.907, 0.8, 0.65)}
            ),
            "component://tool/right": held.tool_poses[held.tool_refs[1]].model_copy(
                update={"position_m": (1.493, 0.8, 0.65)}
            ),
        },
    })
    _queue_verify_held(context, held)
    context.queue_action(MOVE_ACTION, ActionResult(status="succeeded", physical_effect="confirmed"))
    context.queue_action(
        VERIFY_TOOL_LOAD_ACTION,
        ActionResult(status="succeeded", observations=[_tool_load(held.tool_refs)]),
    )
    context.queue_action(
        MOVE_ACTION,
        ActionResult(
            status="succeeded",
            physical_effect="confirmed",
        ),
    )
    # 临时落座会故意保留侧向偏移；只要箱体仍在目标区、有托盘接触且没有
    # 明显倾倒，就可以进入后续一次推正，不叠加固定中心距离。
    staged_supported = _stability_observation(
        stable=False,
        position_error_m=0.045,
    )
    staged_supported_value = dict(staged_supported.value or {})
    staged_supported_state = dict(staged_supported_value["state"])
    staged_supported_state.update({"stable": False, "gripper_empty": False})
    staged_supported_value.update({
        "state": staged_supported_state,
        "stable": False,
        "gripper_empty": False,
    })
    context.queue_action(
        VERIFY_STABILITY_ACTION,
        ActionResult(
            status="succeeded",
            observations=[staged_supported.model_copy(
                update={"value": staged_supported_value}
            )],
            physical_effect="none",
        ),
    )
    context.queue_action(
        RELEASE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_release_observation(held.tool_refs[0], gripper_empty=False)],
            physical_effect="confirmed",
        ),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            # 与第四箱失败回放一致：夹具打开后，内侧下钩已经没有接触，
            # 另一侧仍保持箱体。Skill 应跳过会重新勾住箱沿的 unseat。
            observations=[_robot_state(held, tool_contacts={
                held.tool_refs[0]: False,
                held.tool_refs[1]: True,
            })],
            physical_effect="none",
        ),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            # 两侧均已卸载，但抬高手臂时仍要用同一次实时状态保持另一
            # 末端不动，避免共享躯干把空钩扫回已经落座的箱体。
            observations=[_robot_state(held, tool_contact=False)],
            physical_effect="none",
        ),
    )
    # 已确认下钩无接触时，推正前只需水平退钩并从临时偏置位抬高。
    for _ in range(2):
        context.queue_action(
            MOVE_ACTION, ActionResult(status="succeeded", physical_effect="confirmed")
        )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        LOCATE_OBJECT_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_object_pose(held)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_ACTION,
        ActionResult(
            status="failed",
            error_code="ROBOT_COMMAND_FAILED",
            error_message="轨迹已完成，支撑接触使关节终点未收敛",
            physical_effect="possible",
        ),
    )
    # 第一次推进只完成一部分，箱体虽然仍在目标区并由托盘承重，但距列
    # 中心仍有4.5cm，超过粗粒度完成边界，必须补推后再释放剩余工具。
    supported_after_failed_push = _stability_observation(
        stable=False,
        position_error_m=0.045,
    )
    supported_after_failed_push_value = dict(
        supported_after_failed_push.value or {}
    )
    supported_after_failed_push_state = dict(
        supported_after_failed_push_value["state"]
    )
    supported_after_failed_push_state.update({
        "stable": False,
        "gripper_empty": False,
    })
    supported_after_failed_push_value.update({
        "state": supported_after_failed_push_state,
        "stable": False,
        "gripper_empty": False,
    })
    context.queue_action(
        VERIFY_STABILITY_ACTION,
        ActionResult(
            status="succeeded",
            observations=[supported_after_failed_push.model_copy(
                update={"value": supported_after_failed_push_value}
            )],
            physical_effect="none",
        ),
    )
    context.queue_action(
        LOCATE_OBJECT_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_object_pose(held)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    # 唯一一次补推把中心误差收敛到2cm以内，随后才允许释放和撤离。
    supported_after_retry = _stability_observation(
        stable=False,
        position_error_m=0.019,
    )
    supported_after_retry_value = dict(supported_after_retry.value or {})
    supported_after_retry_state = dict(supported_after_retry_value["state"])
    supported_after_retry_state.update({"stable": False, "gripper_empty": False})
    supported_after_retry_value.update({
        "state": supported_after_retry_state,
        "stable": False,
        "gripper_empty": False,
    })
    context.queue_action(
        VERIFY_STABILITY_ACTION,
        ActionResult(
            status="succeeded",
            observations=[supported_after_retry.model_copy(
                update={"value": supported_after_retry_value}
            )],
            physical_effect="none",
        ),
    )
    context.queue_action(
        RELEASE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_release_observation(held.tool_refs[1], gripper_empty=True)],
            physical_effect="confirmed",
        ),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held, tool_contact=False)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held, tool_contact=False)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held, tool_contact=False)],
            physical_effect="none",
        ),
    )
    # 补推完成后只执行四段最终退钩。
    for _ in range(4):
        context.queue_action(
            MOVE_ACTION, ActionResult(status="succeeded", physical_effect="confirmed")
        )
    supported_with_contact = _stability_observation(stable=False)
    supported_value = dict(supported_with_contact.value or {})
    supported_state = dict(supported_value["state"])
    supported_state.update({"stable": False, "gripper_empty": False})
    supported_value.update({
        "state": supported_state,
        "stable": False,
        "gripper_empty": False,
    })
    supported_with_contact = supported_with_contact.model_copy(
        update={"value": supported_value}
    )
    context.queue_action(
        VERIFY_STABILITY_ACTION,
        ActionResult(
            status="succeeded", observations=[supported_with_contact],
            physical_effect="none",
        ),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held, tool_contact=False)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held, tool_contact=False)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        VERIFY_STABILITY_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_stability_observation()],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_TO_POSTURE_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    assert "release" in [
        event["stage"] for event in context.events
        if event.get("type") == "stage.completed"
    ]
    started_keys = [
        event.get("key", "")
        for event in context.events
        if event.get("type") == "action_started"
    ]
    assert any("early.staged-support" in key for key in started_keys)
    assert not any("early.unseat" in key for key in started_keys)
    assert any("early.disengage" in key for key in started_keys)
    assert any("early.travel_clearance" in key for key in started_keys)
    assert not any("supported-slide.verify" in key for key in started_keys)
    approach_actions = [
        event for event in context.events
        if event.get("type") == "action_started"
        and ".early." in event.get("key", "")
        and event.get("action") == MOVE_ACTION
    ]
    withdraw_actions = [
        action for action in approach_actions
        if action["key"].endswith((
            "early.unseat", "early.disengage",
        ))
    ]
    assert len(withdraw_actions) == 1
    early_tool = context.state.approach_plan.early_release_tool_ref
    assert early_tool is not None
    remaining_tool = next(ref for ref in held.tool_refs if ref != early_tool)
    assert all(
        action["parameters"].get("required_contact_tools", []) == [remaining_tool]
        for action in withdraw_actions
    )
    assert all(
        action["parameters"].get("expected_object_pose") is not None
        for action in withdraw_actions
    )
    assert all(
        action["parameters"].get("fixed_tool_refs", []) == []
        for action in withdraw_actions
    )
    assert all(
        len(action["parameters"]["targets"]) == 1
        for action in withdraw_actions
    )
    assert all(
        {target["tool_ref"] for target in action["parameters"]["targets"]}
        == {early_tool}
        for action in withdraw_actions
    )
    early_clearance = next(
        action for action in approach_actions
        if action["key"].endswith("early.travel_clearance")
    )
    assert len(early_clearance["parameters"]["targets"]) == 1
    assert {
        target["tool_ref"]
        for target in early_clearance["parameters"]["targets"]
    } == {early_tool}
    clearance_by_tool = {
        target["tool_ref"]: target["target_pose"]
        for target in early_clearance["parameters"]["targets"]
    }
    assert clearance_by_tool[early_tool]["position_m"][2] == pytest.approx(
            held.tool_poses[early_tool].position_m[2] + 0.178
    )
    assert early_clearance["parameters"]["purpose"] == "clearance"
    assert early_clearance["parameters"]["required_contact_tools"] == [remaining_tool]
    assert early_clearance["parameters"]["expected_object_pose"] is not None
    assert any(
        event.get("key") == "place.robot-state.early-clearance-anchor"
        for event in context.events
        if event.get("type") == "action_started"
    )
    supported_push = next(
        event for event in context.events
        if event.get("type") == "action_started"
        and event.get("key", "").endswith(".0.release")
        and event.get("action") == MOVE_ACTION
    )
    assert supported_push["parameters"]["purpose"] == "disengage"
    assert supported_push["parameters"]["required_contact_tools"] == []
    assert supported_push["parameters"]["expected_object_pose"] is not None
    assert len(supported_push["parameters"]["targets"]) == 1
    assert supported_push["parameters"]["targets"][0]["tool_ref"] != (
        context.state.approach_plan.early_release_tool_ref
    )
    supported_release_key = next(
        key for key in started_keys
        if ".supported." in key and key.startswith("place.release.")
    )
    assert started_keys.index(supported_release_key) < started_keys.index(
        supported_push["key"]
    )
    assert started_keys.index(early_clearance["key"]) < started_keys.index(
        supported_release_key
    )
    assert started_keys.index(early_clearance["key"]) < started_keys.index(
        supported_push["key"]
    )
    assert started_keys.index(supported_push["key"]) > started_keys.index(
        early_clearance["key"]
    )
    assert any(
        event.get("key") == "place.verify-support-transfer.0"
        for event in context.events
        if event.get("type") == "action_started"
    )
    assert any(
        event.get("key", "").endswith(".1.release")
        for event in context.events
        if event.get("type") == "action_started"
    )
    stability_keys = [
        event["key"]
        for event in context.events
        if event.get("type") == "action_started"
        and event.get("action") == VERIFY_STABILITY_ACTION
        and event.get("key", "").startswith("place.verify-stability.")
    ]
    assert stability_keys == [
        "place.verify-stability.0.0",
        "place.verify-stability.1.0",
    ]
    retreat_purposes = [
        event["parameters"]["purpose"]
        for event in context.events
        if event.get("type") == "action_started"
        and event.get("key", "").startswith("place.retreat.")
    ]
    assert retreat_purposes == [
        "unseat", "disengage", "disengage", "disengage",
    ]
    retreat_actions = [
        event for event in context.events
        if event.get("type") == "action_started"
        and event.get("key", "").startswith("place.retreat.")
    ]
    assert [len(action["parameters"]["targets"]) for action in retreat_actions] == [
        1, 1, 1, 1,
    ]
    assert any(
        event.get("key") == "place.robot-state.final-retreat-anchor"
        for event in context.events
        if event.get("type") == "action_started"
    )
    final_unseat = retreat_actions[0]["parameters"]["targets"][0]
    live_remaining_pose = held.tool_poses[held.tool_refs[1]]
    assert final_unseat["target_pose"]["position_m"][:2] == pytest.approx(
        live_remaining_pose.position_m[:2]
    )
    assert final_unseat["target_pose"]["position_m"][2] == pytest.approx(
        live_remaining_pose.position_m[2] - 0.008
    )
    assert {
        target["tool_ref"]
        for target in retreat_actions[-1]["parameters"]["targets"]
    } == {held.tool_refs[1]}
    final_low_clearance = retreat_actions[-2]["parameters"]["targets"][0]
    final_high_clearance = next(
        target for target in retreat_actions[-1]["parameters"]["targets"]
        if target["tool_ref"] == held.tool_refs[1]
    )
    assert final_low_clearance["tool_ref"] == held.tool_refs[1]
    assert final_low_clearance["target_pose"]["position_m"][2] == pytest.approx(
        live_remaining_pose.position_m[2]
    )
    assert final_low_clearance["target_pose"]["position_m"][:2] == pytest.approx(
        final_high_clearance["target_pose"]["position_m"][:2]
    )
    assert final_high_clearance["target_pose"]["position_m"][2] > (
        final_low_clearance["target_pose"]["position_m"][2]
    )
    assert not any(
        event.get("key") == "place.robot-state.final-clearance-anchor-0"
        for event in context.events
        if event.get("type") == "action_started"
    )
    postplace_clearance = next(
        event for event in context.events
        if event.get("type") == "action_started"
        and event.get("key") == "place.postplace-clearance.0"
    )
    assert postplace_clearance["parameters"]["purpose"] == "disengage"
    assert {
        target["tool_ref"]
        for target in postplace_clearance["parameters"]["targets"]
    } == set(held.tool_refs)
    postplace_by_tool = {
        target["tool_ref"]: target["target_pose"]["position_m"]
        for target in postplace_clearance["parameters"]["targets"]
    }
    # 最终箱体中心相对原计划发生平移后，必须按真实边界重新把两只钩横向
    # 清出；只朝基座方向移动不能解除侧面挂钩。
    assert postplace_by_tool[held.tool_refs[0]][0] <= 0.875
    assert postplace_by_tool[held.tool_refs[1]][0] >= 1.525
    assert any(
        event.get("key") == "place.robot-state.postplace-clearance-verify.0"
        for event in context.events
        if event.get("type") == "action_started"
    )


def test_postplace_contact_retry_uses_fresh_action_generation() -> None:
    """清钩后仍有接触时，Agent 重试清钩而不是重放已结束的旧退钩。"""

    base_slot = _slot_observation(revision="scene-postplace-retry")
    slot = PlacementSlotState.model_validate(base_slot.value).model_copy(update={
        "lateral_clearance_m": {"negative": 0.02, "positive": None},
    })
    context, held = _new_context(
        slot=base_slot.model_copy(update={"value": slot.model_dump(mode="json")})
    )
    held = held.model_copy(update={
        "object_size_m": (0.6, 0.4, 0.34),
        "tool_poses": {
            held.tool_refs[0]: held.tool_poses[held.tool_refs[0]].model_copy(
                update={"position_m": (0.907, 0.8, 0.65)}
            ),
            held.tool_refs[1]: held.tool_poses[held.tool_refs[1]].model_copy(
                update={"position_m": (1.493, 0.8, 0.65)}
            ),
        },
    })
    controller = PlacementController()
    plan = controller.build_approach_plan(
        held, slot, PlacementConstraints()
    )
    placed = PlacedObjectState.model_validate(
        (_stability_observation().value or {})["state"]
    ).model_copy(update={"gripper_empty": False, "stable": False})
    context.checkpoint(PlaceObjectRunState(
        stage="restore_travel_posture",
        verified_held_object=held,
        target_slot=slot,
        approach_plan=plan,
        release_cursor=2,
        released_tool_refs=list(held.tool_refs),
        release_confirmed=True,
        support_transfer_confirmed=True,
        retreat_cursor=4,
        retreat_confirmed=True,
        placed_object=placed,
    ))

    # 第一次清钩动作到位但左钩接触仍在；DeepSeek 选择 retry_retreat。
    for contact in (True, False):
        context.queue_action(
            GET_ROBOT_STATE_ACTION,
            ActionResult(
                status="succeeded",
                observations=[_robot_state(held, tool_contact=contact)],
                physical_effect="none",
            ),
        )
        context.queue_action(
            MOVE_ACTION,
            ActionResult(status="succeeded", physical_effect="confirmed"),
        )
        context.queue_action(
            GET_ROBOT_STATE_ACTION,
            ActionResult(
                status="succeeded",
                observations=[_robot_state(held, tool_contact=contact)],
                physical_effect="none",
            ),
        )
    context.queue_agent_reply({
        "decision": "retry_retreat",
        "reason": "重新清出仍接触箱体的左侧空钩",
    })
    context.queue_action(
        VERIFY_STABILITY_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_stability_observation()],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_TO_POSTURE_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    started_keys = [
        event["key"] for event in context.events
        if event.get("type") == "action_started"
    ]
    assert "place.postplace-clearance.0" in started_keys
    assert "place.postplace-clearance.1" in started_keys
    assert "place.verify-stability.1.0" in started_keys
    assert "place.restore-travel-posture.1.1" in started_keys
    assert not any(key.startswith("place.retreat.") for key in started_keys)


def test_final_empty_tool_evidence_skips_redundant_postplace_clearance() -> None:
    """最终观测已确认工具为空时，不再生成一次多余的清钩动作。"""

    base_slot = _slot_observation(revision="scene-final-clear")
    slot = PlacementSlotState.model_validate(base_slot.value).model_copy(update={
        "lateral_clearance_m": {"negative": 0.02, "positive": None},
    })
    context, held = _new_context(
        slot=base_slot.model_copy(update={"value": slot.model_dump(mode="json")})
    )
    held = held.model_copy(update={"object_size_m": (0.6, 0.4, 0.34)})
    plan = PlacementController().build_approach_plan(
        held, slot, PlacementConstraints()
    )
    placed = PlacedObjectState.model_validate(
        (_stability_observation().value or {})["state"]
    )
    context.checkpoint(PlaceObjectRunState(
        stage="restore_travel_posture",
        verified_held_object=held,
        target_slot=slot,
        approach_plan=plan,
        release_cursor=2,
        released_tool_refs=list(held.tool_refs),
        release_confirmed=True,
        support_transfer_confirmed=True,
        retreat_cursor=4,
        retreat_confirmed=True,
        placed_object=placed,
    ))
    context.queue_action(
        MOVE_TO_POSTURE_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    started_keys = [
        event["key"] for event in context.events
        if event.get("type") == "action_started"
    ]
    assert "place.restore-travel-posture.0.0" in started_keys
    assert not any("postplace-clearance" in key for key in started_keys)


def test_travel_collision_clears_empty_tools_then_retries_posture() -> None:
    """放置已完成时，travel折叠碰邻物只清出空钩，不重放放置动作。"""

    slot_observation = _slot_observation(revision="scene-travel-collision")
    slot = PlacementSlotState.model_validate(slot_observation.value)
    context, held = _new_context(slot=slot_observation)
    plan = PlacementController().build_approach_plan(
        held, slot, PlacementConstraints()
    )
    placed = PlacedObjectState.model_validate(
        (_stability_observation().value or {})["state"]
    )
    context.checkpoint(PlaceObjectRunState(
        stage="restore_travel_posture",
        verified_held_object=held,
        target_slot=slot,
        approach_plan=plan,
        release_cursor=2,
        released_tool_refs=list(held.tool_refs),
        release_confirmed=True,
        support_transfer_confirmed=True,
        retreat_cursor=4,
        retreat_confirmed=True,
        placed_object=placed,
    ))
    context.queue_action(
        MOVE_TO_POSTURE_ACTION,
        ActionResult(
            status="failed",
            error_code="PLANNING_FAILED",
            error_message="travel折叠路径碰到相邻周转箱",
            physical_effect="none",
        ),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held, tool_contact=False)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        VERIFY_STABILITY_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_stability_observation()],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_TO_POSTURE_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    started_keys = [
        event["key"] for event in context.events
        if event.get("type") == "action_started"
    ]
    assert "place.restore-travel-posture.0.0" in started_keys
    assert "place.restore-travel-posture.1.0" in started_keys
    assert "place.postplace-clearance.0" in started_keys
    assert "place.verify-stability.1.0" in started_keys


def test_travel_failure_after_clearance_is_not_reported_as_success() -> None:
    """清钩后仍无法恢复travel时应准确失败，不能跳过完成标准。"""

    slot_observation = _slot_observation(revision="scene-travel-unreachable")
    slot = PlacementSlotState.model_validate(slot_observation.value)
    context, held = _new_context(slot=slot_observation)
    plan = PlacementController().build_approach_plan(
        held, slot, PlacementConstraints()
    )
    placed = PlacedObjectState.model_validate(
        (_stability_observation().value or {})["state"]
    )
    context.checkpoint(PlaceObjectRunState(
        stage="restore_travel_posture",
        verified_held_object=held,
        target_slot=slot,
        approach_plan=plan,
        release_cursor=2,
        released_tool_refs=list(held.tool_refs),
        release_confirmed=True,
        support_transfer_confirmed=True,
        retreat_cursor=4,
        retreat_confirmed=True,
        placed_object=placed,
    ))
    context.queue_action(
        MOVE_TO_POSTURE_ACTION,
        ActionResult(
            status="failed",
            error_code="PLANNING_FAILED",
            error_message="初次travel折叠碰到相邻周转箱",
            physical_effect="none",
        ),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    context.queue_action(
        GET_ROBOT_STATE_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_robot_state(held, tool_contact=False)],
            physical_effect="none",
        ),
    )
    context.queue_action(
        VERIFY_STABILITY_ACTION,
        ActionResult(
            status="succeeded",
            observations=[_stability_observation()],
            physical_effect="none",
        ),
    )
    context.queue_action(
        MOVE_TO_POSTURE_ACTION,
        ActionResult(
            status="failed",
            error_code="PLANNING_FAILED",
            error_message="清钩后travel仍不可达",
            physical_effect="none",
        ),
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "failed"
    assert context.failure is not None
    assert context.failure.code == "TRAVEL_POSTURE_UNREACHABLE"
    assert "清钩后travel仍不可达" in context.failure.message
    assert context.result is None
    assert not any(event["type"] == "agent_requested" for event in context.events)


def test_dense_support_keeps_bilateral_load_until_real_transfer() -> None:
    """已有邻箱时仍由双侧下降，支撑转移后才允许逐侧释放。"""

    base_slot = _slot_observation(revision="scene-dense-support")
    slot_state = PlacementSlotState.model_validate(base_slot.value).model_copy(update={
        "support_center_pose": _pose("pallet-b").model_copy(
            update={"position_m": (1.2, 1.5, 0.075)}
        ),
        "support_occupied": True,
    })
    slot = base_slot.model_copy(update={"value": slot_state.model_dump(mode="json")})
    context, held = _new_context(slot=slot)
    _queue_successful_tail(context, held, support_transferred=True)

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    assert context.state.support_transfer_confirmed is True
    started_keys = [
        event.get("key", "")
        for event in context.events
        if event.get("type") == "action_started"
    ]
    assert not any(".early." in key for key in started_keys)


def test_approach_failure_refreshes_slot_and_replans_locally() -> None:
    """可恢复接近失败由 Controller 刷新观测，不立即占用 Agent。"""

    initial_slot = _slot_observation(revision="scene-42")
    context, held = _new_context(slot=initial_slot)
    _queue_verify_held(context, held)
    context.queue_action(
        MOVE_ACTION,
        ActionResult(
            status="failed",
            error_code="APPROACH_BLOCKED",
            error_message="临时障碍物阻塞接近路径",
            physical_effect="possible",
        ),
    )
    refreshed_slot = _slot_observation(revision="scene-42")
    context.queue_action(
        OBSERVE_SLOT_ACTION,
        ActionResult(
            status="succeeded",
            observations=[refreshed_slot],
            physical_effect="none",
        ),
    )
    _queue_successful_tail(context, held, include_verify=False)

    asyncio.run(run_skill(run, context))

    assert context.status == "completed"
    started = [
        event["action"]
        for event in context.events
        if event["type"] == "action_started"
    ]
    assert sum(1 for event in context.events if event.get("type") == "action_started" and event.get("key", "").startswith("place.approach")) == 3
    assert OBSERVE_SLOT_ACTION in started
    assert not any(event["type"] == "agent_requested" for event in context.events)


def test_failed_stability_verification_escalates_to_agent() -> None:
    """稳定性复核预算耗尽后由 Agent 决定终止，不伪造放置成功。"""

    slot = _slot_observation(revision="scene-42")
    context, held = _new_context(slot=slot)
    _queue_successful_tail(
        context,
        held,
        stability=False,
        stability_attempts=2,
    )
    context.queue_agent_reply(
        {
            "decision": "abort",
            "reason": "物体不稳定，停止当前 Action 并重新规划抓取与放置",
        }
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "failed"
    assert context.failure is not None
    assert context.failure.code == "AGENT_ABORTED_PLACEMENT"
    assert any(event["type"] == "agent_requested" for event in context.events)
    assert context.result is None


def test_unknown_release_state_stops_without_agent_continuation() -> None:
    """释放副作用不确定时必须结束本次执行，不能让 Agent 原地继续运动。"""

    slot = _slot_observation(revision="scene-42")
    context, held = _new_context(slot=slot)
    _queue_verify_held(context, held)
    for _ in range(2):
        context.queue_action(MOVE_ACTION, ActionResult(status="succeeded", physical_effect="confirmed"))
        context.queue_action(
            VERIFY_TOOL_LOAD_ACTION,
            ActionResult(status="succeeded", observations=[_tool_load(held.tool_refs)], physical_effect="none"),
        )
    context.queue_action(
        RELEASE_ACTION,
        ActionResult(
            status="interrupted",
            error_code="RELEASE_TIMEOUT",
            error_message="无法确认夹爪是否已经释放",
            physical_effect="unknown",
        ),
    )

    asyncio.run(run_skill(run, context))

    assert context.status == "failed"
    assert context.failure is not None
    assert context.failure.code == "RELEASE_STATE_UNKNOWN"
    assert not any(
        event["type"] == "agent_requested" for event in context.events
    )
    started_events = [event for event in context.events if event["type"] == "action_started"]
    assert not any(event.get("key", "").startswith("place.retreat") for event in started_events)
    assert not any(event["action"] == VERIFY_STABILITY_ACTION for event in started_events)


def test_on_stop_distinguishes_held_and_released_states() -> None:
    """停止输出必须区分仍持物和已释放但未验证两种物理状态。"""

    slot = _slot_observation(revision="scene-42")
    held_context, held = _new_context(slot=slot)
    held_context.checkpoint(
        PlaceObjectRunState(stage="approach", verified_held_object=held)
    )
    held_context.request_stop()
    held_context.queue_action(
        SAFE_STOP_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    held_outcome = asyncio.run(
        on_stop(
            held_context,
            StopRequest(source="user", reason="用户停止放置"),
        )
    )
    assert held_outcome.safe is True
    assert held_outcome.physical_state == "hold"
    assert held_outcome.requires_intervention is False

    released_context, released_held = _new_context(slot=slot)
    released_context.checkpoint(
        PlaceObjectRunState(
            stage="retreat",
            verified_held_object=released_held,
            release_cursor=2,
            release_confirmed=True,
        )
    )
    released_context.request_stop()
    released_context.queue_action(
        SAFE_STOP_ACTION,
        ActionResult(status="succeeded", physical_effect="confirmed"),
    )
    released_outcome = asyncio.run(
        on_stop(
            released_context,
            StopRequest(source="agent", reason="目标区域出现异常"),
        )
    )
    assert released_outcome.safe is True
    assert released_outcome.physical_state == "released"
    assert released_outcome.requires_intervention is True
