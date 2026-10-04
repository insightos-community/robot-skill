"""place-object 的确定性双工具 Controller。"""

from __future__ import annotations

from math import sqrt

from semantic_robot_skill_sdk import Observation, Pose3D, SkillFailure

from .models import (
    HeldObjectState, LocalRecoveryDecision, PlacementApproachPlan,
    PlacementConstraints, PlacementSlotState, PlacementVerificationObservation,
    PlacementTarget, PlacementWaypoint, PlacedObjectState, RobotStateValue,
    TargetPoseValue, ToolLoadObservationValue, ToolPoseTarget,
)

TARGET_POSE_OBSERVATION = "target_pose"
TOOL_LOAD_OBSERVATION = "robot.tool_load"
ROBOT_STATE_OBSERVATION = "robot.state"
TARGET_SLOT_OBSERVATION = "placement.target_slot"
RELEASED_OBJECT_OBSERVATION = "manipulation.object_released"
PLACED_OBJECT_OBSERVATION = "placement.object_stability"
SUPPORT_TRANSFER_OBSERVATION = "placement.carrying_load"
# 末端目标控制的是load frame，而下钩最前端比load frame更靠近箱体17mm。
# layout001实测证明19mm退出会在随后竖直撤离时重新挂住凹槽并拖动箱体。
# 因此仍保留25mm安全距离；紧邻列不能靠压缩全局距离解决，而是在箱体
# 落座前先撤走靠托盘中心的一侧工具。该值只负责单个钩脚真正退出凹槽，
# 不改变碰撞、接触、滑移或过载判断。
GROOVE_DISENGAGE_LOAD_FRAME_CLEARANCE_M = 0.025
# 下钩已经退出凹槽并不代表整套夹具可以立刻向上撤离。layout001相邻列
# 只有厘米级净空，竖直上抬时需要让钩脚、导向片和打开后的压片整体越过
# 邻箱侧壁。该包络来自当前兼容夹具Profile，与抓取规划使用的52mm一致；
# 它只参与支撑面上的临时偏移计算，实际撤离仍逐段执行完整碰撞检查。
TOTE_CLAMP_LATERAL_ENVELOPE_M = 0.052
# place-object面向当前R1 Pro周转箱夹具；8mm与抓取候选的seat行程相同。
# 释放时必须先反向解除seat，不能让仍承托在凹槽内的下钩直接水平拖动箱体。
# 这属于Robot Skill工艺，不进入通用SDK或仿真Runtime。
GROOVE_UNSEAT_TRAVEL_M = 0.008
# 支撑面承重后，内侧钩退出所需净空可能大于相邻列间隙。箱体推正时会
# 相对已撤出的空手移动数厘米；layout001 的真实失败现场还证明，4cm余量
# 只能让load frame刚好越过箱体，却没有给travel折叠时的钩脚圆弧留出通道。
# 因此在夹具52mm包络外保留8cm水平净空。该值只影响空夹具撤离目标，
# 沿途仍使用原有碰撞检查，不改变放置精度、IK或全局安全阈值。
SUPPORTED_SLIDE_CLEARANCE_MARGIN_M = 0.08
# 先释放的空钩必须明显高于箱沿，才能避免另一只手推正箱体时再次勾住。
# 末端跟踪会有厘米级偏差，因此目标在箱沿上方保留5cm；这不是完成精度，
# 只是一个直观的撤手高度。可达性和沿途碰撞仍交给通用运动规划器判断。
EARLY_TOOL_TOP_CLEARANCE_M = 0.05
# 相邻列逐侧释放必须先让托盘真实承重。目标 Pose 已对应箱底与支撑面
# 接触，只需 5mm 的低速寻底余量覆盖模型和伺服误差。更大的穿透目标会在
# 箱底已经接触后继续驱动双臂，尤其在最后一个密集列中把箱体压斜并卡住
# 夹具。真实支撑接触仍由 Ability 观测确认，不能仅凭命令终点释放工具。
SUPPORT_SEEK_TRAVEL_M = 0.005
# 箱体已由托盘承重不等于姿态已可释放。相对目标倾斜超过约14度时，
# 周转箱可能只靠邻箱或工具侧面临时支撑；此时释放最后一只工具会让
# 箱体继续滑动。该阈值是支撑面上的粗粒度工艺边界，不是毫米级定位精度。
MAX_SUPPORTED_ORIENTATION_ERROR_RAD = 0.25
# 当前拆码垛不要求箱体毫米级居中。支撑面短推后允许4cm的粗粒度
# 中心误差；超过该值才补推，避免“刚落入Region就撤手”。现场连续验证
# 曾出现箱体已稳定、在目标区且工具已空，却因25mm边界多出0.033mm而
# 误报失败。这里保留厘米级列对齐要求，稳定、支撑和姿态仍由通用观测判定。
SUPPORTED_SLIDE_COMPLETION_TOLERANCE_M = 0.04


class PlacementController:
    _RECOVERABLE_APPROACH_ERRORS = {"APPROACH_BLOCKED", "POSE_UNREACHABLE", "TARGET_POSE_STALE"}
    _RECOVERABLE_RETREAT_ERRORS = {"RETREAT_BLOCKED", "TRANSIENT_MOTION_ERROR"}

    def verify_held_object(
        self,
        object_ref: str,
        robot_state_observation: Observation,
        tool_load_observation: Observation,
        object_observation: Observation,
        *,
        robot_ref: str,
    ) -> HeldObjectState:
        """由三个实时通用Observation建立本次place Execution的局部持物状态。"""

        robot_state = self.parse_robot_state(robot_state_observation)
        if robot_state.robot_id != robot_ref:
            raise SkillFailure("ROBOT_MISMATCH", "Robot状态不属于当前Execution绑定的Robot")
        tool_refs = self.discover_tool_refs(robot_state)
        try:
            tool_load = ToolLoadObservationValue.model_validate(
                tool_load_observation.value or {}
            )
            target = TargetPoseValue.model_validate(object_observation.value or {})
        except ValueError as error:
            raise SkillFailure("HELD_STATE_INVALID", f"实时持物观测无法解析：{error}") from error
        if tool_load_observation.kind != TOOL_LOAD_OBSERVATION:
            raise SkillFailure("TOOL_LOAD_KIND_INVALID", "工具承载观测类型不正确")
        if object_observation.kind != TARGET_POSE_OBSERVATION:
            raise SkillFailure("OBJECT_POSE_KIND_INVALID", "物体观测类型不正确")
        if target.object_ref != object_ref:
            raise SkillFailure("OBJECT_MISMATCH", "实时物体观测与请求放置的对象不一致")
        if {item.tool_ref for item in tool_load.tools} != set(tool_refs):
            raise SkillFailure("TOOLS_MISMATCH", "工具承载观测不属于当前双侧工具")
        if (
            not tool_load.condition_satisfied
            or tool_load.slip_detected
            or tool_load.overload_detected
            or tool_load.sensor_fault
        ):
            raise SkillFailure("OBJECT_NOT_HELD", "双侧工具当前没有形成稳定承载")

        tool_poses: dict[str, Pose3D] = {}
        for ref in tool_refs:
            side = robot_state.tool_states[ref].side
            end_effector = robot_state.end_effectors.get(side)
            if end_effector is None:
                raise SkillFailure("TOOL_POSE_MISSING", f"缺少{ref}的实时末端位姿")
            tool_poses[ref] = target.pose.model_copy(update={
                "frame_id": end_effector.frame_id,
                "position_m": end_effector.position,
                "orientation_xyzw": end_effector.quaternion_xyzw,
            })
        return HeldObjectState(
            object_ref=object_ref,
            robot_ref=robot_ref,
            tool_refs=tool_refs,
            tool_poses=tool_poses,
            object_pose=target.pose,
            object_size_m=target.extent_m,
            base_position_m=(
                robot_state.base_pose.position
                if robot_state.base_pose is not None
                else None
            ),
            robot_state_generation=robot_state.generation,
            evidence_refs=list(dict.fromkeys([
                *robot_state_observation.evidence_refs,
                *tool_load_observation.evidence_refs,
                *object_observation.evidence_refs,
            ])),
        )

    @staticmethod
    def discover_tool_refs(state: RobotStateValue) -> tuple[str, str]:
        refs = tuple(
            ref for ref, tool in state.tool_states.items() if tool.kind == "tote_clamp"
        )
        if len(refs) != 2:
            raise SkillFailure("TOOLS_MISMATCH", "周转箱放置需要Robot Profile提供两个tote_clamp工具")
        return refs

    @staticmethod
    def tool_pose_from_robot_state(
        state: RobotStateValue,
        tool_ref: str,
        *,
        revision: str,
    ) -> Pose3D:
        """从当前RobotState取得工具Pose，不复用抓取完成时的旧刚体关系。"""

        tool = state.tool_states.get(tool_ref)
        if tool is None:
            raise SkillFailure("TOOL_STATE_MISSING", f"缺少{tool_ref}的实时工具状态")
        end_effector = state.end_effectors.get(tool.side)
        if end_effector is None:
            raise SkillFailure("TOOL_POSE_MISSING", f"缺少{tool_ref}的实时末端位姿")
        return Pose3D(
            frame_id=end_effector.frame_id,
            position_m=end_effector.position,
            orientation_xyzw=end_effector.quaternion_xyzw,
            # 同一个已持久化RobotState重放时必须生成完全相同的Action内容。
            # 若使用Pose3D默认的当前时间，Worker恢复后会让相同action_key
            # 携带不同observed_at，从而被Pilot的幂等保护正确拒绝。
            observed_at=state.observed_at,
            revision=revision,
        )

    @staticmethod
    def rebase_early_withdrawal(
        support: PlacementWaypoint,
        waypoints: list[PlacementWaypoint],
        *,
        tool_ref: str,
        current_pose: Pose3D,
        skip_unseat: bool = False,
    ) -> list[PlacementWaypoint]:
        """把退钩增量重放到释放后的真实末端位姿。

        箱体接触托盘后，顺应和控制误差会让实际末端与理论支撑位相差数毫米。
        若继续执行预先计算的绝对目标，8mm反向落座可能只真正移动一小段，
        下钩仍留在侧面凹槽内。这里保留原计划的三段相对位移，但从释放后的
        实时Pose重新起算；方向和距离不变，也不修改通用IK或碰撞规则。
        """

        try:
            previous_planned = next(
                target.target_pose for target in support.targets
                if target.tool_ref == tool_ref
            )
        except StopIteration as error:
            raise SkillFailure(
                "RETREAT_PLAN_INVALID", "临时支撑计划缺少待撤离工具"
            ) from error
        if previous_planned.frame_id != current_pose.frame_id:
            raise SkillFailure(
                "RETREAT_FRAME_MISMATCH", "退钩计划与实时工具位姿不在同一坐标系"
            )

        # 夹具打开后，Runtime 有时已经明确观测到下钩不再接触箱体。此时若
        # 仍机械执行第一段向下 unseat，反而会把空钩重新送回箱沿。跳过时只
        # 去掉这一个已完成的相对位移；后续水平退钩和抬高仍从实时 Pose 起算。
        if skip_unseat:
            if not waypoints or waypoints[0].name != "unseat":
                raise SkillFailure(
                    "RETREAT_PLAN_INVALID", "提前撤离计划缺少可跳过的unseat路点"
                )
            skipped = waypoints[0]
            try:
                previous_planned = next(
                    target.target_pose for target in skipped.targets
                    if target.tool_ref == tool_ref
                )
            except StopIteration as error:
                raise SkillFailure(
                    "RETREAT_PLAN_INVALID", "unseat路点缺少待撤离工具目标"
                ) from error
            waypoints = waypoints[1:]

        previous_actual = current_pose
        rebased: list[PlacementWaypoint] = []
        for waypoint in waypoints:
            try:
                planned = next(
                    target.target_pose for target in waypoint.targets
                    if target.tool_ref == tool_ref
                )
            except StopIteration as error:
                raise SkillFailure(
                    "RETREAT_PLAN_INVALID", f"{waypoint.name}缺少待撤离工具目标"
                ) from error
            delta = tuple(
                value - previous
                for value, previous in zip(
                    planned.position_m, previous_planned.position_m, strict=True
                )
            )
            actual = previous_actual.model_copy(update={
                "position_m": tuple(
                    value + change
                    for value, change in zip(
                        previous_actual.position_m, delta, strict=True
                    )
                ),
                # 退钩期间不应因理论落点的姿态残差再次翻腕。
                "orientation_xyzw": current_pose.orientation_xyzw,
                "revision": planned.revision,
            })
            rebased.append(waypoint.model_copy(update={
                "targets": [ToolPoseTarget(tool_ref=tool_ref, target_pose=actual)]
            }))
            previous_planned = planned
            previous_actual = actual
        return rebased

    def parse_target_slot(self, observation: Observation, *, expected_target_ref: str) -> PlacementSlotState:
        if observation.kind != TARGET_SLOT_OBSERVATION:
            raise SkillFailure("TARGET_SLOT_KIND_INVALID", "目标槽位观测类型不正确")
        try:
            slot = PlacementSlotState.model_validate(observation.value or {})
        except ValueError as error:
            raise SkillFailure("TARGET_SLOT_INVALID", f"目标槽位观测无法解析：{error}") from error
        if slot.target_ref != expected_target_ref:
            raise SkillFailure("TARGET_SLOT_MISMATCH", "槽位观测不属于当前 Action 的目标")
        return slot

    def build_approach_plan(
        self,
        held: HeldObjectState,
        slot: PlacementSlotState,
        constraints: PlacementConstraints,
    ) -> PlacementApproachPlan:
        """通过刚体变换生成两个工具的 preplace/release/retreat 目标。

        工具相对物体的位姿来自抓取完成时的 HeldObjectState。放置只把该相对
        变换组合到目标物体位姿，绝不根据工具名称、左右字符串或固定箱宽猜偏移。
        """

        length = sqrt(sum(value * value for value in slot.approach_vector))
        direction = tuple(value / length for value in slot.approach_vector)
        release_object_pose = slot.placement_pose
        current_clearance_m = sum(
            (current - target) * axis
            for current, target, axis in zip(
                held.object_pose.position_m,
                release_object_pose.position_m,
                direction,
                strict=True,
            )
        )
        # 密集码垛时，水平对齐和下降不能合成一条斜线，否则箱体会扫过已经
        # 放好的相邻列并把一侧下钩顶出凹槽。preplace保留当前沿接近方向的
        # 高度，先只完成水平对齐；下一段release再沿接近方向下降到支撑面。
        preplace_clearance_m = max(
            constraints.approach_clearance_m,
            current_clearance_m,
        )
        preplace_object_pose = release_object_pose.model_copy(update={
            "position_m": tuple(
                position + axis * preplace_clearance_m
                for position, axis in zip(release_object_pose.position_m, direction, strict=True)
            ),
            "revision": slot.revision,
        })
        preplace = self._waypoint("preplace", preplace_object_pose, held)
        release = self._waypoint("release", release_object_pose, held)
        unseat = self._offset_waypoint(
            "unseat",
            release,
            tuple(-axis * GROOVE_UNSEAT_TRAVEL_M for axis in direction),
        )
        disengage = self._disengage_waypoint(
            unseat,
            release_object_pose,
            object_size_m=held.object_size_m,
            maximum_distance_m=constraints.retreat_disengage_m,
        )
        # 末端离开凹槽后还要折叠到travel姿态。空钩若只抬半个箱高，
        # 会出现先撤出的工具仍贴着箱沿、随后把箱体重新勾起。这里抬升
        # 一个完整的实时箱高；不追求携物时的高位，避免低位构型的大跨度
        # 多末端IK无法收敛。
        travel_clearance_m = max(
            constraints.approach_clearance_m,
            held.object_size_m[2],
        )
        retreat = self._offset_waypoint(
            "retreat",
            disengage,
            tuple(axis * travel_clearance_m for axis in direction),
        )
        travel_clearance = self._travel_clearance_waypoint(
            retreat,
            release_object_pose,
            object_size_m=held.object_size_m,
            base_position_m=held.base_position_m,
        )
        early_release_tool_ref, early_release_waypoints = self._early_release_plan(
            held, slot, release, direction, constraints
        )
        if early_release_waypoints:
            # 先在空闲侧的临时位置完成水平对齐，随后只做竖直落座；不能从
            # 最终列中心斜向扫到临时位置，否则仍可能碰到已经放好的邻箱。
            staged_support = early_release_waypoints[0]
            preplace_object_pose = staged_support.object_pose.model_copy(update={
                "position_m": tuple(
                    position + axis * preplace_clearance_m
                    for position, axis in zip(
                        staged_support.object_pose.position_m, direction, strict=True
                    )
                ),
                "revision": slot.revision,
            })
            preplace = self._waypoint("preplace", preplace_object_pose, held)
        return PlacementApproachPlan(
            target_ref=slot.target_ref,
            target_revision=slot.revision,
            waypoints=[
                preplace,
                release,
                unseat,
                disengage,
                retreat,
                travel_clearance,
            ],
            early_release_tool_ref=early_release_tool_ref,
            early_release_waypoints=early_release_waypoints,
        )

    def build_free_space_descent_waypoint(
        self,
        preplace: PlacementWaypoint,
        support: PlacementWaypoint,
        approach_vector: tuple[float, float, float],
        *,
        object_height_m: float,
    ) -> PlacementWaypoint | None:
        """把长距离无接触下降和最后的接触落座分开。

        最后一段低速距离按实时箱体高度保留：快速段结束时，新箱底面仍高于
        同层邻箱顶面，进入邻箱侧面高度后才使用接触工艺速度。这里只增加一个
        中间末端目标，实际轨迹仍由Ability/SDK逐点做碰撞检查，不改变通用IK、
        接触或承载判断。
        """

        length = sqrt(sum(value * value for value in approach_vector))
        if length <= 1e-9:
            return None
        direction = tuple(value / length for value in approach_vector)
        total_clearance_m = sum(
            (current - target) * axis
            for current, target, axis in zip(
                preplace.object_pose.position_m,
                support.object_pose.position_m,
                direction,
                strict=True,
            )
        )
        # support 路点会为寻底额外向下延伸 SUPPORT_SEEK_TRAVEL_M；把这段
        # 补回后再留2cm竖直间隙，确保所谓“自由空间”快速段不会进入同层
        # 已放箱体的侧面包络。随后仍有足够距离按低速完成实际落座。
        final_clearance_m = (
            object_height_m + SUPPORT_SEEK_TRAVEL_M + 0.02
        )
        # 小于8cm的可加速行程不值得多创建一个物理Action。
        if total_clearance_m - final_clearance_m < 0.08:
            return None
        offset = tuple(axis * final_clearance_m for axis in direction)
        object_pose = support.object_pose.model_copy(update={
            "position_m": tuple(
                value + delta
                for value, delta in zip(
                    support.object_pose.position_m, offset, strict=True
                )
            ),
        })
        return PlacementWaypoint(
            name="landing",
            object_pose=object_pose,
            targets=[
                ToolPoseTarget(
                    tool_ref=target.tool_ref,
                    target_pose=target.target_pose.model_copy(update={
                        "position_m": tuple(
                            value + delta
                            for value, delta in zip(
                                target.target_pose.position_m, offset, strict=True
                            )
                        ),
                    }),
                )
                for target in support.targets
            ],
        )

    def _early_release_plan(
        self,
        held: HeldObjectState,
        slot: PlacementSlotState,
        release: PlacementWaypoint,
        approach_direction: tuple[float, float, float],
        constraints: PlacementConstraints,
    ) -> tuple[str | None, list[PlacementWaypoint]]:
        """在支撑面承重后先撤内侧手，再由外侧手把箱体推到最终位。

        layout001相邻列的实际净空可能小于下钩完整退出凹槽所需距离。Skill
        只根据ObjectPerception当前快照给出的两侧净空选择临时偏移；Runtime和
        SDK仍只处理通用状态、轨迹与碰撞。双侧均受阻时不猜测危险路径。
        """

        if len(release.targets) != 2 or not slot.lateral_clearance_m:
            return None, []
        inverse_orientation = _quat_inverse(slot.placement_pose.orientation_xyzw)
        candidates: list[tuple[float, float, str]] = []
        for target in release.targets:
            relative_world = tuple(
                target.target_pose.position_m[index]
                - release.object_pose.position_m[index]
                for index in range(3)
            )
            relative_local = _quat_rotate(inverse_orientation, relative_world)
            side = "positive" if relative_local[0] >= 0.0 else "negative"
            clearance = slot.lateral_clearance_m.get(side)
            if clearance is None:
                continue
            current_distance_m = abs(relative_local[0])
            # 临时偏置只补足当前净空相对整套夹具包络的实际缺口，不再叠加
            # 固定经验余量。MuJoCo回放证明只按下钩25mm计算时，打开后的压片
            # 会与同层邻箱碰撞；原有额外3cm又会让箱体落座偏出约7cm，并
            # 依赖失去接触后的空行程把它推回。使用实时净空与52mm夹具包络
            # 的差值，可同时满足退场碰撞和厘米级放置精度。
            required_exit_m = max(
                0.0,
                held.object_size_m[0] / 2
                + TOTE_CLAMP_LATERAL_ENVELOPE_M
                - current_distance_m,
            )
            deficit_m = required_exit_m - clearance
            if deficit_m > 1e-6:
                sign = 1.0 if side == "positive" else -1.0
                candidates.append((deficit_m, sign, target.tool_ref))
        if not candidates:
            return None, []
        if len(candidates) > 1:
            raise SkillFailure(
                "RETREAT_GEOMETRY_INVALID",
                "目标位置两侧净空都不足，无法安全使用逐侧释放和支撑面推入",
            )

        deficit_m, blocked_sign, inner_tool_ref = candidates[0]
        staging_offset_local = (
            -blocked_sign * deficit_m,
            0.0,
            0.0,
        )
        staging_offset_world = _quat_rotate(
            slot.placement_pose.orientation_xyzw, staging_offset_local
        )
        staging_object_pose = slot.placement_pose.model_copy(update={
            "position_m": tuple(
                value + lateral_delta - approach_axis * SUPPORT_SEEK_TRAVEL_M
                for value, lateral_delta, approach_axis in zip(
                    slot.placement_pose.position_m,
                    staging_offset_world,
                    approach_direction,
                    strict=True,
                )
            ),
            "revision": slot.revision,
        })
        staged_support = self._waypoint("release", staging_object_pose, held)
        staged_unseat = self._offset_waypoint(
            "unseat",
            staged_support,
            tuple(-axis * GROOVE_UNSEAT_TRAVEL_M for axis in approach_direction),
        )
        staged_disengage = self._disengage_waypoint(
            staged_unseat,
            staging_object_pose,
            object_size_m=held.object_size_m,
            maximum_distance_m=constraints.retreat_disengage_m,
        )
        # 支撑面短推开始前，把已经退出凹槽的空手直接抬到箱沿之上。
        # 临时偏置正是为了在相邻箱之间提供这段竖直通道；若再增加一次
        # 水平“清出整套夹具”的动作，目标会进入邻箱包络。箱体随后从
        # 空手下方推回最终列，因此无需先为最终箱体位置预留横向空间。
        # 工具当前已经位于箱体中心上方，所需行程应扣除这段现有高度；若
        # 直接再抬“半箱高+夹具包络”，会多走约7cm并迫使IK转动共享躯干，
        # 从而带动仍夹着箱体的另一只手。这里由本次实时箱体尺寸和抓取相对
        # 位姿计算剩余高度，不使用固定的场景坐标。
        early_target = next(
            item for item in staged_disengage.targets
            if item.tool_ref == inner_tool_ref
        )
        current_clearance_m = sum(
            (tool_value - object_value) * axis
            for tool_value, object_value, axis in zip(
                early_target.target_pose.position_m,
                staging_object_pose.position_m,
                approach_direction,
                strict=True,
            )
        )
        required_clearance_m = (
            0.5 * held.object_size_m[2]
            + EARLY_TOOL_TOP_CLEARANCE_M
        )
        early_clearance_m = max(
            constraints.approach_clearance_m,
            required_clearance_m - current_clearance_m,
        )
        staged_clearance = self._offset_waypoint(
            "travel_clearance",
            staged_disengage,
            tuple(axis * early_clearance_m for axis in approach_direction),
        )

        def withdraw_inner(
            waypoint: PlacementWaypoint,
        ) -> PlacementWaypoint:
            moving_target = next(
                item for item in waypoint.targets
                if item.tool_ref == inner_tool_ref
            )
            # 这里只提交正在退出凹槽的内侧工具。仍夹持箱体的外侧工具由
            # required_contact_tools在Action开始时读取实时Pose并固定；若把
            # 两侧都作为运动目标，共享躯干求解可能让空钩重新进入箱体。
            return waypoint.model_copy(
                update={"targets": [moving_target]}
            )

        return inner_tool_ref, [
            staged_support,
            withdraw_inner(staged_unseat),
            withdraw_inner(staged_disengage),
            withdraw_inner(staged_clearance),
        ]

    def _travel_clearance_waypoint(
        self,
        retreat: PlacementWaypoint,
        object_pose: Pose3D,
        *,
        object_size_m: tuple[float, float, float],
        base_position_m: tuple[float, float, float] | None,
    ) -> PlacementWaypoint:
        """在箱体上方把空夹具移到Robot一侧，再折叠到travel姿态。

        直接从箱体正上方做关节插值，钩脚会沿圆弧重新扫过刚放稳的箱体。
        这里使用同一次RobotState的基座位置和实时箱体extent计算水平退出量；
        它只是place-object的局部工艺路点，不改变通用IK、命名姿态或碰撞校验。
        """

        if base_position_m is None:
            return PlacementWaypoint(
                name="travel_clearance",
                object_pose=object_pose,
                targets=retreat.targets,
            )
        toward_base = (
            base_position_m[0] - object_pose.position_m[0],
            base_position_m[1] - object_pose.position_m[1],
            0.0,
        )
        length = sqrt(sum(value * value for value in toward_base))
        if length <= 1e-9:
            return PlacementWaypoint(
                name="travel_clearance",
                object_pose=object_pose,
                targets=retreat.targets,
            )
        direction = tuple(value / length for value in toward_base)
        local_direction = _quat_rotate(
            _quat_inverse(object_pose.orientation_xyzw),
            direction,
        )
        boundary_distance_m = 0.5 * sum(
            abs(axis) * size
            for axis, size in zip(local_direction, object_size_m, strict=True)
        )
        # 只把load frame移出箱体边界仍不够：travel关节折叠时下钩会沿圆弧
        # 外摆，曾实际扫到相邻的高箱。退出量覆盖整套夹具横向包络后再折叠，
        # 不需要识别邻箱ID，也不放宽任何碰撞判断。
        required_clearance_m = (
            boundary_distance_m
            + GROOVE_DISENGAGE_LOAD_FRAME_CLEARANCE_M
            + TOTE_CLAMP_LATERAL_ENVELOPE_M
            + SUPPORTED_SLIDE_CLEARANCE_MARGIN_M
        )
        targets: list[ToolPoseTarget] = []
        for target in retreat.targets:
            relative = tuple(
                target.target_pose.position_m[index] - object_pose.position_m[index]
                for index in range(3)
            )
            current_distance_m = sum(
                value * axis
                for value, axis in zip(relative, direction, strict=True)
            )
            travel_m = max(0.0, required_clearance_m - current_distance_m)
            targets.append(ToolPoseTarget(
                tool_ref=target.tool_ref,
                target_pose=target.target_pose.model_copy(update={
                    "position_m": tuple(
                        value + travel_m * axis
                        for value, axis in zip(
                            target.target_pose.position_m,
                            direction,
                            strict=True,
                        )
                    ),
                }),
            ))
        return PlacementWaypoint(
            name="travel_clearance",
            object_pose=object_pose,
            targets=targets,
        )

    def _disengage_waypoint(
        self,
        release: PlacementWaypoint,
        object_pose: Pose3D,
        object_size_m: tuple[float, float, float],
        *,
        maximum_distance_m: float,
    ) -> PlacementWaypoint:
        """沿两个实时工具的横向轴线退出凹槽。

        工具打开只消除了压紧力，并不意味着下钩已经退出侧面凹槽。固定外撤
        6 cm 在孤立箱上可行，却会在密集码垛的厘米级列间隙中撞到邻箱。
        两个load frame位于左右钩脚处，它们的连线才是夹具实际横向轴线；
        不能使用“物体中心到单个工具”的径向，因为凹槽允许工具沿槽长方向
        偏置，径向会把毫米级退出错误放大成斜向运动。这里只补足到实时箱体
        边界外的必要余量；maximum_distance_m仍是Skill策略允许的机械动作
        上限，而不是固定轨迹。
        """

        if len(release.targets) != 2:
            raise SkillFailure("RETREAT_GEOMETRY_INVALID", "双侧放置必须提供两个实时工具位姿")
        first, second = release.targets
        across = tuple(
            second.target_pose.position_m[index] - first.target_pose.position_m[index]
            for index in range(3)
        )
        across_length = sqrt(sum(value * value for value in across))
        if across_length <= 1e-9:
            raise SkillFailure("RETREAT_GEOMETRY_INVALID", "左右工具无法形成有效横向轴线")
        across_axis = tuple(value / across_length for value in across)

        # extent在物体局部坐标系表达。将工具横向轴变换到物体局部坐标后，
        # 用盒体support distance求真实边界；这样箱体旋转后仍不会退错方向。
        local_across_axis = _quat_rotate(
            _quat_inverse(object_pose.orientation_xyzw),
            across_axis,
        )
        boundary_distance_m = 0.5 * sum(
            abs(axis) * size
            for axis, size in zip(local_across_axis, object_size_m, strict=True)
        )

        targets: list[ToolPoseTarget] = []
        for target in release.targets:
            relative = tuple(
                target.target_pose.position_m[index] - object_pose.position_m[index]
                for index in range(3)
            )
            sign = 1.0 if sum(
                value * axis for value, axis in zip(relative, across_axis, strict=True)
            ) >= 0.0 else -1.0
            direction = tuple(sign * axis for axis in across_axis)
            current_distance_m = sum(
                value * axis for value, axis in zip(relative, direction, strict=True)
            )
            required_distance_m = max(
                0.0,
                boundary_distance_m
                + GROOVE_DISENGAGE_LOAD_FRAME_CLEARANCE_M
                - current_distance_m,
            )
            if required_distance_m > maximum_distance_m:
                raise SkillFailure(
                    "RETREAT_GEOMETRY_INVALID",
                    "当前钩脚退出箱体凹槽所需距离超过Robot Skill允许上限",
                )
            targets.append(ToolPoseTarget(
                tool_ref=target.tool_ref,
                target_pose=target.target_pose.model_copy(update={
                    "position_m": tuple(
                        value + required_distance_m * axis
                        for value, axis in zip(
                            target.target_pose.position_m,
                            direction,
                            strict=True,
                        )
                    ),
                }),
            ))
        return PlacementWaypoint(name="disengage", object_pose=object_pose, targets=targets)

    @staticmethod
    def _offset_waypoint(
        name: str,
        source: PlacementWaypoint,
        offset: tuple[float, float, float],
    ) -> PlacementWaypoint:
        return PlacementWaypoint(
            name=name,
            object_pose=source.object_pose,
            targets=[
                ToolPoseTarget(
                    tool_ref=target.tool_ref,
                    target_pose=target.target_pose.model_copy(update={
                        "position_m": tuple(
                            value + delta
                            for value, delta in zip(
                                target.target_pose.position_m,
                                offset,
                                strict=True,
                            )
                        ),
                    }),
                )
                for target in source.targets
            ],
        )

    def horizontal_alignment_error_m(
        self,
        held: HeldObjectState,
        slot: PlacementSlotState,
    ) -> float:
        """返回实时持物中心与槽位中心的水平距离。

        place-object 只负责槽位上方的短距离末端动作。携物中心尚未对齐时，
        让双臂直接补偿水平误差会把箱体当成长力臂扫过工位，并可能在轨迹中
        失去一侧接触；这种跨工位移动属于 semantic-navigation，而不是放置
        Skill 的局部恢复。这里使用 Ability 刚刚重观测的两个实时位姿，不读取
        Semantic Map，也不引入另一套经验距离。
        """

        if held.object_pose.frame_id != slot.placement_pose.frame_id:
            raise SkillFailure("PLACEMENT_FRAME_MISMATCH", "持物与槽位位姿必须在同一坐标系")
        return sqrt(sum(
            (held.object_pose.position_m[index] - slot.placement_pose.position_m[index]) ** 2
            for index in (0, 1)
        ))

    @staticmethod
    def local_alignment_limit_m(
        held: HeldObjectState,
        constraints: PlacementConstraints,
    ) -> float:
        """按实时箱体足迹给出工作位内的最大水平接近距离。

        该上限只区分“末端局部接近”和“底盘跨工位移动”，不替代Robot Profile
        下的IK与碰撞检查。半个箱体长边覆盖中心对齐，已有 approach_clearance
        再容纳导航栅格、travel 工位和实时持物偏移之间的正常差异；最终是否可达
        仍由后续 IK 与连续碰撞检查决定，不能在这里用厘米级差值提前拒绝。
        """

        return max(held.object_size_m[:2]) / 2 + constraints.approach_clearance_m

    def _waypoint(self, name: str, object_pose: Pose3D, held: HeldObjectState) -> PlacementWaypoint:
        targets = [
            ToolPoseTarget(
                tool_ref=tool_ref,
                target_pose=_transfer_rigid_pose(
                    source_object=held.object_pose,
                    source_tool=held.tool_poses[tool_ref],
                    target_object=object_pose,
                ),
            )
            for tool_ref in held.tool_refs
        ]
        return PlacementWaypoint(name=name, object_pose=object_pose, targets=targets)

    def parse_robot_state(self, observation: Observation) -> RobotStateValue:
        if observation.kind != ROBOT_STATE_OBSERVATION:
            raise SkillFailure("ROBOT_STATE_KIND_INVALID", "Robot 状态 Action 返回了错误观测")
        try:
            return RobotStateValue.model_validate(observation.value or {})
        except ValueError as error:
            raise SkillFailure("ROBOT_STATE_INVALID", f"Robot 状态无法解析：{error}") from error

    def support_transfer_confirmed(
        self,
        observations: list[Observation],
        *,
        object_ref: str,
        target_ref: str,
    ) -> bool:
        """读取 Ability 对最后一段下降做出的实时支撑交接结论。

        这里只接受当前 Action 的 Observation；Skill 不根据轨迹进度、目标
        Pose 或 Runtime 接触字段自行猜测箱体是否已经落座。
        """

        for item in reversed(observations):
            if item.kind != SUPPORT_TRANSFER_OBSERVATION:
                continue
            value = item.value or {}
            transfer = value.get("support_transfer") or {}
            return bool(
                value.get("support_transfer_confirmed") is True
                and transfer.get("object_ref") == object_ref
                and transfer.get("target_ref") == target_ref
                and transfer.get("contact") is True
                and transfer.get("within_target_xy") is True
            )
        return False

    def supported_slide_correction_target(
        self,
        observation: Observation,
        plan: PlacementApproachPlan,
        remaining_target: ToolPoseTarget,
    ) -> ToolPoseTarget:
        """把箱体的实时水平误差转换成外侧工具的一次短推位移。

        调用方会在外侧工具仍贴着箱壁时执行该目标，因此应从工具实时Pose
        叠加箱体中心到目标中心的水平差，而不能复用一个绝对计划Pose。
        后者在首轮推入存在接触顺应时会让补推重复同一终点。竖直位置和
        工具姿态保持不变；实际路径仍由Ability/SDK检查碰撞和力限。
        """

        if observation.kind != TARGET_POSE_OBSERVATION:
            raise SkillFailure(
                "SUPPORTED_SLIDE_OBSERVATION_INVALID",
                "支撑面推入后的物体观测类型不正确",
            )
        try:
            observed = TargetPoseValue.model_validate(observation.value or {})
        except ValueError as error:
            raise SkillFailure(
                "SUPPORTED_SLIDE_OBSERVATION_INVALID",
                f"支撑面推入后的物体观测无法解析：{error}",
            ) from error
        expected = plan.waypoints[1].object_pose
        if observed.pose.frame_id != expected.frame_id:
            raise SkillFailure(
                "SUPPORTED_SLIDE_FRAME_MISMATCH",
                "支撑面短推的实时箱体位姿与目标位姿不在同一坐标系",
            )
        horizontal_delta = (
            expected.position_m[0] - observed.pose.position_m[0],
            expected.position_m[1] - observed.pose.position_m[1],
            0.0,
        )
        return remaining_target.model_copy(update={
            "target_pose": remaining_target.target_pose.model_copy(update={
                "position_m": tuple(
                    value + delta
                    for value, delta in zip(
                        remaining_target.target_pose.position_m,
                        horizontal_delta,
                        strict=True,
                    )
                ),
            }),
        })

    def postplace_clearance_targets(
        self,
        state: RobotStateValue,
        held: HeldObjectState,
        placed: PlacedObjectState,
        *,
        maximum_disengage_m: float,
    ) -> list[ToolPoseTarget]:
        """按放置完成后的真实箱体位姿把两只空工具移到基座侧。

        支撑面接触会让箱体停在目标 Region 内而非理论中心。若仍按旧中心
        计算撤离距离，关节折叠时钩脚可能扫回箱体。这里只根据已经
        通过正式 VerifyPlacement 的最终位姿生成一次水平清空动作。
        """

        targets = [
            ToolPoseTarget(
                tool_ref=tool_ref,
                target_pose=self.tool_pose_from_robot_state(
                    state,
                    tool_ref,
                    revision=placed.scene_revision,
                ),
            )
            for tool_ref in held.tool_refs
        ]
        current = PlacementWaypoint(
            name="travel_clearance",
            object_pose=placed.final_pose,
            targets=targets,
        )
        # 箱体在支撑面上沉降或被最后一只工具推正后，真实中心可能已经偏离
        # 原放置计划。先按当前左右工具轴和最终箱体位姿重新退钩，不能只沿
        # 基座方向平移；否则侧面的钩仍可能挂在箱体上，却被动作到位掩盖。
        disengaged = self._disengage_waypoint(
            current,
            placed.final_pose,
            held.object_size_m,
            maximum_distance_m=maximum_disengage_m,
        )
        base_position = (
            state.base_pose.position if state.base_pose is not None
            else held.base_position_m
        )
        return self._travel_clearance_waypoint(
            disengaged,
            placed.final_pose,
            object_size_m=held.object_size_m,
            base_position_m=base_position,
        ).targets

    def remaining_tool_is_stable(
        self,
        load: ToolLoadObservationValue,
        *,
        tool_ref: str,
        object_ref: str,
    ) -> bool:
        return bool(
            load.condition_satisfied
            and not load.slip_detected
            and not load.overload_detected
            and not load.sensor_fault
            and len(load.tools) == 1
            and load.tools[0].tool_ref == tool_ref
            and load.tools[0].available
        )


    @staticmethod
    def release_result_confirms_tools_empty(observation: Observation) -> bool:
        value = observation.value or {}
        return bool(value.get("released") is True and value.get("gripper_empty") is True)

    def release_result_confirms_tool(
        self,
        observation: Observation,
        *,
        tool_ref: str,
        object_ref: str,
    ) -> bool:
        if observation.kind != RELEASED_OBJECT_OBSERVATION:
            return False
        value = observation.value or {}
        tools = value.get("tools")
        tool = tools.get(tool_ref) if isinstance(tools, dict) else None
        released_refs = value.get("released_tool_refs")
        return bool(
            value.get("object_ref") == object_ref
            and tool
            and value.get("released") is True
            and isinstance(released_refs, list)
            and tool_ref in released_refs
        )

    def parse_placed_state(self, observation: Observation, *, object_ref: str, target: PlacementTarget) -> PlacedObjectState:
        """只信正式 Observation 的嵌套 state，不读取 Action output 或来源名称猜结果。"""

        if observation.kind != PLACED_OBJECT_OBSERVATION:
            raise SkillFailure("PLACEMENT_EVIDENCE_KIND_INVALID", "稳定性观测类型不正确")
        try:
            verified = PlacementVerificationObservation.model_validate(observation.value or {})
        except ValueError as error:
            raise SkillFailure("PLACEMENT_EVIDENCE_INVALID", f"稳定性观测无法解析：{error}") from error
        placed = verified.state
        if placed.object_ref != object_ref or placed.target_ref != target.target_ref:
            raise SkillFailure("PLACEMENT_EVIDENCE_MISMATCH", "稳定性证据与当前对象或目标不匹配")
        # 目标区域成员关系由ObjectPerception结合实时槽位几何判断。这里不再
        # 叠加固定中心距离：现场已出现箱体稳定、支撑和Region均成立，却因
        # 4cm边界相差几毫米被误报失败。明显倾倒仍由角度与稳定性证据排除。
        if (
            not verified.independent_verification or not verified.within_target
            or not verified.stable or not verified.support_contact
            or not verified.gripper_empty or not placed.stable or not placed.gripper_empty
            # 感知侧的 stable 只表示短时间内位姿不再变化。箱体斜靠在邻箱
            # 或钩脚上时也可能暂时静止，因此最终完成仍要排除肉眼可见的
            # 大角度倾斜；这里与释放前支撑判断共用粗粒度角度边界。
            or placed.orientation_error_rad > MAX_SUPPORTED_ORIENTATION_ERROR_RAD
            or placed.observed_duration_ms < target.stability_duration_ms
            or verified.observed_duration_ms < target.stability_duration_ms
        ):
            raise SkillFailure("PLACEMENT_NOT_STABLE", "物体尚未满足稳定放置完成条件")
        return placed

    def support_contact_confirmed(
        self,
        observation: Observation,
        *,
        object_ref: str,
        target: PlacementTarget,
    ) -> bool:
        """确认释放前箱体已经进入目标且由支撑面承重。

        此时保留侧工具仍贴着箱体，最终放置校验必然会报告
        ``gripper_empty=false``。这里不能复用最终完成条件；只读取同一份
        实时 Observation 中的目标成员关系、支撑接触和观测时长，最终释放
        后仍由 ``parse_placed_state`` 严格确认工具为空和放置稳定。
        """

        return self.supported_placed_state(
            observation,
            object_ref=object_ref,
            target=target,
        ) is not None

    def supported_placed_state(
        self,
        observation: Observation,
        *,
        object_ref: str,
        target: PlacementTarget,
    ) -> PlacedObjectState | None:
        """返回已进入目标并由支撑面承重的实时物体状态。

        密集放置时空钩可能仍贴着箱体，使最终 ``stable/gripper_empty`` 尚未
        成立；该状态只用于按真实箱体位姿清出工具，不能直接作为放置完成。
        """

        if observation.kind != PLACED_OBJECT_OBSERVATION:
            return None
        try:
            verified = PlacementVerificationObservation.model_validate(
                observation.value or {}
            )
        except ValueError:
            return None
        placed = verified.state
        if not (
            placed.object_ref == object_ref
            and placed.target_ref == target.target_ref
            and verified.independent_verification
            and verified.within_target
            and verified.support_contact
            and placed.orientation_error_rad
            <= MAX_SUPPORTED_ORIENTATION_ERROR_RAD
            and verified.observed_duration_ms >= target.stability_duration_ms
        ):
            return None
        return placed

    def supported_slide_retry_allowed(
        self,
        observation: Observation,
        *,
        object_ref: str,
        target: PlacementTarget,
    ) -> bool:
        """判断箱体是否仍在支撑面上，可以再补一次水平短推。

        这里不要求已达到最终中心，否则“需要补推”永远不可达；但箱体
        必须仍在目标区、由目标支撑承重且没有明显倾倒。调用方只允许一次
        基于新观测的补推，不形成毫米级循环试探。
        """

        if observation.kind != PLACED_OBJECT_OBSERVATION:
            return False
        try:
            verified = PlacementVerificationObservation.model_validate(
                observation.value or {}
            )
        except ValueError:
            return False
        placed = verified.state
        return bool(
            placed.object_ref == object_ref
            and placed.target_ref == target.target_ref
            and verified.independent_verification
            and verified.within_target
            and verified.support_contact
            and placed.orientation_error_rad
            <= MAX_SUPPORTED_ORIENTATION_ERROR_RAD
            and verified.observed_duration_ms >= target.stability_duration_ms
        )

    def supported_slide_completed(
        self,
        observation: Observation,
        *,
        object_ref: str,
        target: PlacementTarget,
    ) -> bool:
        """确认支撑面推进已把箱体送入对应列，而不只是在托盘上静止。"""

        placed = self.supported_placed_state(
            observation,
            object_ref=object_ref,
            target=target,
        )
        return bool(
            placed is not None
            and placed.position_error_m <= SUPPORTED_SLIDE_COMPLETION_TOLERANCE_M
        )

    def recover_approach(self, *, error_code: str | None, recoveries: int, constraints: PlacementConstraints) -> LocalRecoveryDecision:
        if error_code in self._RECOVERABLE_APPROACH_ERRORS and recoveries < constraints.max_approach_recoveries:
            return LocalRecoveryDecision(disposition="refresh_target", reason="接近受阻，刷新槽位后重建双工具短计划")
        return LocalRecoveryDecision(disposition="request_agent", reason="接近失败超出局部恢复范围")

    def recover_retreat(self, *, error_code: str | None, recoveries: int, constraints: PlacementConstraints) -> LocalRecoveryDecision:
        if error_code in self._RECOVERABLE_RETREAT_ERRORS and recoveries < constraints.max_retreat_recoveries:
            return LocalRecoveryDecision(disposition="retry", reason="沿同一双工具安全撤离目标有限重试")
        return LocalRecoveryDecision(disposition="request_agent", reason="释放后无法安全撤离，需要 Robot Agent 决策")


def _transfer_rigid_pose(*, source_object: Pose3D, source_tool: Pose3D, target_object: Pose3D) -> Pose3D:
    if source_object.frame_id != source_tool.frame_id or source_object.frame_id != target_object.frame_id:
        raise SkillFailure("PLACEMENT_FRAME_MISMATCH", "物体、工具和目标位姿必须在同一坐标系")
    inverse_object = _quat_inverse(source_object.orientation_xyzw)
    relative_position = _quat_rotate(
        inverse_object,
        tuple(a - b for a, b in zip(source_tool.position_m, source_object.position_m, strict=True)),
    )
    relative_orientation = _quat_multiply(inverse_object, source_tool.orientation_xyzw)
    target_position_offset = _quat_rotate(target_object.orientation_xyzw, relative_position)
    return Pose3D(
        frame_id=target_object.frame_id,
        position_m=tuple(a + b for a, b in zip(target_object.position_m, target_position_offset, strict=True)),
        orientation_xyzw=_quat_normalize(_quat_multiply(target_object.orientation_xyzw, relative_orientation)),
        observed_at=target_object.observed_at,
        revision=target_object.revision,
    )


def _quat_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_inverse(value):
    x, y, z, w = value
    norm = x * x + y * y + z * z + w * w
    if norm <= 1e-12:
        raise SkillFailure("PLACEMENT_ORIENTATION_INVALID", "源物体姿态不是有效四元数")
    return (-x / norm, -y / norm, -z / norm, w / norm)


def _quat_normalize(value):
    norm = sqrt(sum(item * item for item in value))
    if norm <= 1e-12:
        raise SkillFailure("PLACEMENT_ORIENTATION_INVALID", "目标工具姿态不是有效四元数")
    return tuple(item / norm for item in value)


def _quat_rotate(quaternion, vector):
    q = _quat_normalize(quaternion)
    vector_q = (vector[0], vector[1], vector[2], 0.0)
    rotated = _quat_multiply(_quat_multiply(q, vector_q), _quat_inverse(q))
    return rotated[:3]
