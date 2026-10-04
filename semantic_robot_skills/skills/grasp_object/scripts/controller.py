"""抓取 Skill 的 schema v2 确定性 Controller。"""

from __future__ import annotations

from typing import Literal

from semantic_robot_skill_sdk import Action, FeedbackRequest

from .models import (
    CloseToolsParameters,
    GenerateCandidatesParameters,
    GraspCandidate,
    GraspObjectInput,
    GetRobotStateParameters,
    HoldObjectParameters,
    LiftObjectParameters,
    LocateObjectParameters,
    MoveTargetsParameters,
    PlanTransportPostureParameters,
    SafeStopGraspParameters,
    SetOpeningParameters,
    ToolPoseTarget,
    ToolSetpoint,
    VerifyGraspParameters,
    VerifyToolLoadParameters,
    VerifyPregraspParameters,
)


# 这些值是周转箱抓取实现的默认动作策略，不属于调用者需要理解的业务输入。
# 设备相关上限仍由 RobotDeployment/Profile 约束，Ability 会在下发前取更严格值。
MINIMUM_TARGET_CONFIDENCE = 0.65
MAXIMUM_GRIP_FORCE_N = 60.0
HOLDING_POSITION_M = 0.015
# 周转箱下钩的 insert 只负责进入短边连续凹槽的待落座区域，不需要命中
# 单个位姿点。1.5 cm 能覆盖箱体被首侧承载后的顺应位移，且仍小于凹槽深度；
# 真正抓牢继续由后续 seat、闭合和双侧承载验证决定。设备的关节、力和碰撞
# 上限仍由 RobotDeployment 与 Ability 负责，不在这里重复设置毫米级门槛。
MAXIMUM_PREGRASP_ERROR_M = 0.015
# 该值只决定箱体重观测后是否补做一次槽口对齐动作，不参与成功/失败判定。
# 保持较小阈值可避免把箱体位移直接留给最后一段水平 insert 吸收。
SECONDARY_REALIGN_SHIFT_M = 0.004
LIFT_HEIGHT_TOLERANCE_M = 0.001
STABLE_DURATION_MS = 500


class DepalletizingGraspController:
    def locate_target(self, skill_input: GraspObjectInput) -> Action:
        return Action.from_model(
            action_type="perception.locate_object",
            parameters=LocateObjectParameters(
                object_ref=skill_input.object_ref,
                minimum_confidence=MINIMUM_TARGET_CONFIDENCE,
                pose_hint=skill_input.target.pose_hint,
                extent_hint_m=skill_input.target.extent_hint_m,
                category_hint=skill_input.target.category_hint,
            ),
            timeout_seconds=8,
            label="观测周转箱真实位姿",
        )

    def generate_candidates(
        self,
        skill_input: GraspObjectInput,
        *,
        target_pose,
        object_extent_m,
        target_revision: str,
        active_strategy: str,
        engaged_tool_ref: str | None = None,
        secondary_resume_phase: Literal["insert"] | None = None,
    ) -> Action:
        return Action.from_model(
            action_type="grasp.generate_candidates",
            parameters=GenerateCandidatesParameters(
                object_extent_m=object_extent_m,
                target_revision=target_revision,
                preferred_strategy=active_strategy,
                maximum_candidates=3,
                object_ref=skill_input.object_ref,
                target_pose=target_pose,
                engaged_tool_ref=engaged_tool_ref,
                secondary_resume_phase=secondary_resume_phase,
            ),
            timeout_seconds=6,
            label="生成双工具抓取候选",
        )

    def plan_approach(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        target_revision: str,
    ) -> list[Action]:
        """生成首轮接近动作；单侧策略在形成首侧接触前只移动首侧。"""

        targets = candidate.hook_insert_poses
        primary_side = self.primary_side(candidate)
        if primary_side is not None:
            primary_ref = self.tool_ref_for_side(candidate, primary_side)
            targets = [item for item in targets if item.tool_ref == primary_ref]
        # 非工作臂保持当前travel/安全姿态。箱体外拉并重新观测后，
        # 第二侧才根据真实新位姿生成紧凑接近路径。
        actions = [self.open_tools(skill_input, candidate=candidate)]
        actions.extend(
            self._engagement_actions(
                skill_input,
                candidate,
                targets,
                target_revision,
                label="核对首侧钩脚已进入凹槽",
                include_clearance=True,
            )
        )
        return actions

    def plan_secondary_engagement(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        primary_side: str,
        target_revision: str,
    ) -> list[Action]:
        """保持外拉侧承载，只提交另一侧工具的接近目标。"""

        secondary_side = "right" if primary_side == "left" else "left"
        primary_ref = self.tool_ref_for_side(candidate, primary_side)
        secondary_ref = self.tool_ref_for_side(candidate, secondary_side)

        def secondary(items: list[ToolPoseTarget]) -> list[ToolPoseTarget]:
            return [next(item for item in items if item.tool_ref == secondary_ref)]

        clearance = secondary(candidate.clearance_poses)
        transfer = secondary(candidate.transfer_poses)
        pregrasp = secondary(candidate.pregrasp_poses)
        insert = secondary(candidate.hook_insert_poses)
        return [
            # 第二侧在首侧外拉期间保持travel姿态。重观测确认完整末端通道
            # 后，clearance才让它在身体附近短前展并抬高；第一侧由
            # required_contact_tools在整段运动中持续监控。
            self.move_targets(
                skill_input,
                candidate,
                clearance,
                "clearance",
                target_revision,
                0.15,
                required_contact_tools=[primary_ref],
            ),
            self.move_targets(
                skill_input,
                candidate,
                transfer,
                "transfer",
                target_revision,
                0.15,
                required_contact_tools=[primary_ref],
            ),
            # 第二手在连续凹槽中心线外侧垂直下降，再执行最后的短水平
            # 插入；两段都保持首侧承载，不绕箱体前角，也不斜扫箱沿。
            self.move_targets(
                skill_input,
                candidate,
                pregrasp,
                "pregrasp",
                target_revision,
                0.12,
                required_contact_tools=[primary_ref],
            ),
            self.move_targets(
                skill_input,
                candidate,
                insert,
                "insert",
                target_revision,
                0.05,
                required_contact_tools=[primary_ref],
            ),
            Action.from_model(
                action_type="perception.verify_pregrasp",
                parameters=VerifyPregraspParameters(
                    object_ref=skill_input.object_ref,
                    candidate_id=candidate.candidate_id,
                    planned_object_pose=candidate.planned_object_pose,
                    expected_targets=insert,
                    target_revision=target_revision,
                    maximum_position_error_m=MAXIMUM_PREGRASP_ERROR_M,
                ),
                timeout_seconds=5,
                label="保持首侧承载并核对第二侧插入位置",
            ),
        ]

    def _engagement_actions(
        self,
        skill_input: GraspObjectInput,
        candidate: GraspCandidate,
        targets,
        target_revision: str,
        *,
        label: str,
        include_clearance: bool = True,
    ) -> list[Action]:
        selected = {value.tool_ref for value in targets}
        clearance = [
            item for item in candidate.clearance_poses if item.tool_ref in selected
        ]
        transfer = [
            item for item in candidate.transfer_poses if item.tool_ref in selected
        ]
        approach = [
            item for item in candidate.approach_poses if item.tool_ref in selected
        ]
        # 双侧工具和躯干是一个耦合运动系统。direct_bilateral 从 transfer 开始
        # 就必须联合规划；若逐侧移动，首侧求解出的躯干姿态可能与尚在 travel 的
        # 另一侧夹具冲突。单侧外拉候选传入的 transfer 本来就只有一侧，因此同一
        # 调用同时覆盖两种策略，不需要额外执行分支。
        transfer_actions: list[Action] = []
        if include_clearance:
            transfer_actions.append(
                self.move_targets(
                    skill_input,
                    candidate,
                    clearance,
                    "clearance",
                    target_revision,
                    0.15,
                )
            )
        transfer_actions.append(
            self.move_targets(
                skill_input,
                candidate,
                transfer,
                "transfer",
                target_revision,
                0.15,
            )
        )
        return [
            *transfer_actions,
            self.move_targets(
                skill_input,
                candidate,
                approach,
                "pregrasp",
                target_revision,
                0.15,
            ),
            self.move_targets(
                skill_input,
                candidate,
                targets,
                "insert",
                target_revision,
                0.08,
            ),
            Action.from_model(
                action_type="perception.verify_pregrasp",
                parameters=VerifyPregraspParameters(
                    object_ref=skill_input.object_ref,
                    candidate_id=candidate.candidate_id,
                    planned_object_pose=candidate.planned_object_pose,
                    expected_targets=list(targets),
                    target_revision=target_revision,
                    maximum_position_error_m=MAXIMUM_PREGRASP_ERROR_M,
                ),
                timeout_seconds=5,
                label=label,
            ),
        ]

    @staticmethod
    def primary_side(candidate: GraspCandidate) -> str | None:
        if candidate.strategy == "left_extract_first":
            return "left"
        if candidate.strategy == "right_extract_first":
            return "right"
        return None

    @staticmethod
    def tool_ref_for_side(candidate: GraspCandidate, side: str) -> str:
        return next(
            item.tool_ref for item in candidate.tool_targets if item.side == side
        )

    def move_targets(
        self,
        skill_input: GraspObjectInput,
        candidate: GraspCandidate,
        targets,
        purpose: str,
        target_revision: str,
        speed: float,
        *,
        required_contact_tools: list[str] | None = None,
    ) -> Action:
        return Action.from_model(
            action_type="motion.move_end_effector",
            parameters=MoveTargetsParameters(
                targets=list(targets),
                purpose=purpose,
                object_ref=skill_input.object_ref,
                candidate_id=candidate.candidate_id,
                target_revision=target_revision,
                maximum_speed_mps=speed,
                max_contact_force_n=(
                    MAXIMUM_GRIP_FORCE_N if purpose in {"insert", "seat"} else None
                ),
                required_contact_tools=tuple(required_contact_tools or ()),
            ),
            timeout_seconds=15,
            feedback=FeedbackRequest(
                observation_kinds=["end_effector_progress", "collision_proximity"],
                interval_ms=200,
            ),
            label=f"执行双工具 {purpose} 短动作",
        )

    def open_tools(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        tool_refs: list[str] | None = None,
    ) -> Action:
        """打开全部工具，或只释放外拉阶段临时夹持的指定工具。"""

        selected = set(tool_refs or ())
        values = [
            item.model_copy(
                update={
                    "maximum_force_n": min(item.maximum_force_n, MAXIMUM_GRIP_FORCE_N)
                }
            )
            for item in candidate.opening_setpoints
            if not selected or item.tool_ref in selected
        ]
        return Action.from_model(
            action_type="gripper.set_opening",
            parameters=SetOpeningParameters(tools=values),
            timeout_seconds=8,
            label="打开选定夹具到预抓取开度",
        )

    def close_tools(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        tool_refs: list[str],
    ) -> Action:
        # 候选由GraspPlanning结合当前工具Profile生成。Skill只转发本次候选，
        # 不再用一个经验常量覆盖Provider选择；Ability/SDK仍校验设备物理上限。
        values = [
            item
            for item in candidate.clamp_setpoints
            if item.tool_ref in set(tool_refs)
        ]
        return Action.from_model(
            action_type="gripper.close",
            parameters=CloseToolsParameters(
                object_ref=skill_input.object_ref,
                tools=values,
                candidate_id=candidate.candidate_id,
                grasp_pose=skill_input_pose(candidate),
            ),
            timeout_seconds=8,
            feedback=FeedbackRequest(
                observation_kinds=["grasp_contact", "tool_state"], interval_ms=100
            ),
            label="闭合周转箱夹具并确认接触",
        )

    def recover_after_partial_close(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        target_revision: str,
    ) -> list[Action]:
        """释放未形成稳定承载的夹具，并沿原接近路径退出箱沿。

        close/seat 失败时下钩可能仍在凹槽内。下一轮不能直接从 clearance
        重新接近，否则运动规划会在当前接触姿态的 0 秒位置拒绝整条路径。
        这里仅反向执行已有候选的两段局部几何：先回到 insert 高度解除
        上沿承载，再水平退到 approach 点。``extract`` 会继续检查邻箱、
        托盘和 Robot 自碰撞，只允许当前目标箱体上的受控退出接触。
        """

        return [
            self.open_tools(skill_input, candidate=candidate),
            self.move_targets(
                skill_input,
                candidate,
                candidate.hook_insert_poses,
                "extract",
                target_revision,
                0.03,
            ),
            self.move_targets(
                skill_input,
                candidate,
                candidate.approach_poses,
                "extract",
                target_revision,
                0.05,
            ),
        ]

    def seat_tools(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        tool_refs: list[str],
        target_revision: str,
        required_contact_tools: list[str] | None = None,
    ) -> Action:
        """轻微抬起已插入凹槽的工具，让下钩贴合凹槽上沿内侧。

        insert先把完整钩脚水平送入槽内，seat只执行短距离向上就位；随后才
        闭合上夹片。这里不依赖任何额外内部构件；是否形成可用接触仍由
        VerifyToolLoad根据实时力、方向和相对运动判断。
        """

        selected = set(tool_refs)
        targets = [
            item for item in candidate.hook_seat_poses if item.tool_ref in selected
        ]
        return self.move_targets(
            skill_input,
            candidate,
            targets,
            "seat",
            target_revision,
            0.02,
            required_contact_tools=required_contact_tools,
        )

    def lift_object(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        distance_m: float | None = None,
    ) -> Action:
        return Action.from_model(
            action_type="motion.lift_held_object",
            parameters=LiftObjectParameters(
                object_ref=skill_input.object_ref,
                tools=skill_input.tool_refs,
                candidate_id=candidate.candidate_id,
                distance_m=(
                    skill_input.minimum_lift_height_m
                    if distance_m is None
                    else distance_m
                ),
                # 抬升属于周转箱抓取策略，不由通用 SDK 猜测“持物即统一降速”。
                # 5 cm/s 在真实接触 Smoke 的可用区间内，同时避免过慢轨迹让
                # 箱体长期单侧受力后从另一侧抓取凹槽滑出。实际滑移仍由 Runtime
                # 实时检测并立即 stop+hold，不能靠降低安全阈值换取通过。
                maximum_speed_mps=0.05,
            ),
            timeout_seconds=10,
            feedback=FeedbackRequest(
                observation_kinds=["lift_progress", "grasp_contact"], interval_ms=100
            ),
            label="双侧稳定承载后抬升",
        )

    def verify_grasp(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        initial_object_pose,
    ) -> Action:
        return Action.from_model(
            action_type="perception.verify_grasp",
            parameters=VerifyGraspParameters(
                object_ref=skill_input.object_ref,
                tools=skill_input.tool_refs,
                candidate_id=candidate.candidate_id,
                initial_object_pose=initial_object_pose,
                minimum_lift_height_m=skill_input.minimum_lift_height_m,
                lift_height_tolerance_m=LIFT_HEIGHT_TOLERANCE_M,
                stable_duration_ms=STABLE_DURATION_MS,
            ),
            timeout_seconds=8,
            label="独立验证双侧承载和抬升",
        )

    def hold_object(self, skill_input: GraspObjectInput, *, reason: str) -> Action:
        tools = [
            ToolSetpoint(
                tool_ref=ref,
                target_position_m=HOLDING_POSITION_M,
                maximum_force_n=MAXIMUM_GRIP_FORCE_N,
                hold=True,
            )
            for ref in skill_input.tool_refs
        ]
        return Action.from_model(
            action_type="gripper.hold_object",
            parameters=HoldObjectParameters(
                object_ref=skill_input.object_ref, tools=tools, reason=reason
            ),
            timeout_seconds=5,
            label="安全保持双侧夹具",
        )

    def safe_stop(
        self, skill_input: GraspObjectInput, *, reason: str, mode: str
    ) -> Action:
        tools = [
            ToolSetpoint(
                tool_ref=ref,
                target_position_m=HOLDING_POSITION_M,
                maximum_force_n=MAXIMUM_GRIP_FORCE_N,
                hold=True,
            )
            for ref in skill_input.tool_refs
        ]
        return Action.from_model(
            action_type="gripper.hold_object",
            parameters=SafeStopGraspParameters(
                object_ref=skill_input.object_ref, tools=tools, reason=reason, mode=mode
            ),
            timeout_seconds=5,
            label="安全停止抓取执行",
        )

    def plan_transport_posture(
        self,
        skill_input: GraspObjectInput,
        *,
        held_object,
        target_revision: str,
    ) -> Action:
        return Action.from_model(
            action_type="grasp.plan_transport_posture",
            parameters=PlanTransportPostureParameters(
                object_ref=skill_input.object_ref,
                object_pose=held_object.object_pose,
                object_extent_m=held_object.object_size_m,
                target_revision=target_revision,
                tool_refs=skill_input.tool_refs,
            ),
            timeout_seconds=6,
            label="按真实抓取关系规划携物姿态",
        )

    def prepare_transport(
        self,
        skill_input: GraspObjectInput,
        *,
        candidate: GraspCandidate,
        transport_poses,
        target_revision: str,
    ) -> Action:
        """把已抬起周转箱收拢到可转运姿态。

        transport_poses 在抓取验证后根据实时工具—箱体关系生成；Skill 不写死
        关节角，也不会把抓取前的理想seat位姿当作负载闭环目标。
        """

        return self.move_targets(
            skill_input,
            candidate,
            transport_poses,
            "transport",
            target_revision,
            0.07,
            required_contact_tools=list(skill_input.tool_refs),
        )

    def get_robot_state(self) -> Action:
        return Action.from_model(
            action_type="robot.get_state",
            parameters=GetRobotStateParameters(),
            timeout_seconds=8,
            label="读取转运姿态下的Robot状态",
        )

    def verify_tool_load(
        self,
        skill_input: GraspObjectInput,
        *,
        tool_refs: list[str] | tuple[str, ...] | None = None,
    ) -> Action:
        return Action.from_model(
            action_type="robot.verify_tool_load",
            parameters=VerifyToolLoadParameters(
                tool_refs=tuple(tool_refs or skill_input.tool_refs)
            ),
            timeout_seconds=8,
            label="连续验证转运姿态下的双工具承载",
        )


def skill_input_pose(candidate: GraspCandidate):
    """闭合动作只需语义参考位姿；使用两侧插入目标的中点。"""
    left, right = candidate.hook_insert_poses
    return left.target_pose.model_copy(
        update={
            "position_m": tuple(
                (a + b) / 2
                for a, b in zip(
                    left.target_pose.position_m,
                    right.target_pose.position_m,
                    strict=True,
                )
            )
        }
    )
