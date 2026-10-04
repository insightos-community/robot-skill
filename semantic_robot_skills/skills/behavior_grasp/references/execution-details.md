# 抓取执行说明

接近几何见 scripts/approach.py；持物目标见 scripts/carry.py。两个计算均属于 Skill 专用几何，不提供新的远程动作。

motion.move_end_effector 增加可选 contact_object_ref，必须同时提供 approach_offset。它限定手指接触目标；attached_object_ref 用于闭爪后的携物规划。
SDK 对推进路径分别检查“手指关闭碰撞、目标保留”和“手指开启碰撞、目标移除”，组合得到仅手指与目标豁免。
SDK 重定时使用现有 0.4 rad/s 躯干、0.5 rad/s 手臂控制速度，以及 1.2 rad/s² 加速度；这些是控制限制，距离与目标不使用固定经验偏移。
实际路径再次经碰撞检查，Runtime 在轨迹执行结束后继续等待末端稳定。

预备段默认传入 pregrasp_planner=ik_filter：每次候选检查都从当前实测关节和场景快照调用 IK 求解，关闭该求解器的碰撞代价与约束以生成至多 8 个解，然后恢复所有原有检查。候选按预计关节运动时间排序，生成 32 节点关节直连路径并按现有限速重定时；cuRobo 对整段路径进行环境与自碰撞复核，采样间隔不超过任一关节 0.5°。不加载离线测试轨迹。全部候选被拒绝时回退到原 cuRobo 预备段规划；接触推进仍使用原有定姿规划及选择性接触检查。两段均通过才执行，同一次 plan_only 生成的完整轨迹沿用已有起点校验后复用。

动作反馈 pregrasp_planning 记录 requested、used、ik_s、check_s、候选数量及 selected_candidate；回退时另含 fallback_reason。显式设置 pregrasp_planner=curobo 可恢复原规划。新参数需要新版 Ability/SDK/Runtime；不要在旧 Runtime 上启用此 Skill。碰撞检查清空慢速自碰撞内核的结果缓冲区，避免上一批候选的碰撞结果污染下一批；这不会减少碰撞模型或关闭路径检查。

旧 pregrasp_offset_m、lift_offset_m 已移除。持物结果通过 carry_pose 返回。

仅回退或 carry_mode=lift_first 时，由 scripts/liftoff.py 下发离地目标：基于实测末端转换到 world/odom，沿该坐标系 +Z 增加输入 liftoff_height_m（默认 0.01 m，有限正数），朝向保持。普通 IK 只移动抓取臂。本次抬升通过 position_tolerance_m=0.003 请求 Runtime 的 3 mm 到位容差；Runtime 返回动作成功后直接进入 cuRobo 携物规划，不再执行 grasp:liftoff-after-state 或 grasp:liftoff-after-eef 二次测量。Runtime 的到位、稳定性及超时判据不变，失败、停止或中断仍由 Skill 原样传播，无自动重试。此短段不经过 cuRobo 避障，适用于已确认闭爪后的竖直离地。增大高度不保证 IK 收敛。

检查点保存 liftoff_target、liftoff_height_m、liftoff_verification 和原有 Action 结果；恢复时复用保存的高度与绝对目标，不从当前位置重新叠加输入高度，已成功的 Action 不重放。旧检查点缺少 liftoff_height_m 时按原固定值 0.01 m 解释。新的 liftoff_verification 标记 source=runtime_action_result，包含 action_key、action_status、requested_lift_m、position_tolerance_m 与 confirmed；不再生成 measured_pose、target_error_m、orientation_error_rad 或 lift_world_z_m，避免把请求高度当成实测结果。历史检查点中的旧测量字段仍可读取。

自动观察姿态保持现有关节限位的 5° 余量；膝部朝前边界采用实测验证过的 8° 余量：torso_joint1 ≥ 8°，避免自然观察优化选中 0 rad 边界。姿态评分的归一化尺度保持原值，其余躯干关节继续联合求解。目标完整入镜、躯干几何与自碰撞约束继续参与求解；此几何余量不构成力矩可行性保证。

0.1.47 的自动观察姿态要求躯干自碰撞球模型在已有 buffer 之外再保留 0.01 m 间距，避免下蹲目标落在 cuRobo 起始自碰撞边界。求解数值可行性容差仍为 1e-6。observation_diagnostics 记录 torso_clearance_required_m 和 torso_clearance_min_m；这些是计算目标的间距，不是到位后的实测验证。显式关节角、offset 和不调整躯干的模式不受此约束影响。没有可行观察姿态时按原流程失败，不下发该躯干动作。


直接携物由 scripts/carry_motion.py 组织，默认 carry_mode=direct。先保存原有闭爪后几何，使用 motion.move_end_effector 的 allow_support_contact=true、attached_object_ref 和 plan_only=true 做无运动规划。SDK 根据当前支撑接触与水平面几何生成临时接触例外；在线 IK 候选经过关节直连、重定时、互补碰撞检查与离开支撑面的方向检查。不允许进一步压入超过 0.1 mm 的数值容差，首次恢复完整碰撞可行后禁止再进入碰撞。PhysX 物理碰撞始终开启。

携物 Skill 不调用 gripper.get_state 或 grasp:verify-holding。原有闭爪 Action 的接触完成条件及 SDK 附着模型校验保留；不新增持有确认动作。抓持几何仍来自闭合后的实测末端与先前识别结果，不能把它当作携物后再次测量或夹持验证。

仅 plan_only 的 support_departure_unavailable、collision_plan_failed、invalid_start_state 失败允许选择 lift_first；选择和 carry_fallback_reason 先写检查点再执行抬升。执行失败、停止、中断、超时等原样返回。已有 liftoff_target 或旧 carry 记录的历史检查点继续 lift_first，不因新默认值重复进入直接携物。策略及 allow_support_contact 纳入轨迹缓存匹配。新版需 Ability 0.5.22 和同步更新的 Runtime/SDK。
