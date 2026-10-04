# 现有能力组合

perception.locate_object(object_ref,prompt,pose_hint,minimum_confidence) 返回 body 可见物体包围盒，并在 visible_upper_surface 中提供有限数量的上缘深度点、竖直方向、高度及像素尺度。
可选 model_profile 沿用现有识别模型配置。
motion.get_end_effector_state(axis) 返回实测位姿，当前 Ability 使用 torso_link4。Skill 复用抓取的躯干 FK，根据 robot.get_state 的实测关节转换到 body；世界坐标使用实测底盘位姿转换。
motion.move_end_effector(axis,target,purpose,target_revision,motion_mode=curobo,use_torso=true) 联合躯干与手臂执行 clearance/insert/place/retreat。
gripper.set_opening(tools) 完成指定开度，Skill 要求 verified=true，再用 gripper.get_state 确认持物身份为空。
gripper.hold_object 用于停止后保持。全部使用 schema 2。

不再依赖 resolve_placement_target、validate_placement_path 或 verify_placement schema 3。
区域与开口由 Task 提供，或在顶部 drop 中由 Skill 拟合可见圆形桶沿。
robot.get_state() 提供 base_pose，用于计算 body 中的世界竖直方向。
持物尺寸和抓持关系可由识别包围盒与末端状态计算，或由 Task 成对提供。
自动识别只在规划前运行，不自动移动躯干或手臂寻找视角。识别失败会停止后续动作。
最终状态仅确认请求运动、张爪和撤离，完整放置关系需额外现场观察。

自动持物侧通过已有 `gripper.get_state`（schema_version=2，空参数读取双手）获取。
需要每侧的 positions_m、efforts_n、effort_limits_n、closed_positions_m、open_positions_m 双指反馈。
五次采样及分类规则与按钮 Skill 保持一致；歧义或显式手侧冲突在识别和运动前结束。
