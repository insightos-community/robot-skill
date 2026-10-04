# 初始化能力调用

`robot.get_state` / schema 2 提供关节观测。
`motion.move_arm_joint` / schema 2 接收 `arm_side`、`motion_mode=ik`、`joint_names` 和 `positions_rad`。每个请求只包含对应手臂七个关节，不修改躯干或夹爪。
Skill 仅调用这两个已有 Action，未扩展 SDK 或 Ability 接口。
