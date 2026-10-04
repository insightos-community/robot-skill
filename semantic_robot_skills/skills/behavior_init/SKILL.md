---
name: behavior-init
description: 保持当前躯干，将双臂初始化为伸直下垂或屈肘抬起姿态
category: robot_skill
version: 0.1.6
runtime:
  api_version: 1
  python: '>=3.11'
  entrypoint: scripts.skill:run
  stop_entrypoint: scripts.skill:on_stop
  input_model: scripts.models:Input
  state_model: scripts.models:State
  result_model: scripts.models:Result
  controllers: {}
required_actions:
- type: robot.get_state
  schema_version: 2
- type: motion.move_arm_joint
  schema_version: 2
stop_actions:
- type: motion.move_arm_joint
  schema_version: 2
debug_input: {}
---

# 双臂初始化

使用 `arm_posture` 选择姿态：

- `{"arm_posture":"down"}`：每臂七个关节设为 0 rad，恢复伸直自然下垂姿态。
- `{"arm_posture":"raised"}`：沿用旧版准备姿态，双肩外展 ±0.6 rad，肘关节按当前躯干 pitch 补偿，屈肘抬起。
- `{}`：默认 `down`，兼容现有任务输入。

读取当前关节状态后，先左臂、后右臂执行选定的目标，保持当前躯干和夹爪状态。`down` 是相对当前躯干的自然下垂，不会额外调整倾斜躯干以保持世界坐标竖直。

## 执行与结果

1. `init:read-state`：读取 torso_joint1~4 和双臂关节，验证完整性，计算并校验目标关节限位，保存姿态选择和目标。
2. `init:left`：通过 `motion.move_arm_joint` 下发选定的左臂七关节目标。
3. `init:right`：左臂成功后下发选定的右臂七关节目标。

两个手臂动作均成功才完成。结果记录 arm_posture、当前躯干 pitch、关节目标、相应的 FK 末端参考位姿和证据。FK 位姿仅用于结果说明，实际动作直接执行关节目标，避免末端 IK 选择不同的肘部姿态。

## 能力与恢复

复用 `robot.get_state` 和 `motion.move_arm_joint`（schema 2），controllers 为空。关节动作使用 `motion_mode=ik` 的直接关节目标接口，沿用底层速度限制、实测到位与停止机制，每臂超时 150 秒；不运行 cuRobo 规划。适用于双手空载且运动空间无遮挡的初始化。

稳定 key 为 `init:left` / `init:right`，恢复复用检查点中的目标与已完成结果；输入姿态与已有目标不一致时拒绝恢复。失败停止后续动作并保留原错误。Pilot 停止当前 Action 并确认终态后调用 `on_stop`，返回 `pilot_confirmed_terminal`。

详见 [执行说明](references/execution-details.md)。
