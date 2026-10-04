---
name: behavior-upright
description: 联合恢复 R1Pro 躯干与双臂到固定直立持物姿态，确认到位及持物保持
category: robot_skill
version: 0.1.1
runtime:
  api_version: 1
  python: ">=3.11"
  entrypoint: scripts.skill:run
  stop_entrypoint: scripts.skill:on_stop
  input_model: scripts.models:Input
  state_model: scripts.models:State
  result_model: scripts.models:Result
  controllers: {}
required_actions:
  - { type: robot.get_state, schema_version: 2 }
  - { type: gripper.get_state, schema_version: 2 }
  - { type: motion.move_arm_joint, schema_version: 2 }
stop_actions:
  - { type: gripper.hold_object, schema_version: 2 }
debug_input: {}
---

# 恢复直立持物姿态

## 执行入口

scripts/skill.py:run；scripts/geometry.py 生成固定关节目标。适用于 ISAAC/BEHAVIOR 部署。

## 可执行输入

```json
{}
```

可选 timeout_seconds 为单次动作墙钟预算，默认900秒，包含 cuRobo 规划。
双手原生持物身份确定持物侧。单手持物时，躯干回原生初始角、持物臂回自然屈肘位、空闲臂回 init；无持物时双臂均回 init。双手持物时停止并要求明确目标。
固定目标随包保存；自然屈肘关节解由现有 R1Pro 模型计算，左右手分别求解。夹爪开度和夹持力不下发修改。

## 大期望

躯干及双臂18关节到达目标，最大实测角误差不超过0.02rad；运动后持物侧及物体身份保持。底盘不移动。

## Stage 与期望

1. upright:read-holding、upright:read：读取持物反馈与关节状态，保存固定目标。
2. upright:restore：一次提交关节目标，由 SDK/cuRobo 联合避障规划并执行；全部关节已到位时跳过。
3. upright:verify、upright:verify-holding：验证全部关节及持物身份。

## Controller 与原子能力

controllers 为空。使用已有 motion.move_arm_joint，motion_mode=curobo；Skill 不生成或传输轨迹。SDK 使用关节空间规划、当前场景障碍与原生持物附着模型，Runtime 负责完整轨迹执行及停止。普通 joint target 调用仍沿用原实现。

## 局部恢复与 Agent 决策

保存关节目标、持物身份与 Action 结果，恢复执行时复用稳定 key。失败即停止后续步骤，不自动重试；结果不代表原生任务成功。

## 停止

Pilot 先确认当前能力动作停止，再调用 on_stop；gripper.hold_object 保持当前关节与夹爪状态并确认停止。
