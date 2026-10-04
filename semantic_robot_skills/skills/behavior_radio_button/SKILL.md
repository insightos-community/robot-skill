---
name: behavior-radio-button
description: 保持躯干，持物臂屈肘后内收观察；未找到按钮时恢复屈肘、翻转并小幅后收，再识别并提交按钮末端目标
category: robot_skill
version: 0.1.37
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
- type: gripper.get_state
  schema_version: 2
- type: robot.get_state
  schema_version: 2
- type: gripper.set_opening
  schema_version: 2
- type: motion.get_torso_state
  schema_version: 2
- type: motion.get_end_effector_state
  schema_version: 2
- type: motion.move_end_effector
  schema_version: 2
- type: perception.locate_object
  schema_version: 2
- type: sensor.capture_rgbd
  schema_version: 2
stop_actions:
- type: gripper.hold_object
  schema_version: 2
debug_input: {}
---

# 收音机按钮操作

## 执行入口

scripts/skill.py:run 组织收臂、末端定向观察、按钮识别及按钮接近；on_stop 保持当前状态。

## 可执行输入

```json
{}
```

默认 operating_side=auto。启动时读取双手夹爪的力、开度和位置稳定性，推断唯一持物侧，另一侧用于操作。
最小输入为 {}。可显式提供 left/right 作为预期操作侧，必须与反馈判断一致；冲突时停止。
双侧都受力、都空闲、位置不稳定或反馈不完整时，不会下发收臂或夹爪动作。
wrist_sensor_id 省略时使用对应的 left_wrist_camera/right_wrist_camera 别名，由 Runtime 解析当前实例的真实相机。
wrist_model_profile 省略时使用 sam31-{operating_side}-wrist。也可显式提供自定义相机和识别配置；两者须匹配操作侧。
已知的相反侧相机或标准识别配置会在输入校验阶段被拒绝。
可选 button_ref、button_prompt、wrist_model_profile、minimum_confidence、maximum_force_n、timeout_seconds
保持原有语义。三个未参与执行的 chest_* 输入已删除。

## 大期望

保持当前躯干，收臂及观察位可达，按钮识别成功并完成末端接近。
结果包含 holding_side、operating_side 和 hand_selection 采样判定依据。
结果级别为 button_localized_and_approach_motion；原生任务是否成功以现场评测为准。

## Stage 与期望

1. radio:gripper-state:0~4：通过 gripper.get_state 连续读取双手，按力/力限比、开度和位置变化推断持物侧。结果保存到检查点，后续恢复沿用已选角色。
2. radio:natural:eef → radio:retract:eef：先提交自然屈肘对应的末端目标；成功后保持目标朝向与高度，向后 10 cm、向中线 10 cm，提交第二个末端目标。左右手内收方向镜像。
3. radio:inspect:1:eef：根据实测持物末端和操作腕标定计算观察目标，沿 body Z 下移 5 cm、保持原朝向，再交给 SDK 执行。
4. radio:locate:1：识别按钮；找到后计算按钮末端目标，通过 motion.move_end_effector 执行。
5. 首次未找到时恢复自然屈肘目标，提交夹爪延伸轴 180° 翻转后的最终朝向，再提交向中线 10 cm、向后 1 cm 的目标。
6. 重新计算观察目标并识别一次；识别成功则提交按钮目标，失败则报告并结束。

每个阶段完成后才进入下一阶段。收臂、翻转和观察动作使用 motion_mode=curobo、use_torso=false。
识别成功后的 radio:approach-button:eef 使用 motion_mode=ik、position_tolerance_m=0.003，直接执行按钮末端目标，不经过 cuRobo 碰撞规划；Isaac 物理碰撞仍保留。
躯干与另一只手保持当前目标。按钮动作完成后沿用原有实测位姿复核，原生任务成功与动作到位分别记录。
Skill 只确定末端目标，SDK 决定关节构型和完整路径。自然屈肘用于生成参考末端位姿，执行时的肘部构型由 SDK 求解。
翻转只约束最终位置与朝向，中间运动由 cuRobo 规划。

## Controller 与原子能力

运动学、镜头对准、视场和翻转路径均位于 Skill 专用脚本，controllers 为空。
复用已有状态、采图、识别、末端和夹爪 Action，见 [接口说明](references/ability-contract.md)。
左右观察位按操作手一侧约束镜像；数值求解和执行容差保留原有设置。

## 局部恢复与 Agent 决策

保存持物/操作侧判定、计划与已完成 Action，未找到按钮只进行一次翻转重试。末端目标 Action 失败时停止后续阶段，保留结果；已完成的 Action 恢复时不重复下发。
角色无法区分返回 HOLDING_SIDE_UNCERTAIN，显式侧冲突返回 OPERATING_SIDE_MISMATCH。
识别失败、不可达或执行失败保留错误并停止；未引入场景代际或同帧恢复协议。

## 停止

Pilot 停止当前 Action 后，调用 gripper.hold_object 保持夹爪状态，并按 holding 结果报告停止确认。
