---
name: behavior-grasp
description: 识别、抓取并携物，返回尺寸和抓持变换，供后续放置复用
category: robot_skill
version: 0.1.47
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
  - { type: motion.compute_observation_pose, schema_version: 2 }
  - { type: sensor.capture_rgbd, schema_version: 2 }
  - { type: motion.set_torso_state, schema_version: 2 }
  - { type: motion.get_end_effector_state, schema_version: 2 }
  - { type: motion.move_end_effector, schema_version: 2 }
  - { type: perception.locate_object, schema_version: 2 }
  - { type: gripper.set_opening, schema_version: 2 }
  - { type: gripper.close, schema_version: 2 }
stop_actions:
  - { type: gripper.hold_object, schema_version: 2 }
debug_input:
  object_ref: radio_89
  prompt: red portable radio
  side: left
  approach_preference: auto
  grasp_offset_m: [0.0, 0.0, 0.02]
  opening_m: 0.03
  maximum_force_n: 20.0
---

# 参数化抓取与携物

## 执行入口

`scripts/skill.py:run` 组织阶段，`on_stop` 保持当前姿态与夹持。
输入、检查点和结果由 `scripts/models.py` 定义。

## 可执行输入

```json
{"object_ref":"can_of_soda_115","prompt":"soda can","side":"auto",
 "grasp_offset_m":[0,0,0],"opening_m":0.05,"maximum_force_n":50,
 "liftoff_height_m":0.01}
```

side 默认 auto，可指定 left/right。grasp_offset_m 在自动朝向模式下使用物体局部坐标。
opening_m 为单指行程，maximum_force_n 为夹持力上限。
carry_mode 默认 direct：优先尝试允许原支撑面初始接触的直接携物；只规划失败且没有移动时回退到抬升后携物。可设 lift_first 使用原流程。
liftoff_height_m 为回退或 lift_first 分支沿 world/odom +Z 的抬升高度（米），必须为有限正数，默认 0.01。
例如 0.03 表示抬升 3 cm；仅改变离地段，不改变后续携物目标或 3 mm 到位容差。
approach_preference 可选 auto/downward/horizontal，用于候选排序；开合方向保持垂直于识别主轴。
orientation_xyzw 可显式指定符合开合方向要求的朝向；省略时自动搜索。
torso 默认 auto，可指定 positions_rad、offset_m，或 null 保持当前观察姿态。
operation_torso 默认 auto，允许 SDK 联合移动躯干与抓取手；null 固定躯干。
pregrasp_planner 默认 ik_filter，在线求 IK 并用 cuRobo 检查预备段路径；可显式设为 curobo 使用原规划。需要支持此参数的新版 Ability、SDK 和 Runtime。
minimum_confidence 默认 0.3，timeout_seconds 默认 900，为每个 Action 的墙钟预算。
perception_profile 可选择感知配置。接近距离与持物目标由内部几何计算，无 pregrasp_offset_m 或 lift_offset_m 输入。

## 大期望

目标识别成功，完整接近路径可行，闭爪动作完成，携物到位。携物后不再读取夹爪状态确认持有。
结果返回 grasp_side、grasp_pose、carry_pose、carry_mode、carry_fallback_reason、liftoff_verification、候选与包围证据。直接携物时 liftoff_verification 为空。
verification_level=contact_and_carry_motion；原生任务成功由任务评测另行确认。

## 抓取结果传给放置

成功结果额外返回 `object_size_m`、`eef_from_object` 和 `held_geometry` 来源说明。
通过 `robot.get(execution_id=抓取执行ID)` 读取该次成功执行的 `execution.result`；
把 `object_ref`、`object_size_m`、`eef_from_object` 原样传给 `behavior-place`，
并将 `grasp_side` 传为 `side`。目标容器、释放方式和夹爪参数仍按放置契约填写。
不要用 `grasp_pose` 或 `carry_pose` 代替 `eef_from_object`，也不要让模型重新计算四元数。
两个几何字段必须同时非空，且与当前仍持有的同一物体、同一次抓取对应。
放置使用这两个字段会跳过持物重新识别；没有提供目标区域时，仍识别垃圾桶并拟合开口。

`object_size_m` 是识别 OBB 的三轴尺寸（米），`eef_from_object` 把该 OBB 坐标系
映射到实际夹爪坐标系，含 `position_m` 和 `orientation_xyzw`（x/y/z/w）。
尺寸与朝向必须成对复用，不可替换为模型资产坐标轴下的尺寸。
识别前后读取底盘位姿，在 RGBD 时间插值，仅允许静止底盘的微小漂移
（端点差不超过 1 mm、0.1°）；随后结合闭合后、抬升前的实测末端位姿计算并保存变换。
识别后底盘位姿的变化通过固定坐标系变换处理；计算使用实测末端位姿。

这是基于抓取前可见点云的估计：假设识别至闭合期间物体没有被推动，携物过程中没有滑移。
它不证明完整物体尺寸或夹持内的实时物体位姿。重抓、滑移、释放或场景重置后不可复用。
旧版本检查点缺少识别锚定或抓持几何时，返回的两个字段为 null，不能用零变换补齐；
此时继续使用放置 Skill 的识别路径，或重新抓取获得新结果。

## Stage 与期望

1. 保存观察前的躯干参考姿态，调整观察姿态并识别目标。
2. 张开候选手，依据物体包围盒和主轴生成抓取候选并检查夹爪包围关系。
3. 沿夹爪伸出方向反向计算接近点，距离由识别包围盒与张开手指碰撞包络确定。
4. 默认在线调用 IK 接口生成最多 8 个预备姿态解，按预计关节移动时间排序；从当前姿态生成对应运动路径，并由 cuRobo 检查整段环境碰撞与自碰撞（关节采样间隔不超过 0.5°）。全部被拒绝时回退到原 cuRobo 路径规划；回退仍失败时才换下一个抓取候选。最后的定姿直线推进仍用现有 cuRobo 接触规划；两段均通过后才执行。同次调用可复用已检查轨迹，不读取历史测试轨迹。
5. 闭爪并核对 object_ref、candidate_id。原生持物身份可提供夹持证据，力反馈分支保留。
6. 闭爪后保留原有底盘与末端位姿读数，用于保存抓持变换及回退目标。默认调用只规划的直接携物请求；Runtime 根据当前接触及几何确定支撑面，在线求 8 个 IK 候选，复核整段路径。只允许持物与原支撑面初始接触，不允许继续下压；离开后所有样本必须通过完整碰撞检查。PhysX 碰撞不变。
7. 直接携物规划成功后执行同一轨迹；无支撑面或无可行路径时，先按保存的 liftoff_height_m 执行普通 IK 抬升，再用原 cuRobo 携物。停止、中断、规划超时或执行中失败不触发抬升回退。携物目标仍使用观察前参考躯干对应的自然屈肘位姿。
8. Runtime 等待实测到位和稳定。Skill 不下发 grasp:verify-holding，不增加携物前后的持有确认动作。闭爪动作自身原有的完成条件沿用。

## Controller 与原子能力

Skill 只提供末端目标、接近偏移、允许接触的目标引用及附着物引用。
回退或 lift_first 分支的离地动作使用 motion.move_end_effector 的 ik 模式，目标来自实测位姿；该短段沿用普通 IK 控制，不提供 cuRobo 全路径避障。增大 liftoff_height_m 不等于保证可达或通过碰撞检查。接近与收臂使用 curobo 模式；轨迹规划、选择性碰撞复核与重定时在 SDK 层。
只有指定手指可以接触指定目标，手掌和手臂继续避开目标，手指继续避开其他障碍和身体。
Runtime 负责仿真步进、执行与实际到位反馈。runtime.controllers 为空。

## 局部恢复与 Agent 决策

仅无运动的数值规划失败允许比较下一候选。运动、夹持、携物失败后停止后续阶段并保留夹持。
检查点保存目标、识别与紧凑结果，关节轨迹留在 SDK，不写入 Worker 检查点。
当前持物复核使用 Isaac 的原生持物身份，物理夹持模式的终态验证需另行验证。

## 停止

Pilot 停止当前 Action；on_stop 调用 gripper.hold_object，保留夹爪状态。
