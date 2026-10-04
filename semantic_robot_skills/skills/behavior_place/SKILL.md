---
name: behavior-place
description: 从识别或任务几何计算放置与顶部投放，执行释放和撤手
category: robot_skill
version: 0.1.7
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
- type: perception.locate_object
  schema_version: 2
- type: motion.get_end_effector_state
  schema_version: 2
- type: motion.move_end_effector
  schema_version: 2
- type: gripper.set_opening
  schema_version: 2
stop_actions:
- type: gripper.hold_object
  schema_version: 2
debug_input:
  object_ref: item
  side: left
  object_size_m:
  - 0.1
  - 0.08
  - 0.12
  eef_from_object:
    position_m:
    - 0
    - 0
    - 0.08
  target:
    object_ref: surface
    type: support_surface
    region:
      body_from_region:
        position_m:
        - 0.6
        - 0.1
        - 0.5
      extent_m:
      - 0.6
      - 0.5
      - 0
  release_opening_m: 0.05
  maximum_force_n: 20
---

# 几何区域放置

## 执行入口

scripts/skill.py:run 执行，scripts/geometry.py 计算放置路径，on_stop 保持当前姿态。

## 可执行输入

必填 object_ref、target、release_opening_m 和 maximum_force_n。
side 可省略或填 auto，默认优先根据 held_object_ref 与 object_ref 匹配确定持物侧；反馈缺少身份字段时，根据双手力、开度和位置稳定性判断；显式 left/right 作为期望侧，与反馈冲突时停止。
object_size_m 与 eef_from_object 可一起省略，此时必填 prompt；Skill 调用 perception.locate_object 获取持物包围盒，结合实际末端状态计算抓持变换。
可选 perception_profile 选择已有识别配置，minimum_confidence 默认 0.3。
eef_from_object 是物体包围盒中心坐标系相对夹爪的变换。尺寸与变换一起提供时使用任务输入。
识别尺寸来自单视角可见点云，遮挡可能低估完整物体尺寸；结果中的 object_geometry 记录来源和估计值。

```json
{
  "object_ref":"item", "side":"left", "object_size_m":[0.1,0.08,0.12],
  "eef_from_object":{"position_m":[0,0,0.08]},
  "target":{"object_ref":"surface", "type":"support_surface",
    "region":{"body_from_region":{"position_m":[0.6,0.1,0.5]},"extent_m":[0.6,0.5,0]}},
  "release_opening_m":0.05,"maximum_force_n":20
}
```

target.type 为 support_surface、top_open_container 或 side_open_container。
region 使用 body_from_region；局部 XY 为区域平面、+Z 向上。
place 模式必填 region.extent_m：Z=0 为支撑面，容器内部高度为 extent_m.z。
容器必填 opening.region_from_opening 和 opening.extent_m，opening.shape 默认为 rectangle，也可为 circle（extent_m 为直径）；开口局部 XY 为通行平面、+Z 指向外部。
顶部 drop 模式可省略 region.extent_m，只检查开口几何及物体投影，释放高度由开口平面确定。
调用方声明容器类型。顶部 drop 可省略 target.region 并提供 target.prompt；此时默认开口朝上，从识别结果中的上缘点云拟合水平圆形开口，圆弧覆盖、拟合残差和多采样带中心稳定性通过后计算投放位置。
通过 robot.get_state 获取底盘姿态，将世界竖直方向转换到 body；可选 target.perception_profile 指定容器识别配置，缺省沿用 perception_profile。
执行结果 target_geometry.source=visible_circular_rim，记录中心、半径、拟合误差和可见圆弧。按圆形边界检查持物投影。自动拟合适用于可见圆形上缘；其他形状可显式提供 target.region。桶壁厚度未被观测时会在结果中注明。

可选 placement_pose_hint 指物体中心在区域坐标系中的目标。
place 默认保持姿态并居中接触支撑面；drop 默认对准开口中心。
drop 从夹爪延伸轴水平、竖直向下两种朝向中选择旋转量较小者，保留物体相对夹爪的变换。
这一步为几何姿态选择，可达性由实际末端动作判定；失败会停止释放。
drop 提供 hint 时使用其开口平面内位置，并从 hint 对应末端朝向计算水平或向下姿态；法向释放高度由变换后的物体厚度和净空计算。
release_mode 默认为 place，顶部开口支持 drop。clearance_m 由 Task 按需指定，默认 0；
timeout_seconds 默认 180。识别后的目标及底盘在本次操作期间保持稳定。

顶部投放可按下列结构提供开口几何（数值仅用于说明格式）：

```json
{
  "object_ref": "item", "side": "left", "object_size_m": [0.06, 0.06, 0.12],
  "eef_from_object": {"position_m": [0, 0, 0.06]},
  "target": {
    "object_ref": "container", "type": "top_open_container",
    "region": {
      "body_from_region": {"position_m": [0.6, 0.1, 0.7]},
      "opening": {
        "region_from_opening": {"position_m": [0, 0, 0]},
        "extent_m": [0.3, 0.25]
      }
    }
  },
  "release_mode": "drop", "clearance_m": 0.02,
  "release_opening_m": 0.05, "maximum_force_n": 20
}
```

这里区域原点与开口中心重合，无需指定底部位置或内部深度。
自动识别持物时，可删除示例中的 object_size_m 和 eef_from_object，增加描述实际物体的 prompt。
完全使用识别估计的顶部投放输入示例：

```json
{
  "object_ref": "item", "prompt": "soda can",
  "target": {"object_ref": "container", "prompt": "open container", "type": "top_open_container"},
  "release_mode": "drop", "release_opening_m": 0.05, "maximum_force_n": 20
}
```

识别发生在放置 Skill 开始、机器人到达投放位置之后。识别不足时停止后续动作，不自动调整观察姿态。

## 大期望

路径几何有效，所有末端运动成功，张爪输出 verified=true 且持物身份已清空，随后完成撤手。release_verification 保存释放确认结果。
verification_level=release_and_retreat_motion，只报告释放和撤手动作完成；未独立验证物体落稳或 on/inside 关系。

## Stage 与期望

0. place:gripper-state:0~4 读取双手反馈，确定唯一持物侧并保存；判断不明确或与显式 side 冲突时停止。
1. place:read-eef 读取末端位姿；若返回 torso_link4 或世界坐标，使用 place:read-body 的实测关节/底盘姿态转换到 body；缺少持物几何时 place:locate-held 识别物体并计算抓持变换。
2. 省略目标区域时，place:locate-target 识别容器，place:read-body 获取底盘姿态，拟合可见圆形桶沿，计算开口中心与高度。
3. 生成放置路径；drop 调整夹爪为水平或竖直向下。
4. place:preplace / place:enter / place:seat：按支撑面或开口类型依次执行接近、进入和放置。
5. place:release：使用既有张爪接口并检查开度到位；place:verify-release 确认持物手 held_object_ref 为空，确认释放后才撤手。
6. place:retreat*：按相反方向撤离。drop 在开口上方释放后撤离。

接近距离由物体投影厚度及指定净空计算；侧入口进入高度取当前可用垂直余量中间位置。

## Controller 与原子能力

本分支通过 motion.move_end_effector 使用 cuRobo 联合规划躯干和持物臂，Skill 只提交 body 坐标系中的末端位置和朝向。携物接近包含 attached_object_ref，释放确认后的撤手使用空手规划。底盘保持不动。


所有区域、开口、物体到末端的计算在 Skill 完成，controllers 为空。
复用 gripper.get_state、robot.get_state、perception.locate_object、motion.get_end_effector_state、motion.move_end_effector、gripper.set_opening 和 gripper.hold_object。
原草案三个新接口依赖已移除；复用既有 SDK/Ability 动作，识别结果增加 visible_upper_surface 点云字段，详见 [接口说明](references/ability-contract.md)。

## 局部恢复与 Agent 决策

保存已判断的持物侧、计划与已完成 Action；恢复执行时沿用已判断手侧，释放后不重新选手。失败停止，保留原错误。已完成动作按稳定 key 复用。
已移除代际、快照时效、计划摘要和重新对齐流程。

## 停止

Pilot 停止当前 Action 后，gripper.hold_object 保留当时夹爪状态。停止结果由 holding 字段确认。


持物侧判定复用 behavior-radio-button 的 holding_side.py，随本包独立打包。
连续读取五次双手反馈，间隔 0.2 秒，优先匹配 held_object_ref；只有所有反馈均缺少身份字段时才使用力和开度判定。身份歧义或反馈不完整时停止。
结果 holding_side、hand_selection 记录选择及其反馈依据；无需填写相机 ID。
