---
name: behavior-nav
description: 导航到指定对象，或从剩余候选中选择路径最近的可达对象；返回选中对象和到达结果
category: robot_skill
version: 0.1.9
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
- type: navigation.follow_route
  schema_version: 2
- type: navigation.verify_arrival
  schema_version: 2
stop_actions:
- type: navigation.follow_route
  schema_version: 2
debug_input:
  object_name: radio_89
---


# 物体操作站位导航

## 执行入口

- `scripts/skill.py:run` 组织站位计算、导航与到达检查。
- 导航 Ability 读取其绑定实例的几何，通过 `object_approach.py` 计算站位。
- `scripts/models.py` 定义输入、检查点和结果，`on_stop` 确认底盘保持。
- 部署与脚本来源见 [执行说明](references/execution-details.md)。

## 可执行输入

```json
{
  "object_name": "radio_89",
  "maximum_speed_mps": 0.15,
  "minimum_clearance_m": 0.05,
  "arrival_radius_m": 0.06,
  "timeout_seconds": 180
}
```

`object_name` 为场景中的精确对象名称。示例 `radio_89` 对应当前收音机；其他场景使用其实际对象名。
Task 提供物体名称，Skill 自动产生底盘目标坐标和朝向。输入模型拒绝已删除的 `side`、`navigation_purpose`、旧 `target`、`target_ref` 及坐标字段。

| 参数 | 默认值 | 含义 |
|---|---|---|
| object_name | 与候选列表二选一 | 场景中的精确对象名称 |
| candidate_object_names | 与 object_name 二选一 | 1–20 个不重复的剩余候选对象名称 |
| maximum_speed_mps | 0.15 | 底盘最高线速度，范围 (0,0.3] m/s |
| minimum_clearance_m | 0.05 | 需小于等于部署 profile 的 navigation_margin |
| arrival_radius_m | 0.06 | 基座平面位置到达容差，米；显式提供时按输入值执行 |
| timeout_seconds | 180 | 路线执行超时预算，秒 |

## 大期望

站位计算成功、`navigation.follow_route` 成功，且 `navigation.verify_arrival` 返回
`verdict=achieved`、有限的容差内距离及 `final_pose_ref` 后，Skill 返回成功。

结果包含计算出的 odom 目标、物体和支撑物名称、地图 ID、Semantic 场景实例标识（UUID 字符串）及 run_id，
以及最终位姿引用和到达距离。独立检查覆盖位置，`yaw_verified=false`；朝向由导航后端执行。
脚本校验站位可容纳底盘；路线可达性由导航能力检查。站位计算结果尚未包含手臂 IK 检查。

## Stage 与期望

1. `nav:follow`：将物体名称编码为 `route_ref={"object_name":"radio_89"}`。Ability 读取当前碰撞几何和固定 odom 对齐信息，生成实例地图，调用 `ObjectApproachFinder` 计算站位并执行导航；结果携带计算出的 odom 目标。
2. `nav:verify`：保存目标检查点，向到达能力传入同一个 odom 目标和容差；验证通过后输出完成结果。

## Controller 与原子能力

`runtime.controllers` 为空。Skill 通过 `navigation.follow_route`、`navigation.verify_arrival`
调用导航 Ability，schema_version=2。Ability 使用绑定的 SDK 实例读取几何与定位，地图缓存位于
SDK navigation_cache_dir 下的 object-approach 子目录；站位计算模块随能力代码部署。

脚本沿用更新后的 `object_approach.py` 算法：以物体及支撑物几何选边，使用实际底盘碰撞轮廓，
站位与支撑物边缘的最小间隔为 0.32 m，轮廓检查附加余量为 0.02 m，基础朝向指向物体中心。
站位直接使用现有算法结果。站位计算失败返回 `OBJECT_APPROACH_FAILED`。

## 局部恢复与 Agent 决策

导航 Action 的结果及计算出的站位保存在检查点中，同一执行恢复时复用结果；导航与验证使用稳定 Action key。
新任务会重新读取当前几何。物体未找到或站位计算失败时返回 `OBJECT_APPROACH_FAILED`，
并附带脚本原因；到达未确认时返回 `ARRIVAL_NOT_CONFIRMED`。

`object_name` 按精确名称匹配。调用方可根据失败信息修正目标或安排后续任务。
更换仿真实例后应创建新执行。

## 停止

Pilot 先停止当前 Action 并确认终态，再调用 `on_stop`。
Skill 使用 `navigation.follow_route` 的 `reason/mode` 变体请求底盘 hold。
只有结果为 `succeeded` 且 `stop_evidence.holding=true` 才报告保持已确认；其余结果报告
`physical_state=unknown` 和 `requires_intervention=true`。

## 从剩余对象中选择（0.1.8）

候选模式要求 Ability 0.5.24 或更新版本。输入示例：

```json
{"candidate_object_names": ["can_of_soda_113", "can_of_soda_114", "can_of_soda_115"]}
```

object_name 与 candidate_object_names 必须且只能提供一个。候选模式在同一份当前场景快照上评估每个对象，跳过不可达者，按到各对象首个可行操作站位的路径长度选择最近者；相同距离按对象名排序。这不是遍历全部站位的全局最短路优化，也不检查手臂 IK。选择后实际导航并验证到达。

成功结果 approach.object_name 是本轮真正选中的对象。selection 包含 policy、selected_object_name、map_id、candidates；每个候选包含 object_name、status，以及 path_length_m 或失败原因。所有候选均不可达时返回 NO_REACHABLE_OBJECT，并列出逐个原因，不启动导航。

后续抓取步骤通过 robot.get(execution_id) 读取本轮导航成功结果，将 approach.object_name 填入 object_ref；不要根据输入候选顺序猜对象。放置使用本轮抓取成功结果的 object_ref、object_size_m、eef_from_object。只有投放成功的对象才能从下一轮候选集合中排除；仅导航或抓取成功不表示清理完成。每次新的候选导航执行都会重新读取几何与定位。剩余集合由调用方提供，本 Skill 不跨执行记忆已清理对象。
