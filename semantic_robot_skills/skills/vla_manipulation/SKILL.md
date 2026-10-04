---
name: vla-manipulation
description: 使用当前机器人已绑定的 VLA 操作指定物体或完成原生场景任务，独立验收并保留图像
category: robot_skill
version: 0.1.10
runtime:
  api_version: 1
  python: ">=3.11"
  entrypoint: scripts.skill:run
  stop_entrypoint: scripts.skill:on_stop
  input_model: scripts.models:ManipulationInput
  state_model: scripts.models:ManipulationState
  result_model: scripts.models:ManipulationResult
  controllers: {}
required_actions:
  - { type: vla.get_model_binding, schema_version: 2 }
  - { type: vla.observe_environment, schema_version: 2 }
  - { type: vla.execute_policy, schema_version: 2 }
  - { type: vla.set_execution_limit, schema_version: 2 }
stop_actions:
  - { type: vla.hold_robot, schema_version: 2 }
debug_input:
  objective: native_task
---

# VLA 操作

## 执行输入

通过 `robot.get(skill_name="vla-manipulation")` 读取本契约和已安装版本，
再使用 `robot.run` 启动。输入对象包含以下字段：

| 字段 | 类型与默认值 | 含义 |
| --- | --- | --- |
| objective | 必填字符串，`grasp` 或 `native_task` | 只抓取，或完成当前原生任务 |
| instruction | 必填非空字符串 | 交给当前绑定模型的原生任务指令，保持官方语言和目标范围 |
| target_source_id | 可选字符串，默认 null；grasp 时必填 | 当前地图中指定抓取物体的 source_id |
| max_actions | 正整数，默认 400 | 本次执行的动作预算 |
| timeout_seconds | 正数，默认 300 | 本次策略执行的最长墙钟时间（秒） |

例如，完成当前黑碗场景的原生任务：

```json
{"objective":"native_task","instruction":"pick up the black bowl between the plate and the ramekin and place it on the plate","max_actions":400,"timeout_seconds":300}
```

`robot.run` 返回 accepted 和 execution_id 后，本轮启动请求结束；阶段、图像和
验收结果继续在执行面板更新。动作预算进度与任务成功分别判断。
调试时从当前场景及模型绑定读取任务指令后填写 `instruction`；通用 Skill 不预填
某个 LIBERO 任务，避免在 BEHAVIOR 或其他任务中误用示例指令。

## 适用范围与验收

仅选择声明此 Skill 的机器人。模型、相机、控制器由机器人型号 Ability 的部署
绑定确定，Agent 不需要搜索模型目录或生成模型动作。Franka/LIBERO 和
R1Pro/BEHAVIOR 使用各自的型号 Ability；是否可执行以当前绑定和就绪状态为准。
R1Pro 的首个 π0.5 radio 绑定仅开放 `native_task`，使用绑定声明的官方完整指令：
`Turn on the radio receiver that's on the table in the living room.`
该绑定的指定物体抓取验收尚未开放，也不表示它能够完成其他已导入场景。

`objective=grasp` 必须给出地图目标的 `target_source_id`。只要求抓取时不能改成
完整任务；以双指接触、目标抬升及相对手部稳定作为抓取目标的验收依据。
`objective=native_task` 使用用户选择的原生任务指令，结果由上游评测器判断。

两种模式均在首次观测到目标成立后，允许同一 VLA 策略继续最多 20 个模型动作，
随后按原目标独立复核。尾段包含在 `max_actions` 和 `timeout_seconds` 内，不延长
总预算，不自动重试；用户停止和 reset 随时生效。只抓取模式若在尾段把物品放下，
最终抓取验收仍会失败，不能用“曾经抓住”替代最后仍稳定持物。

尾段按首次成功反馈中已确认完成的动作数设置截止点，不按消息条数、帧数或等待
秒数计数。记录 `first_success_actions`、`post_success_actions` 和
`post_success_complete`；首次成功发生在预算末尾时，尾段可能不足 20 步。
此版本需要 Ability 提供 `vla.set_execution_limit`，用于收紧原执行预算。

阶段：`validate_target` 确认目标 → `prepare` 核对机器人模型绑定 →
`execute_policy` 执行并检查目标 → `verify_result` 独立验收、保存图像。

控制执行成功不等于任务成功。模型错误、目标未达成和停止未确认直接形成
明确问题，不自动循环请求 Agent 重试。reset 后旧 generation 的执行和验收失效。

各型号的任务成功率以真实执行记录为准；安装或动作执行成功不能替代任务验收。
