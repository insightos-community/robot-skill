# Semantic Robot Skills

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库包含正式 `semantic_robot_skill_sdk`、隔离 Python Worker，以及三个可恢复
Robot Skill：`semantic-navigation`、`grasp-object`、`place-object`。

每个 Skill 只保留 `SKILL.md`、`scripts/`、`references/`、`tests/`（发布包可省略）和
`requirements.lock`。Pilot 只扫描 `SKILL.md` frontmatter；仓库不再维护
`robot-skill.yaml` 或独立 Skill Library。

Skill 负责固定 Stage、期望判断、局部恢复和类型化 Agent 请求。它只能产生声明过的
Action，不能导入 Robot SDK、ROS、MuJoCo、Isaac 或模型库。Pilot 用
`python -m semantic_robot_skill_sdk.worker --skill-dir <dir>` 启动隔离 Worker，stdin/stdout
只传 JSON-RPC 2.0；图像、点云和视频只传 Artifact 引用。

当前测试使用 `MockSkillContext`，覆盖稳定 Action key、Feedback 游标、checkpoint、局部
恢复、停止和三 Skill 拆码垛闭环。真机和仿真后端未接入时由 Pilot/Ability 明确失败。

## 工程结构

- `semantic_robot_skill_sdk/`：Worker 协议、执行上下文与 SDK。
- `semantic_robot_skills/skills/`：导航、抓取、放置三个 Skill 包。
- `tests/`：Worker / SDK 测试；各 Skill 另有自己的测试。

## 构建与安装

Robot Skill Runtime SDK 以 Wheel 安装到 Pilot 创建的 Skill 环境。三个具体 Skill
分别打包为包含 `SKILL.md` 和脚本的 Zip，由 Semantic Server Robot Skill Registry
保存和下发；它们不随 R1 Pro 机器人类型包预装。

需要 Python **3.11+** 和 uv；quick-start 使用 **3.13**。

```bash
uv venv --python 3.13
uv pip install -e . pytest
PATH="$PWD/.venv/bin:$PATH" make test
uv build --wheel
```

也可以使用标准构建命令：

```bash
python -m build --wheel
make test
```

构建产物名为 `semantic_robot_skill_sdk-<version>-py3-none-any.whl`，Wheel 只包含
`semantic_robot_skill_sdk`。仓库中的三个具体 Skill 不进入这个 Wheel，避免绕过
Server Registry 的安装、启用和版本选择。

正式安装流程为：

```text
Server 发布 Skill 包
→ 选择 Robot 与精确版本
→ Pilot 下载到 staging
→ 校验 SKILL.md、入口、required_actions 和 requirements.lock
→ 创建独立环境并安装 Skill SDK Wheel
→ 原子切换 active 版本
```

同型号 Robot 可以安装不同的 Skill 版本。升级 Robot SDK、Ability 或 Pilot 不会隐式
替换 Robot Skill；正在执行的 Worker 始终使用启动时固定的 Skill 目录和环境。

## 本地调试

Worker 入口只用于 Pilot 或 Runtime 测试：

```bash
python -m semantic_robot_skill_sdk.worker --skill-dir \
  semantic_robot_skills/skills/grasp_object
```

stdin/stdout 专用于 JSON-RPC。直接运行 Worker 不会连接 Robot，也不能绕过 Pilot
调用 Ability。完整流程验证必须由 Server 下发 Skill，并通过 Pilot 的 Action 路由调用
当前 Robot 的精确 Ability 实例。

## 常见问题

“Robot Skill 尚未安装”表示注册表缺少指定包或版本。仅构建 SDK Wheel 或激活 Robot
Bundle 无法解决，需要发布请求的精确 Skill 版本后重试。

从 Fake / 仿真测试开始；连接硬件后 Skill 可能产生实际运动，只能在受控环境执行。

## 相关文档

[详细技术参考](README.reference.md) · [Skill 工程](semantic_robot_skills/skills/)

## 许可证

Copyright 2026 InsightOS。自有代码采用 [Apache-2.0](LICENSE)；第三方组件与资产请查看
[NOTICE](NOTICE) 和[许可范围](LICENSE_SCOPE.md)。

## 三个平台的构建复现

参见 [glibc、musl 与 macOS 构建说明](README.build.md)：包含已锁定的源码版本、实际脚本
入口、工具要求、本地与 CI 指令、产物位置和平台验证范围。
