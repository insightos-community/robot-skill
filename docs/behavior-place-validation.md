> 历史 0.1.0 设计与验证记录。当前本地 0.1.1 已按 [精简记录](behavior-skills-local-simplification-20260922.md) 替换新增接口依赖。

# behavior-place 首版验收记录

日期：2026-09-22。范围：独立 Robot Skill、几何计算和待接入能力契约。

已实现：

- support_surface / top_open_container / side_open_container 三类几何流程；
- place 与显式 drop 两种释放策略，目标名称只参与引用核对；
- 旋转包围盒、开口尺寸、内部区域及抓持相对变换计算；
- 运动前路径验证、释放前实测核对、释放与撤离分别确认；
- 中断恢复、稳定 Action key、释放不确定时通过新观测确认；
- 原格式 SKILL.md、脚本、依赖锁、输入输出模型、停止入口和引用文档。

离线验收：

```bash
PYTHONPATH=. python -m pytest tests/test_behavior_place.py tests/test_skill_package_contract.py -q
```

结果：41 项通过。其中 36 项为新增放置行为测试，5 项为已有包结构/边界检查。
未添加版本号或固定配置值断言。

使用 `semantic build <behavior_place目录> --output <归档路径>` 成功生成独立 ZIP。
从 ZIP 解包后启动真实 Worker 子进程，通过 JSON-RPC 完成初始化和三类输入校验；
归档无 Python 字节码缓存，文档相对链接全部存在。

当前验证层级：离线几何 + Mock Action 执行 + 真实 Worker 加载/输入协议。
尚未执行现场运动、容器插入、释放和物理稳定性验证。

待集成能力：

| Action | 版本 | 状态 |
|---|---:|---|
| perception.resolve_placement_target | 2 | 新契约，待 Ability/SDK 接入区域和持物几何 |
| motion.validate_placement_path | 2 | 新契约，待实现持物及空手撤退路径验证 |
| perception.verify_placement | 3 | 新契约；现有版本 2 仍为占位实现 |
| motion.move_end_effector | 2 | 复用现有动作 |
| gripper.set_opening / gripper.hold_object | 2 | 复用现有动作 |

本轮未修改 SDK、Ability 或 Runtime，未发布 Registry、未下发机器人。
后续先补齐契约并核对能力匹配，再验收三类区域的实际流程。
