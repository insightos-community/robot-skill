# 先识别再调整操作姿态：职责收敛（2026-09-22）

最终组合：behavior-grasp 0.1.15、r1pro-behavior-atomic-abilities 0.5.7。

## 最终实现

观察姿态计算 → 单次躯干调整 → 视觉识别 → 操作 IK 计算 → 单次躯干调整 →
读取实测关节校验路径 → 接近、闭爪、抬升。

observation.py、operation.py 与 planner.py 均为 Skill 专用脚本。姿态由当前关节、
目标几何、相机标定和机器人模型求解，保留已有数值容差与路径约定。
识别后保持底盘不动，直接使用识别得到的 body 坐标。

相对最初 Ability 提交 a21f1ba，仅在 CaptureRGBD 的既有返回值中补充
calibration.intrinsic_matrix、calibration.camera_pose_body，转发 SDK 原有元数据。
Task、输入模型、SDK 和 Runtime 接口保持原状。

撤回中间版本的 natural/snapshot、allow_backoff 输入，以及 Ability 中的观察优化器。
按用户最新要求移除新增同帧绑定、代际检查、目标位移恢复校验及 anchoring.py。
原有采图协议中的 generation/sequence 字段保留原样。

## 验证

- 抓取流程、失败阻断、几何、双手镜像、真实记录的操作 IK/路径与自然观察计算：81 passed。
- Ability 采图标定、原有观察计算和识别：33 passed，21 subtests passed。
- Skill 包结构与独立 Worker 输入校验通过；两种组件使用 semantic build 打包。
- 之前全量 Ability 中的停止状态断言失败，在未修改的 a21f1ba 中已复现：
  test_reset_changes_client_and_retires_pending_commands_without_replay 期望 interrupted，实际 stopped。

真实记录取自 Picking Up Trash 的关节、相机标定和视觉目标，用于离线求解。
本轮没有下发现场抓取动作，实际碰撞和抓取成功仍待完整流程执行验证。
