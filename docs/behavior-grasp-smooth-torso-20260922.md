# 抓取躯干改为平滑轨迹

behavior-grasp 0.1.18 的观察与操作躯干调整都使用既有 trajectory_duration_seconds，
取代 0.1.17 的 speeds_rad_s 物理关节限速。每阶段只提交一个动作，SDK 内部产生共享进度的五次轨迹。
目标关节速度峰值上限 0.2 rad/s、加速度峰值上限 1.2 rad/s²；时长由当前和目标的最大角位移计算。
保留末端到位窗口、30 秒轨迹预算和用户的现实 Action 超时；状态检查点保存计算后的时长。
未修改姿态求解、SDK、Ability 或 Runtime。序列和原子动作继续遵循已有控制权互斥。

验证：原有流程/恢复/失败停止/包结构 23 项通过。两种躯干 Action 参数经现有 Ability 模型验证。
通过真实 SDK 的 torso_trajectory.plan 生成最近失败案例的采样，分别验证 30 Hz 与 60 Hz 下的同步进度、
到位点、有限差分速度和加速度。第一段约 12.474 秒，第二段约 16.022 秒仿真时间；实际目标速度峰值小于 0.2 rad/s。
包含目标已到位情况。证据位于 validation/grasp-smooth-torso/verification.json。
本次验证不含物理执行，不承诺实际关节完全跟随；只在本地修改和打包，未同步远端。
