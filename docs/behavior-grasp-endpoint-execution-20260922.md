# 抓取改为阶段终点执行

behavior-grasp 0.1.16 保留原有镜头平面采样和关节连续性检查，实际执行改为四个末端目标：
clearance → pregrasp → approach → 闭爪确认 → lift。
每个阶段使用一次 motion.move_end_effector。采样结果保留在诊断记录中，供离线分析。

目标位姿仍使用原求解结果，闭爪与接触确认顺序保持原样；输入参数无需变化。
Runtime 使用自己的 IK 与插值生成实际运动。本地采样可达性不等同于实际插值或碰撞验证。

验证覆盖左右臂的真实记录几何、采样约束、四个阶段目标、动作顺序、失败停止及恢复。
独立 Worker initialize / validate_input 通过。仅更新 Skill，SDK/Ability/Runtime 接口保持原样。
本轮不下发抓取动作，现场接近路径与抓取结果待后续执行验证。
