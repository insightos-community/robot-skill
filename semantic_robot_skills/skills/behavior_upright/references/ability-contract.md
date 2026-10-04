# 联合直立持物恢复

robot.get_state 读取18关节；gripper.get_state 判断持物侧及身份。
motion.move_arm_joint 使用 motion_mode=curobo 和完整关节目标，SDK/Runtime 联合规划、执行、确认并停止，保留夹爪控制。
躯干初始角为 [1.025,-1.45,-0.47,0] rad。持物臂自然屈肘关节解由既有 carry 算法在该躯干姿态下独立求得，保存在 carry_postures.json；空闲臂使用原 init 目标。
全关节实测误差不超过0.02rad且持物身份保持后完成。轨迹不进入 Skill 检查点。
