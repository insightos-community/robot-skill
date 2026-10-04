# 末端目标接口

复用 motion.move_end_effector：axis、target（body 位置和 xyzw 朝向）、purpose、target_revision，
收臂、翻转和观察动作传 motion_mode=curobo、use_torso=false，由 SDK/Runtime 规划和执行完整路径。
识别后的 radio:approach-button:eef 传 motion_mode=ik、position_tolerance_m=0.003，直接执行单臂末端目标，不附加 cuRobo 专用参数。
Skill 的 MotionPlan 保存 side、pose、diagnostics。末端反馈通过 motion.get_end_effector_state 核对。
任一末端 Action 失败后停止后续步骤；完成记录用于恢复。

收臂分两阶段：自然屈肘参考末端目标，然后向后 10 cm、向中线 10 cm。
首次识别失败后恢复屈肘，提交沿末端延伸轴翻转 180° 的目标朝向，再向中线 10 cm、向后 1 cm。
翻转过程由 cuRobo 规划，完整路径的可行性由执行端决定。
操作手观察目标沿 body Z 下移 5 cm，保持目标朝向；重新检查目标视场。
按钮识别成功后计算最终末端位置与朝向，下发一次 IK 末端动作；目标生成、完成后的位姿复核和检查点恢复语义不变。
夹爪保持、双手反馈、相机和感知接口沿用现有协议。
