# 导航执行

0.1.9 将最终到位验收 arrival_radius_m 的默认值从 0.05 m 增至 0.06 m。该值同时传入 navigation.verify_arrival 并用于 Skill 的距离复核；显式输入仍按指定值执行。不改变站位、路线控制停止判据、避障间距或朝向检查。

继续使用现有 ObjectApproachFinder 与 SDK 导航链路，真实几何、底盘占位和路线检查沿用原实现。
导航目标与到达检查使用同一计算结果。物体目标不按抓取侧偏移，Skill 不接收 side。

navigation_purpose 在当前 Ability 中只写入结果元数据，不影响目标、路径或到达检查。
Skill 不再暴露此输入；为了满足已有 FollowRoute/VerifyArrival 协议，内部统一传 transit。
minimum_clearance_m 仍检查部署 profile 的 navigation_margin，因此保留；它不能按调用修改地图的实际安全边距。
maximum_speed_mps、arrival_radius_m、timeout_seconds 分别控制底盘速度、到达判据和 Action 等待预算。

未修改 Ability/SDK/Runtime 接口。现有算法的支撑物间隔和碰撞余量保持原设置。
本版 behavior-nav 0.1.7 要求已去掉抓取侧固定偏移的 Ability 0.5.8；当前远端符合该条件。
已有任务调用新版时，应删除导航输入中的 side 和 navigation_purpose。

0.1.8 增加 candidate_object_names，与单目标 object_name 互斥；要求 Ability 0.5.24 的多目标选择。结果中的 approach.object_name 供后续 Agent 读取。Framework 未增加对象绑定或跨步骤强制校验。
