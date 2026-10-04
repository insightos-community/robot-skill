# BEHAVIOR Skill 输入清理

核对六个当前 Skill 的 Input、脚本引用和相关 Ability 处理链。

- behavior-init：输入为空，无可清理参数。
- behavior-nav 0.1.7：删除 side（只回显校验，不参与站位）和 navigation_purpose（只透传日志）。
  内部保留固定 transit 用途以满足已有 Action 协议；object_name、maximum_speed_mps、minimum_clearance_m、arrival_radius_m、timeout_seconds 都有实际用途。
  minimum_clearance_m 校验请求不超过部署 navigation_margin，实际规划间距仍来自部署。
- behavior-grasp 0.1.17：现有输入用于目标几何、识别、姿态分支、偏移、夹爪控制、置信度或超时；全部保留。
  0.2 rad/s 躯干速度由 Skill 内部设置，没有新增外部 speed 输入。
- behavior-radio-button 0.1.29：object_ref/object_prompt 和旧 chest_* 已删除；现有按钮引用、视觉提示、模型、相机、操作侧、置信度、闭爪力和超时均参与执行或结果核对。
  button_ref 有默认值，可省略；wrist_model_profile 省略时按 operating_side 选择。
- behavior-upright：唯一可选输入 timeout_seconds 用于运动等待预算。
- behavior-place：所有输入均参与几何路径、夹爪释放、目标身份关联或超时；全部保留。

任务迁移：新版导航只传物体名称及可选速度/间距/半径/超时，删除 side 和 navigation_purpose；
抓取和操作的侧向参数仍需按各自 Skill 声明。

验证：导航恢复与包结构共 6 项通过。实际生成的 follow_route/verify_arrival Action 通过 Ability 输入模型校验。
独立 ZIP Worker 初始化和最小输入校验通过。未增加版本号/配置值断言测试，未执行物理动作，未同步远端。
