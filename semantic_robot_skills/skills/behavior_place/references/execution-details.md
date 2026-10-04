# 放置计算

自动持物几何：perception.locate_object 的 verification.pose 和 extent_m 属于同一个可见包围盒。
eef_from_object = inverse(body_from_eef) @ body_from_observed_box。
读取结果校验 body 坐标系、物体引用、识别置信度及正的三维尺寸，再计算放置路径。
估计值保存在检查点和执行结果，恢复时复用已完成的识别与运动。

body_from_object_goal = body_from_region @ region_from_object_goal。
body_from_eef_goal = body_from_object_goal @ inverse(eef_from_object)。

支撑面：物体最下点落在区域 Z=0，先在上方接近，再下降、释放和退回。
顶部 place：提供内部尺寸和顶部开口，检查内部区域与开口，随后从外侧进入。
顶部 drop：省略内部尺寸时使用开口平面。
省略整个 target.region 时，从 perception.locate_object 返回的 visible_upper_surface 拟合水平圆形桶沿。使用圆心和上缘高度计算开口；开口半径取多采样带拟合半径最小值，再扣除一个像素对应的空间采样尺度。
上缘点带由观测高度的 99.5% 分位及该帧像素尺度确定，Ability 最多返回 1024 点。Skill 比较 1/2/3 像素采样带：圆弧至少覆盖半圆，RMS 和中心变化不超过一个像素尺度。拟合失败时报告无法确定开口，调用方也可显式提供区域。
夹爪延伸轴选择水平或竖直向下，按当前朝向的最小旋转量决定；物体相对夹爪变换保持不变。
该姿态选择不包含离线 IK 或碰撞判断，实际移动成功后才张爪。
检查物体投影及净空能通过开口后，使物体最下点沿开口法向离平面 clearance_m，再释放。
接近点在释放点外侧增加一个物体投影厚度，释放后沿原方向撤手。
placement_pose_hint 提供朝向参考及开口平面内位置；末端朝向仍转换到水平或向下，法向高度由释放计算确定。
drop 的 target_object_body 与 release_object_body 均表示释放位姿，未推测内部落稳位置。
侧开口：在内部可用高度中间位置通过入口，下降到支撑面；释放后抬起并从入口退出。
place 检查物体包围盒是否能进入提供区域；drop 检查开口通行尺寸。路径实际碰撞由运行验证评估。

几何来源和假设保存在 target_geometry 中。容器识别只使用既有 perception.locate_object；未读取全场景真值几何。
顶部近似由 drop 模式声明触发，类别文字仅用于识别。未实现实际内缘识别和落稳确认。


## 自动判断持物侧

启动时读取五帧双手 gripper.get_state。优先使用 held_object_ref 匹配输入 object_ref；每帧均须唯一匹配同一只手。显式空值表示该手没有持物身份；身份不匹配、变化、重复或不完整时停止。只有全部反馈都缺少身份字段时，才使用原有力/开度/稳定性判定。无需新增接口。
side 默认 auto；显式手侧须与反馈一致。无唯一候选报 HOLDING_SIDE_UNCERTAIN；显式冲突报 HOLDING_SIDE_MISMATCH。
力反馈回退分支沿用原分类阈值：稳定位置范围 0.5 mm，负载力比例下四分位至少 5%，另一侧上四分位不超过 1%，平均开度距两端至少 1%。
判定保存到检查点 hand_selection，恢复时不重采样；后续物体识别、目标识别、路径、释放及撤手按原流程执行。
helper 随独立 Skill 包携带，运行时不依赖另一 Skill 的安装位置。


## 坐标、联合规划与释放确认

末端回读经实测躯干关节或底盘位姿转换为 body 后，才与识别包围盒计算抓持变换。
转换复用抓取 Skill 的 R1Pro 躯干 FK，helper 随独立 Skill 打包，无跨 Skill 运行依赖。
到达较低目标时允许 cuRobo 联合调整躯干，后续目标继续保持在底盘 body 坐标系中。
张爪开度到位后读取 gripper.get_state，确认所选手 held_object_ref 为空；有持物身份或反馈缺失时停止撤手。
release_verification 只确认张爪与原生持物身份解除；容器内落稳与任务成功仍由观测或原生评测判断。

圆形开口按持物各包围盒角点的径向距离检查，矩形开口沿用各轴边界检查。桶壁厚度未被观测时在诊断中注明；自动拟合不读取原生物体几何。
