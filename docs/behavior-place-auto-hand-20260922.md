# behavior-place 0.1.3 自动持物侧

省略 side 或设置 auto 时，在放置流程开始读取五次双手反馈，使用按钮 Skill 的 holding_side.py 确定持物侧。显式 left/right 作为期望侧，反馈冲突时停止。采样和判据保持一致；helper 随独立包携带，运行时不依赖按钮 Skill 的目录。

hand_selection 保存到检查点，恢复时保持手侧，释放后不重采样。结果增加 holding_side 和 hand_selection。识别、几何计算、运动、释放和撤手流程保持原有行为。

无唯一侧返回 HOLDING_SIDE_UNCERTAIN；显式冲突返回 HOLDING_SIDE_MISMATCH；反馈缺失或损坏返回 PLACEMENT_FAILED。所有这些失败均发生在物体识别和运动之前。此分类判断受力候选，物体身份由后续识别检查。

## 验证

- 放置与按钮持物判定相关 76 项测试通过：覆盖自动左右侧、显式一致/冲突、双手受力/空载、不稳定、瞬时力、缺失反馈、读取失败、释放前后检查点恢复，以及原有 place/drop 几何和执行顺序。
- 既有真实双手反馈记录的 36 个五帧窗口均得到右侧受力候选，与原按钮判定一致。
- 本次没有下发物理动作，没有修改 SDK、Ability 或 Runtime。
