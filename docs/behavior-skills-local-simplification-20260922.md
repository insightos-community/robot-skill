# Robot Skills 本地精简验证

本轮仅修改本地隔离工作区，未同步、上传或部署远端。

|组件|本地版本|行为|
|---|---|---|
|behavior-init|0.1.2|从 SDK 示例准备姿态做 FK，每臂一次末端 IK；移除十段插值|
|behavior-nav|0.1.6|保留原有导航链路；配套 Ability 删除固定抓取侧 5 cm 补偿|
|behavior-radio-button|0.1.27|删除失效 chest_* 输入和固定收臂补偿；视觉持物几何加腕部标定计算观察位|
|behavior-place|0.1.1|任务提供区域、开口和抓持关系；使用现有末端、张爪、保持接口|
|r1pro-behavior-atomic-abilities|0.5.8|删除导航固定侧移与识别中固定 radio_89 真值调试查询|

behavior-upright 保留原生初始姿态和既有轨迹执行，其目标定义有模型来源。本轮复核相关测试。
SDK、Runtime、Pilot 和模型配置未修改。未新增原始场景几何返回、Task 或 SDK 接口。

## 验证

- 58 项 Skill 测试通过，覆盖初始化动作数量/失败停止、双手镜像、观察及翻转数值解、导航结果流、直立和放置几何/执行。
- 41 项 Ability 测试通过，另有 21 个子测试通过，覆盖保留导航站位及视觉识别。
- 四个新版 Skill 的独立 Worker initialize 和 validate_input 均通过。
- 根据已有现场关节记录离线求解 init 的左右末端目标，位置及朝向残差通过 1 mm / 0.5° 检查。
- 本地使用 semantic build 生成四个 Skill 包与 Ability 包。记录位于 validation/other-skills-local。

## 输入和结果变更

操作 Skill 现在需要 operating_side、object_ref、object_prompt、wrist_sensor_id。
实际观察使用当前持物的视觉包围盒与腕部标定，不再使用固定 13 cm 持物中心、观察距离/高度范围。
收臂、观察各为一次末端目标；轴向翻转继续使用已验证的连续关节目标序列。

放置 Skill 需要 object_size_m、eef_from_object、target.region（容器含 opening）等任务几何。
内部计算全部在 Skill 脚本中。移除新增的区域解析、路径验证和 schema 3 放置验证依赖，以及
场景代际、快照时效和异常释放恢复。结果 verification_level=release_and_retreat_motion，
仅确认请求运动、开度到位和撤手，尚未独立验证物体落稳或最终 on/inside 关系。

本轮验证为本地接口/数值计算验证，未执行现场初始化、抓取或放置。完整物理流程需后续部署后验证。
