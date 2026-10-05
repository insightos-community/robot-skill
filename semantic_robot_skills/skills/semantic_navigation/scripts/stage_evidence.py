# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""低频 Stage RGB 证据；随每个独立 Skill ZIP 分发，不接触运动控制。

三个 Skill 内的本文件保持相同（包契约测试校验）。R1 Pro chassis/tote_gripper
Robot Profile 都声明 sensors.camera=camera，Runtime 暴露 camera.rgb；
这是 Robot 型号契约，不是 layout 相机。SkillContext 没有传感器目录接口。
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from semantic_robot_skill_sdk import Action, SkillCancelled, SkillContext

RGB_SENSOR_ID = "camera.rgb"
CAPTURE_TIMEOUT_SECONDS = 3.0


async def capture_stage_rgb(
    ctx: SkillContext,
    state: BaseModel,
    *,
    skill_name: str,
    point: str = "entry",
) -> None:
    """每个 Execution/Stage/point 最多采一帧；失败仅表示缺证据。

    使用 Pilot 原有 Action journal 幂等，不增加尝试计数或在控制循环中重采。
    安全 retreat/on_stop 不调用本函数；取消仍向上传播，不能被采图容错吞掉。
    """
    stage = str(getattr(state, "stage", ""))
    # 正式 RpcSkillContext 已提供 execution_id；不能为缺失身份的宿主造全局 key。
    execution_id = getattr(ctx, "execution_id", None)
    if not isinstance(execution_id, str) or not execution_id or not stage:
        _unavailable(ctx, stage, "当前 Context 缺少 Execution/Stage 身份，未采图")
        return
    key = "stage-rgb:" + json.dumps(
        [execution_id, skill_name, stage, point], ensure_ascii=True, separators=(",", ":")
    )
    ctx.check_cancelled()
    # Pilot 的 observation.recorded 从最近 checkpoint 取得 Stage，尤其是首阶段。
    ctx.checkpoint(state)
    try:
        result = await ctx.execute(
            key,
            Action(
                type="sensor.capture_rgbd",
                schema_version=2,
                parameters={"sensor_ids": [RGB_SENSOR_ID]},
                timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
                label=f"Stage RGB 证据：{stage}/{point}",
            ),
        )
    except SkillCancelled:
        raise
    except Exception as error:
        # 不把摄影/交换目录/传输问题升级为动作失败，也不把它当成完成证据。
        _unavailable(ctx, stage, f"RGB 证据采集不可用：{type(error).__name__}")
        return
    if result.status != "succeeded":
        _unavailable(ctx, stage, f"RGB 采集未成功：{result.error_code or result.status}")
        return
    frames = [
        observation for observation in result.observations
        if observation.kind == "sensor.frame"
        and (observation.value or {}).get("sensor_id") == RGB_SENSOR_ID
        and (observation.value or {}).get("media_type") in {"image/jpeg", "image/png"}
    ]
    refs = list(dict.fromkeys(
        ref for observation in frames
        for ref in [*observation.evidence_refs, *observation.artifact_refs]
        if isinstance(ref, str) and ref.startswith(("pilot-artifact://", "artifact://"))
    ))
    if not frames or not refs:
        _unavailable(ctx, stage, "未收到可发布的 RGB 帧引用，采集或 Pilot 导入不可用")
        return
    stored_refs = getattr(state, "evidence_refs", None)
    if isinstance(stored_refs, list):
        stored_refs.extend(ref for ref in refs if ref not in stored_refs)
        ctx.checkpoint(state)
    frame = frames[0]
    value = frame.value or {}
    observed_at = value.get("observed_at") or frame.observed_at.isoformat()
    ctx.report(
        "stage.evidence",
        stage=stage,
        summary="已采集阶段图像",
        observation_summary=(
            f"sensor_id={RGB_SENSOR_ID}; observed_at={observed_at}; source={frame.source}"
        ),
        evidence_refs=refs,
    )


def _unavailable(ctx: SkillContext, stage: str, reason: str) -> None:
    ctx.report(
        "stage.evidence_unavailable",
        stage=stage,
        summary="阶段图像暂不可用",
        deviation=reason,
    )
