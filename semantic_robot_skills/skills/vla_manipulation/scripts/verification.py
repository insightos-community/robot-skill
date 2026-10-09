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

"""根据物理事实验收，不把模型响应或控制成功当成业务成功。"""

from math import dist

from .models import ManipulationInput, ManipulationState


def assess(data: dict, inputs: ManipulationInput, state: ManipulationState) -> bool:
    if data["generation"] != state.generation:
        raise ValueError("场景已重置，本次 VLA 验收失效")
    if inputs.objective == "native_task":
        return data["evaluation"]["success"] is True
    target = data["target"]
    if target["source_id"] != inputs.target_source_id:
        raise ValueError("实时验收物品与指定目标不一致")
    position = target["pose"]["position"]
    hand = data["robot_state"]["end_effectors"]["hand"]["position"]
    relative = [position[i] - hand[i] for i in range(3)]
    bilateral = any(contact["source_id"] == inputs.target_source_id
                    and contact["bilateral_contact"] is True
                    for contact in data["contacts"]["contacts"])
    opening = data["robot_state"]["grippers"]["hand"]
    # Panda 抓取验收：目标离开初始支撑高度、双指接触同一目标、夹爪不是空闭合，
    # 且连续采样中目标相对手部位置稳定。阈值只属于本 Skill 的验收，不修改
    # SDK/Ability 控制器或上游 BDDL；native_task 完全使用原生评测结果。
    eligible = (bilateral and opening > .001
                and position[2] - state.initial_target_height >= .02)
    previous = state.previous_sample
    same_sample = previous and data["sequence"] <= previous["sequence"]
    if same_sample:
        return False  # 重复观测不能累计“稳定时长”。
    if (not eligible or (previous and dist(relative, previous["relative"]) > .015)):
        state.stable_since = None
    elif state.stable_since is None:
        state.stable_since = data["sim_time"]
    state.previous_sample = {"sequence": data["sequence"], "relative": relative}
    return (eligible and state.stable_since is not None
            and data["sim_time"] - state.stable_since >= .15)
