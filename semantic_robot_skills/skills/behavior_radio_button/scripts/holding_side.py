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

"""Infer arm roles from bilateral gripper feedback without moving either arm."""
import numpy as np

SAMPLE_COUNT = 5
SAMPLE_INTERVAL_SECONDS = .2
# Role discrimination thresholds, normalized by the reported force limit.
# These classify a loaded candidate and a quiet peer, not successful grasping.
LOADED_FORCE_RATIO = .05
QUIET_FORCE_RATIO = .01
OPENING_END_TOLERANCE = .01
POSITION_STABILITY_M = .0005


def infer_holding_side(samples):
    if len(samples) < SAMPLE_COUNT:
        raise ValueError("判断持物侧需要完整的双手夹爪采样")
    rows = {side:[] for side in ("left","right")}
    for tools in samples:
        if len(tools) != 2 or {t.get("side") for t in tools} != set(rows):
            raise ValueError("夹爪反馈须同时包含唯一的 left 和 right")
        for tool in tools: rows[tool["side"]].append(tool)
    diagnostics = {}
    for side, tools in rows.items():
        def array(key):
            value=np.asarray([t[key] for t in tools],dtype=float)
            if value.ndim!=2 or value.shape[1]!=2 or not np.isfinite(value).all():
                raise ValueError(f"{side} 夹爪 {key} 须为两个有限手指值")
            return value
        positions=array("positions_m");forces=np.abs(array("efforts_n"));limits=array("effort_limits_n")
        closed=array("closed_positions_m");opened=array("open_positions_m");span=opened-closed
        if np.any(limits<=0) or np.any(span<=0):raise ValueError("夹爪力限和行程须为正")
        ratios=np.max(forces/limits,axis=1)
        opening=np.mean((positions-closed)/span,axis=1)
        motion=float(np.max(np.ptp(positions,axis=0)))
        stable=motion<=POSITION_STABILITY_M
        away_from_end=bool(np.all((opening>OPENING_END_TOLERANCE)&(opening<1-OPENING_END_TOLERANCE)))
        loaded=bool(stable and away_from_end and np.quantile(ratios,.25)>=LOADED_FORCE_RATIO)
        quiet=bool(stable and np.quantile(ratios,.75)<=QUIET_FORCE_RATIO)
        diagnostics[side]={"loaded_candidate":loaded,"quiet":quiet,"stable":stable,
            "median_efforts_abs_n":np.median(forces,axis=0).tolist(),
            "median_positions_m":np.median(positions,axis=0).tolist(),
            "median_opening_ratio":float(np.median(opening)),"position_range_m":motion,
            "force_ratio_q25":float(np.quantile(ratios,.25)),"force_ratio_q75":float(np.quantile(ratios,.75))}
    candidates=[side for side in rows if diagnostics[side]["loaded_candidate"]
                and diagnostics["right" if side=="left" else "left"]["quiet"]]
    holding=candidates[0] if len(candidates)==1 else None
    return {"holding_side":holding,"operating_side":("right" if holding=="left" else "left") if holding else None,
        "method":"bilateral_force_opening_stability","sample_count":len(samples),"hands":diagnostics,
        "thresholds":{"loaded_force_ratio":LOADED_FORCE_RATIO,"quiet_force_ratio":QUIET_FORCE_RATIO,
            "opening_end_tolerance":OPENING_END_TOLERANCE,"position_stability_m":POSITION_STABILITY_M}}
