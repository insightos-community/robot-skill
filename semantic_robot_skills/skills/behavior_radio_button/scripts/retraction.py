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

"""Semantic retraction goals; SDK plans each motion between endpoints."""
import numpy as np
from .models import MotionPlan


def retraction_offset(side, *, after_flip=False):
    if side not in ('left','right'):raise ValueError('Unknown holding side')
    return np.array([-.01 if after_flip else -.10, -.10 if side=='left' else .10, 0.])


def shifted_plan(plan, *, after_flip=False):
    offset=retraction_offset(plan.side,after_flip=after_flip)
    return MotionPlan(side=plan.side,
        pose=plan.pose.model_copy(update={'position_m':(np.asarray(plan.pose.position_m)+offset).tolist()}),
        diagnostics={'retraction_offset_body_m':offset.tolist()})


def plan_two_stage(measured, sample, torso, side, planner):
    pose,diagnostics=planner(measured,sample,torso,side)
    natural=MotionPlan(pose=pose,side=side,diagnostics=diagnostics)
    return natural,shifted_plan(natural)


def plan_flip_and_retract(measured, sample, torso, side, planner):
    pose,diagnostics=planner(measured,sample,torso,side)
    flip=MotionPlan(pose=pose,side=side,diagnostics=diagnostics)
    return flip,shifted_plan(flip,after_flip=True)
