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

"""Time the existing SDK quintic torso trajectory from measured displacement."""
import math

SPEED_LIMIT_RAD_S = 0.4
ACCELERATION_LIMIT_RAD_S2 = 1.2


def duration_seconds(current, target):
    if len(current) != 4 or len(target) != 4 or not all(math.isfinite(x) for x in (*current, *target)):
        raise ValueError("躯干轨迹需要四个有限起始角和目标角")
    delta = max(abs(a-b) for a,b in zip(current,target))
    # Same SDK quintic profile and endpoint hold as behavior-upright. The extra
    # 0.1 s covers control-period rounding beyond the SDK's >=0.5 s hold.
    travel = max(1.875*delta/SPEED_LIMIT_RAD_S,
                 math.sqrt((10/math.sqrt(3))*delta/ACCELERATION_LIMIT_RAD_S2))
    duration = travel + .6
    if duration > 30:
        raise ValueError("当前角位移无法在单次躯干轨迹预算内满足速度限制")
    return duration
