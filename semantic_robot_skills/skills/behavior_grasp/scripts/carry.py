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

"""Natural-elbow endpoint referenced to the pre-observation torso."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .kinematics import ArmModel

def limb_directions(arm, q):
    t=arm.mount.copy();index=0;frames={}
    for row,origin,axis in zip(arm.chain,arm.origins,arm.axes):
        t=t@origin
        if row['kind']=='revolute':
            a=q[index];m=np.eye(4);m[:3,:3]=np.eye(3)+np.sin(a)*axis+(1-np.cos(a))*(axis@axis)
            t=t@m;index+=1;frames[index]=t.copy()
    return -frames[3][:3,2], -frames[5][:3,2]


def natural_carry(torso, side, reference):
    arm=ArmModel(side,torso)
    lower,upper=arm.lower+np.deg2rad(5),arm.upper-np.deg2rad(5)
    natural=np.array([-(torso[0]+torso[1]-torso[2]),0.,-torso[3],-np.pi/2,0.,0.,0.])
    def residual(y):
        upper_axis,fore=limb_directions(arm,y);axis=arm.forward(y)[:3,:3]@arm.optical
        return np.r_[upper_axis-[0,0,-1],fore-[1,0,0],axis[1],min(0,axis[0]),.0003*(y-natural)]
    rng=np.random.default_rng(20260928);solutions=[]
    for seed in [natural]+[rng.uniform(lower,upper) for _ in range(10)]:
        fit=least_squares(residual,np.clip(seed,lower+1e-8,upper-1e-8),bounds=(lower,upper),
                          max_nfev=240,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        u,f=limb_directions(arm,fit.x);t=arm.forward(fit.x);axis=t[:3,:3]@arm.optical
        errors=np.rad2deg([np.arccos(np.clip(-u[2],-1,1)),np.arccos(np.clip(f[0],-1,1)),np.arcsin(np.clip(abs(axis[1]),0,1))])
        if max(errors)<1 and axis[0]>=0:
            solutions.append((np.linalg.norm(fit.x-natural),fit.x,t,errors))
    if not solutions:raise ValueError('未找到大臂向下、小臂向前、镜头位于 XZ 面的自然屈肘姿态')
    _,goal,t,errors=min(solutions,key=lambda v:v[0])
    return reference.model_copy(update={'position_m':t[:3,3].tolist(),'orientation_xyzw':R.from_matrix(t[:3,:3]).as_quat().tolist()})

