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

"""Natural elbow posture, optical look-at and axial roll using measured robot joints."""
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import qmc
from scipy.spatial.transform import Rotation as R
from .chest_planner import ArmModel, ChestPlanError, model
from .models import Pose


def current(arm, sample, measured):
    if len(sample.names) != len(sample.positions_rad) or len(set(sample.names)) != len(sample.names):
        raise ValueError('关节观测名称和角度不匹配')
    values = dict(zip(sample.names, sample.positions_rad))
    q = np.array([values[n] for n in arm.names])
    if not np.isfinite(q).all():
        raise ValueError('关节观测含非有限值')
    actual = arm.forward(q)
    if (np.linalg.norm(actual[:3, 3]-measured.position_m) > .005 or
        (R.from_matrix(actual[:3,:3])*R.from_quat(measured.orientation_xyzw).inv()).magnitude() > np.deg2rad(1)):
        raise ValueError('实测末端位姿与运动学模型不一致')
    return q


def as_pose(t, reference):
    return Pose(position_m=t[:3,3].tolist(), orientation_xyzw=R.from_matrix(t[:3,:3]).as_quat().tolist(),
                revision=reference.revision, observed_at=reference.observed_at)


def limb_directions(arm, q):
    t=arm.mount.copy();index=0;frames={}
    for row,origin,axis in zip(arm.chain,arm.origins,arm.axes):
        t=t@origin
        if row['kind']=='revolute':
            a=q[index];m=np.eye(4);m[:3,:3]=np.eye(3)+np.sin(a)*axis+(1-np.cos(a))*(axis@axis)
            t=t@m;index+=1;frames[index]=t.copy()
    return -frames[3][:3,2], -frames[5][:3,2]


def plan_natural(measured, sample, torso, side):
    arm=ArmModel(side,torso);q=current(arm,sample,measured)
    lower,upper=arm.lower+np.deg2rad(5),arm.upper-np.deg2rad(5)
    natural=np.array([-(torso[0]+torso[1]-torso[2]),0.,-torso[3],-np.pi/2,0.,0.,0.])
    def residual(y):
        upper_axis,fore=limb_directions(arm,y);axis=arm.forward(y)[:3,:3]@arm.optical
        return np.r_[upper_axis-[0,0,-1],fore-[1,0,0],axis[1],min(0,axis[0]),.0003*(y-natural)]
    rng=np.random.default_rng(20260928);solutions=[]
    for seed in [natural,q]+[rng.uniform(lower,upper) for _ in range(10)]:
        fit=least_squares(residual,np.clip(seed,lower+1e-8,upper-1e-8),bounds=(lower,upper),
                          max_nfev=240,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        u,f=limb_directions(arm,fit.x);t=arm.forward(fit.x);axis=t[:3,:3]@arm.optical
        errors=np.rad2deg([np.arccos(np.clip(-u[2],-1,1)),np.arccos(np.clip(f[0],-1,1)),np.arcsin(np.clip(abs(axis[1]),0,1))])
        if max(errors)<1 and axis[0]>=0:
            solutions.append((np.linalg.norm(fit.x-natural),fit.x,t,errors))
    if not solutions:raise ChestPlanError('未找到大臂向下、小臂向前、镜头位于 XZ 面的自然屈肘姿态')
    _,goal,t,errors=min(solutions,key=lambda v:v[0])
    return as_pose(t,measured),{'limb_errors_deg':errors.tolist()}


def plan_observation(holding, measured, sample, torso, side, frame):
    arm=ArmModel(side,torso);q=current(arm,sample,measured)
    point=np.asarray(holding.position_m,dtype=float)
    calibration=frame["calibration"]
    k=np.asarray(calibration["intrinsic_matrix"],dtype=float)
    body_camera=np.asarray(calibration["camera_pose_body"],dtype=float)@np.diag([1.,-1.,-1.,1.])
    mount=np.linalg.inv(arm.forward(q))@body_camera
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError("腕部相机内参无效")
    size=np.array([frame["width"]-1,frame["height"]-1])
    if min(size)<=0:raise ValueError("腕部图像尺寸无效")
    lower,upper=arm.lower+np.deg2rad(5),arm.upper-np.deg2rad(5)
    sign=1 if side=="left" else -1
    tool_camera=np.linalg.norm(mount[:3,3])
    # Preserve the measured viewing distance as a soft preference. The requested
    # downward translation is applied after this nominal look-at solution.
    reference_distance=np.linalg.norm(body_camera[:3,3]-point)
    reach=sum(np.linalg.norm(o[:3,3]) for o in arm.origins)+np.linalg.norm(arm.tool[:3,3])
    def camera(y):return arm.forward(y)@mount
    def optical(y):
        c=camera(y);return c[:3,:3].T@(point-c[:3,3])
    def limits(y):
        c=camera(y);p=optical(y)
        uv=(k@p)[:2]/max(p[2],np.finfo(float).eps)/size
        return np.r_[p[2]-np.finfo(float).eps,uv,1-uv,
            sign*(c[1,3]-point[1]),np.linalg.norm(c[:3,3]-point)-tool_camera]
    def cost(y):
        c=camera(y)
        distance=np.linalg.norm(c[:3,3]-point)
        return float(np.sum(((y-q)/(upper-lower))**2)+((distance-reference_distance)/reach)**2+(1+c[2,1])**2)
    seeds=[q,(lower+upper)/2]+list(qmc.scale(qmc.Halton(7,scramble=False).random(14),lower,upper))
    solutions=[]
    for seed in seeds:
        fit=minimize(cost,np.clip(seed,lower+1e-8,upper-1e-8),method='SLSQP',bounds=list(zip(lower,upper)),
            constraints=[{'type':'eq','fun':lambda y:optical(y)[:2]},{'type':'ineq','fun':limits}],
            options={'maxiter':250,'ftol':1e-10})
        if np.isfinite(fit.x).all() and np.linalg.norm(optical(fit.x)[:2])<.001 and limits(fit.x).min()>-1e-7:
            solutions.append((cost(fit.x),fit.x))
    if not solutions:raise ChestPlanError('未找到朝向持物末端且可达的腕部观察位置')
    _,goal=min(solutions,key=lambda v:v[0]);t=arm.forward(goal)
    nominal=as_pose(t,measured)
    offset=np.array([0.,0.,-.05])
    target=nominal.model_copy(update={'position_m':(t[:3,3]+offset).tolist()})
    target_matrix=np.eye(4)
    target_matrix[:3,:3]=R.from_quat(target.orientation_xyzw).as_matrix()
    target_matrix[:3,3]=target.position_m
    target_camera=target_matrix@mount
    local=target_camera[:3,:3].T@(point-target_camera[:3,3])
    uv=(k@local)[:2]/max(local[2],np.finfo(float).eps)/size
    if (local[2]<=0 or np.any(uv<0) or np.any(uv>1)
            or sign*(target_camera[1,3]-point[1])<0
            or np.linalg.norm(target_camera[:3,3]-point)<tool_camera):
        raise ChestPlanError('观察位下移后持物末端超出原有视场或观察侧约束')
    return target,{
        'look_at_body_m':point.tolist(),'camera_body':target_camera.tolist(),
        'nominal_observation_pose':nominal.model_dump(mode='json'),
        'observation_offset_body_m':offset.tolist(),
        'candidate_count':len(solutions),'reference_distance_m':float(reference_distance),
        'camera_distance_m':float(np.linalg.norm(target_camera[:3,3]-point))}


def plan_roll(measured,sample,torso,side):
    """Request the opposite face at the current endpoint; cuRobo chooses the path."""
    if side not in ('left','right'):raise ValueError('Unknown holding side')
    sign=1 if side=='left' else -1
    rotation=R.from_quat(measured.orientation_xyzw)*R.from_rotvec([0.,0.,sign*np.pi])
    return measured.model_copy(update={'orientation_xyzw':rotation.as_quat().tolist()}),{
        'target_roll_rad':float(sign*np.pi),'planner':'curobo'}


def plan_button(measured, sample, torso, side, button):
    """Choose a reachable button pose with free wrist orientation and useful tool direction."""
    arm=ArmModel(side,torso);q=current(arm,sample,measured)
    position=np.asarray(button.pose.position_m,dtype=float)
    direction=position-np.asarray(measured.position_m)
    if np.linalg.norm(direction)<.005:
        direction=arm.forward(q)[:3,2]
    direction/=np.linalg.norm(direction)
    reference=R.from_quat(measured.orientation_xyzw).as_matrix()
    lower,upper=arm.lower+np.deg2rad(5),arm.upper-np.deg2rad(5)
    def cost(y):
        t=arm.forward(y);angle=R.from_matrix(reference.T@t[:3,:3]).magnitude()
        return float(angle**2+.05*np.sum((y-q)**2))
    def facing(y):
        return float(arm.forward(y)[:3,2]@direction-np.cos(np.deg2rad(60)))
    rng=np.random.default_rng(20260922);solutions=[]
    for seed in [q,(lower+upper)/2]+[rng.uniform(lower,upper) for _ in range(10)]:
        fit=minimize(cost,np.clip(seed,lower+1e-8,upper-1e-8),method='SLSQP',bounds=list(zip(lower,upper)),
                     constraints=[{'type':'eq','fun':lambda y:arm.forward(y)[:3,3]-position},
                                  {'type':'ineq','fun':facing}],options={'maxiter':250,'ftol':1e-10})
        if not np.isfinite(fit.x).all():continue
        t=arm.forward(fit.x)
        if (np.linalg.norm(t[:3,3]-position)<=.001 and facing(fit.x)>=-1e-7
            and np.all(fit.x>=lower-1e-8) and np.all(fit.x<=upper+1e-8)):
            solutions.append((cost(fit.x),fit.x,t))
    if not solutions:raise ChestPlanError('按钮位置未找到满足关节余量和夹爪接近方向的末端姿态')
    _,target,t=min(solutions,key=lambda v:v[0])
    pose=Pose(position_m=position.tolist(),orientation_xyzw=R.from_matrix(t[:3,:3]).as_quat().tolist(),
              revision=button.pose.revision,observed_at=button.pose.observed_at)
    return pose,{'joint_solution_rad':target.tolist(),
                 'minimum_joint_margin_deg':float(np.rad2deg(np.min(np.r_[target-arm.lower,arm.upper-target]))),
                 'orientation_adjustment_deg':float(np.rad2deg(R.from_matrix(reference.T@t[:3,:3]).magnitude())),
                 'tool_approach_angle_deg':float(np.rad2deg(np.arccos(np.clip(t[:3,2]@direction,-1,1)))),
                 'candidate_count':len(solutions)}
