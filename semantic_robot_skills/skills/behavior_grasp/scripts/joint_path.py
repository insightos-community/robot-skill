"""Collision-constrained joint paths and finite, single-submission trajectories."""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator,CubicHermiteSpline
from .kinematics import ArmModel
from .operation import JOINT_MARGIN_RAD,POSITION_TOLERANCE_M,PLANE_TOLERANCE_RAD
from .self_collision import robot_collision,require_joint_path,joint_map
from .torso_motion import SPEED_LIMIT_RAD_S,ACCELERATION_LIMIT_RAD_S2
from .trajectory_timing import retime_path

CONTROL_PERIOD_S=1/30
ARM_SPEED_LIMIT_RAD_S=.5
COLLISION_CLEARANCE_M=.001
STATION_DISTANCE_M=.005

def as_joints(values):return {'names':list(values),'positions_rad':list(values.values())}


def refine_retraction(state,goal,torso,side):
    model=robot_collision();arm=ArmModel(side,[state[f'torso_joint{i}'] for i in range(1,5)])
    names=arm.names;initial=np.array([state[n] for n in names]);goal=np.array(goal)
    lo,hi=arm.lower+JOINT_MARGIN_RAD,arm.upper-JOINT_MARGIN_RAD
    torso_names=[f'torso_joint{i}' for i in range(1,5)];start_torso=np.array([state[n] for n in torso_names])
    pairs=[p for p in model.pairs if any(n.startswith(side+'_') for n in p)]
    def constraints(q):
        samples=[]
        for t in np.linspace(0,1,7):
            samples.append(model.distances({**state,**dict(zip(names,initial+t*(q-initial)))},pairs))
            samples.append(model.distances({**state,**dict(zip(names,q)),
                **dict(zip(torso_names,start_torso+t*(np.array(torso)-start_torso)))},pairs))
        return np.min(samples,axis=0)-COLLISION_CLEARANCE_M
    fit=minimize(lambda q:float(np.sum(((q-goal)/(hi-lo))**2)),np.clip(goal,lo,hi),method='SLSQP',bounds=list(zip(lo,hi)),
        constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':60,'ftol':1e-8})
    if not np.isfinite(fit.x).all() or constraints(fit.x).min() < -1e-5:
        raise ValueError(f'{side} 未找到无自碰撞的收臂及躯干过渡')
    require_joint_path(as_joints(state),names,[fit.x],'grasp:retract:'+side)
    return fit.x


def solve_pose(state,side,pose,seed,anchor):
    model=robot_collision();arm=ArmModel(side,[state[f'torso_joint{i}'] for i in range(1,5)])
    lo,hi=arm.lower+JOINT_MARGIN_RAD,arm.upper-JOINT_MARGIN_RAD
    pairs=[p for p in model.pairs if any(n.startswith(side+'_') for n in p)]
    rotation=Rotation.from_quat(pose.orientation_xyzw).as_matrix();cache={}
    def distances(q):
        key=np.asarray(q).tobytes()
        if key not in cache:cache[key]=model.distances({**state,**dict(zip(arm.names,q))},pairs)
        return cache[key]
    def error(q):
        t=arm.forward(q)
        return np.r_[t[:3,3]-pose.position_m,Rotation.from_matrix(rotation.T@t[:3,:3]).as_rotvec()]
    fit=minimize(lambda q:float(np.sum(((q-anchor)/(hi-lo))**2)),np.clip(seed,lo,hi),method='SLSQP',bounds=list(zip(lo,hi)),
        constraints=[{'type':'eq','fun':error},{'type':'ineq','fun':lambda q:distances(q)-COLLISION_CLEARANCE_M}],
        options={'maxiter':100,'ftol':1e-9})
    err=error(fit.x);gaps=distances(fit.x)
    if (not np.isfinite(fit.x).all() or np.linalg.norm(err[:3])>POSITION_TOLERANCE_M
            or np.linalg.norm(err[3:])>PLANE_TOLERANCE_RAD or gaps.min()<COLLISION_CLEARANCE_M-1e-5):
        raise ValueError(f'{side} 碰撞约束 IK 未满足目标: 位置误差 {np.linalg.norm(err[:3]):.4f} m，最小间隙 {gaps.min():.4f} m')
    return fit.x


def time_path(names,rows,stage):
    rows=np.asarray(rows,dtype=float)
    keep=np.r_[True,np.max(np.abs(np.diff(rows,axis=0)),axis=1)>1e-10];rows=rows[keep]
    limits=np.array([SPEED_LIMIT_RAD_S if n.startswith('torso_') else ARM_SPEED_LIMIT_RAD_S for n in names])
    if len(rows)==1:
        samples=np.repeat(rows,2,axis=0)
    else:
        spans=np.maximum(np.max(np.abs(np.diff(rows,axis=0))/limits,axis=1),CONTROL_PERIOD_S)
        times=np.r_[0.,np.cumsum(spans)]
        slopes=PchipInterpolator(times,rows,axis=0).derivative()(times);slopes[[0,-1]]=0
        curve=CubicHermiteSpline(times,rows,slopes,axis=0)
        samples=retime_path(curve,limits,ACCELERATION_LIMIT_RAD_S2,CONTROL_PERIOD_S)
    require_joint_path({'names':names,'positions_rad':samples[0].tolist()},names,samples,stage)
    # Finite terminal hold uses the existing torso sequence's settling window.
    samples=np.vstack([samples,np.repeat(samples[-1:],math.ceil(.5/CONTROL_PERIOD_S),axis=0)])
    return {'joint_names':list(names),'positions_rad':samples.tolist(),'control_period_s':CONTROL_PERIOD_S}


def plan_joint_paths(plan,measured,side,target_torso,retraction,operation_diagnostics):
    state=joint_map(measured);names=robot_collision().moving;start=state.copy()
    approach=[[state[n] for n in names]];arm_names=[f'{side}_arm_joint{i}' for i in range(1,8)]
    if retraction:
        for row in retraction['arms']:
            q=refine_retraction(state,row['positions_rad'],target_torso,row['arm_side'])
            row['positions_rad']=q.tolist();state.update(zip(row['joint_names'],q));approach.append([state[n] for n in names])
        tn=[f'torso_joint{i}' for i in range(1,5)]
        require_joint_path(as_joints(state),tn,[target_torso],'grasp:operation-torso')
        state.update(zip(tn,target_torso));approach.append([state[n] for n in names])
    current=np.array([state[n] for n in arm_names]);paths={}
    for stage,index in [('pregrasp',1),('grasp',2),('lift',3)]:
        pose=getattr(plan,stage);rows=[current.tolist()]
        if stage=='pregrasp':
            seed=(operation_diagnostics['target_joint_positions_rad'][index] if operation_diagnostics else current)
            q=solve_pose(state,side,pose,np.array(seed),current);rows.append(q.tolist())
        else:
            origin=getattr(plan,'pregrasp' if stage=='grasp' else 'grasp')
            count=max(1,int(np.ceil(np.linalg.norm(np.asarray(pose.position_m)-origin.position_m)/STATION_DISTANCE_M)))
            for t in np.linspace(0,1,count+1)[1:]:
                waypoint=pose.model_copy(update={'position_m':(np.asarray(origin.position_m)+t*(np.asarray(pose.position_m)-origin.position_m)).tolist()})
                q=solve_pose(state,side,waypoint,current,current);rows.append(q.tolist());current=q
        require_joint_path(as_joints(state),arm_names,rows,'grasp:'+stage)
        paths[stage]=rows
        if stage=='lift':
            lift=[[state[n] for n in names]]
            for qrow in rows[1:]:
                state.update(zip(arm_names,qrow));lift.append([state[n] for n in names])
        else:
            for qrow in rows[1:]:
                state.update(zip(arm_names,qrow));approach.append([state[n] for n in names])
        current=np.array(rows[-1])
    trajectories={'approach':time_path(names,approach,'grasp:approach-trajectory'),
                  'lift':time_path(names,lift,'grasp:lift-trajectory')}
    return {'trajectories':trajectories,'paths':paths,'retraction_plan':retraction,
            'validation':'native_exclusions_collision_constrained_joint_path_and_timed_samples'}
