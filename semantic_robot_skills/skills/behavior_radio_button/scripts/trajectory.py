"""Vendored collision IK and trajectory timing from behavior-grasp 0.1.31.

Kept inside this standalone Skill package; no cross-Skill runtime dependency.
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from .chest_planner import ArmModel
from .self_collision import robot_collision, require_joint_path

JOINT_MARGIN_RAD=np.deg2rad(5)
POSITION_TOLERANCE_M=.001
PLANE_TOLERANCE_RAD=np.deg2rad(.5)
CONTROL_PERIOD_S=1/30
ARM_SPEED_LIMIT_RAD_S=.5
SPEED_LIMIT_RAD_S=.4
ACCELERATION_LIMIT_RAD_S2=1.2
COLLISION_CLEARANCE_M=.001


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
        vmax=np.zeros(len(names));amax=np.zeros(len(names))
        for i,length in enumerate(np.diff(times)):
            a,b,c,_=curve.c[:,i,:]
            for j in range(len(names)):
                probes=[0.,length]
                if abs(a[j])>1e-15:
                    t=-b[j]/(3*a[j])
                    if 0<t<length:probes.append(t)
                vmax[j]=max(vmax[j],*(abs(3*a[j]*t*t+2*b[j]*t+c[j]) for t in probes))
                amax[j]=max(amax[j],abs(2*b[j]),abs(6*a[j]*length+2*b[j]))
        scale=max(1.,float(np.max(vmax/limits)),math.sqrt(float(amax.max())/ACCELERATION_LIMIT_RAD_S2))
        count=math.ceil(times[-1]*scale/CONTROL_PERIOD_S);scale=count*CONTROL_PERIOD_S/times[-1]
        curve=CubicHermiteSpline(times*scale,rows,slopes/scale,axis=0)
        samples=curve(np.arange(count+1)*CONTROL_PERIOD_S)
    require_joint_path({'names':names,'positions_rad':samples[0].tolist()},names,samples,stage)
    # Finite terminal hold uses the existing torso sequence's settling window.
    samples=np.vstack([samples,np.repeat(samples[-1:],math.ceil(.5/CONTROL_PERIOD_S),axis=0)])
    return {'joint_names':list(names),'positions_rad':samples.tolist(),'control_period_s':CONTROL_PERIOD_S}
