"""Couple a compact retraction candidate with the complete endpoint replay."""
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from . import retract
from .kinematics import ArmModel, torso_in_body
from .operation import JOINT_MARGIN_RAD, validate_fixed_poses
from .path_replay import validate_endpoint_path, PathReplayError
from .self_collision import require_retraction, SelfCollisionError


def coupled_retraction(plan, predicted, side, measured, target, source, baseline_plan):
    observed = retract.joint_values(measured['joints'])
    torso = np.array([observed[f'torso_joint{i}'] for i in range(1,5)])
    arm=ArmModel(side,torso)
    initial=np.array([observed[n] for n in arm.names])
    baseline=np.array(next(a['positions_rad'] for a in baseline_plan['arms'] if a['arm_side']==side))
    low,high=arm.lower+JOINT_MARGIN_RAD,arm.upper-JOINT_MARGIN_RAD
    box=retract.target_box(source,measured['base_pose'])
    mounts=np.array([torso_in_body(q) for q in retract.interpolation(torso,np.array(target))])
    first=retract.link_points(arm,initial)[0];shoulder=first[0];reach=np.linalg.norm(np.diff(first,axis=0),axis=1).sum();sign=1 if side=='left' else -1
    fractions=np.linspace(0,1,int(np.ceil(np.max(high-low)/retract.SAMPLE_ANGLE_RAD))+1)[:,None]
    def constraints(q):
     points=retract.link_points(arm,q)[0];path=retract.link_points(arm,initial+fractions*(q-initial))
     body_path=retract.in_body(path,np.repeat(mounts[:1],len(path),axis=0))
     sweep=retract.in_body(np.repeat(points[None],len(mounts),axis=0),mounts)
     return np.r_[retract.segment_box_margins(body_path,box).ravel(),retract.segment_box_margins(sweep,box).ravel(),
      sign*path[:,:,1].ravel(),sign*(points[1:,1]-shoulder[1]),shoulder[2]-points[3,2],points[-1,2]-points[3,2],shoulder[2]-points[-1,2]]
    points = (plan.orientation_selection or {}).get('points')
    if points:
        goal = np.array(next(p['joint_positions_rad'] for p in points if p['key']=='grasp:pregrasp'))
    else:
        samples = validate_fixed_poses(plan,predicted,side,sample_path=True)
        goal = np.array(next(p['joint_positions_rad'] for p in samples if p['key']=='pregrasp'))
    seen = []; failures = []; last_error = None
    for weight in (0., .5, 1.):
        def objective(q):
            points = retract.link_points(arm,q)[0]
            compactness = np.mean(np.sum(((points[3:]-shoulder)/reach)**2,axis=1))
            future = np.mean(((q-goal)/(high-low))**2)
            movement = np.mean(((q-initial)/(high-low))**2)
            center = np.mean(((q-(low+high)/2)/(high-low))**2)
            return float((1-weight)*compactness+weight*future+movement+center)
        for seed in (baseline,np.clip(goal,low,high)):
            fit = minimize(objective,seed,method='SLSQP',bounds=list(zip(low,high)),
                constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':180,'ftol':1e-9})
            if not np.isfinite(fit.x).all() or constraints(fit.x).min() < -1e-7:continue
            q=fit.x
            if any(np.linalg.norm(q-v)<1e-4 for v in seen):continue
            seen.append(q.copy())
            # Entire joint interpolation stays inside the same margin bounds.
            path=initial+fractions*(q-initial)
            if np.min(np.r_[np.ravel(path-arm.lower),np.ravel(arm.upper-path)]) < JOINT_MARGIN_RAD-1e-5:continue
            trial=deepcopy(predicted)
            for name,value in zip(arm.names,q):trial['positions_rad'][trial['names'].index(name)]=float(value)
            result=deepcopy(baseline_plan)
            selected=next(a for a in result['arms'] if a['arm_side']==side)
            selected['positions_rad']=q.tolist()
            try:
                collision_checks=require_retraction(measured['joints'],result)
                replay=validate_endpoint_path(plan,trial,side)
            except (PathReplayError,SelfCollisionError) as error:
                last_error=error
                failures.append({k:v for k,v in error.diagnostic.items() if k!='samples'})
                continue
            result['self_collision']=collision_checks
            result['diagnostics'][side].update(
                retracted_sweep_margin_m=retract.sweep_margin(arm,q,mounts,box),
                retracted_tool_torso_m=retract.link_points(arm,q)[0,-1].tolist(),
                objective_blend=weight,minimum_joint_margin_deg=float(np.rad2deg(np.min(np.r_[q-arm.lower,arm.upper-q]))))
            replay['retraction_plan']=result
            replay['rejected_retractions']=failures
            return replay
    if last_error is not None:raise last_error
    raise ValueError(f'{side} 未找到收臂过渡及连续接近同时可行的组合')
