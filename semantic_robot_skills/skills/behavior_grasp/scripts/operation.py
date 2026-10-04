"""Jointly solve torso and endpoint reach from calibrated robot geometry."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
from .kinematics import ArmModel, torso_in_body
from .geometry import LOWER, UPPER
from .grasp_axes import axes, DOWNWARD_EPSILON, prefer_downward

# Same numerical acceptance as the existing camera-plane path planner.
POSITION_TOLERANCE_M = .001
PLANE_TOLERANCE_RAD = np.deg2rad(.5)
JOINT_MARGIN_RAD = np.deg2rad(5)


def operation_pose(plan, orientation, joints, side, automatic, principal_axis=None, *, seed_limit=None):
    # Low-target branch is derived from the native knee height at its joint limit.
    knee_low = .34265 + .4 * np.cos(UPPER[0] - JOINT_MARGIN_RAD)
    low_target = plan.grasp.position_m[2] < knee_low
    warm = None
    try:
        regular = _operation_pose(plan, orientation, joints, side, automatic, principal_axis,
                                  seed_limit=seed_limit)
        warm = np.r_[regular[0], np.asarray(regular[1]['target_joint_positions_rad']).ravel()]
    except ValueError:
        if not low_target: raise
    if low_target:
        target, diagnostics = _operation_pose(plan, orientation, joints, side, automatic,
            principal_axis, seed_limit=None, forward_lower=True, warm_start=warm)
        diagnostics['torso_branch'] = 'low_target_forward_lower_link'
        diagnostics['lower_link_forward_deg'] = float(np.rad2deg(target[0]+target[1]))
        diagnostics['low_target_height_boundary_m'] = float(knee_low)
        return target, diagnostics
    regular[1]['torso_branch'] = 'standard'
    return regular


def _operation_pose(plan, orientation, joints, side, automatic, principal_axis=None, *,
                    seed_limit=None, forward_lower=False, warm_start=None):
    observed = dict(zip(joints["names"], joints["positions_rad"]))
    torso = np.array([observed[f"torso_joint{i}"] for i in range(1,5)])
    arm = ArmModel(side,torso)
    initial = np.array([observed[n] for n in arm.names])
    targets = [getattr(plan,k) for k in ("clearance","pregrasp","grasp","lift")]
    al,au=arm.lower+JOINT_MARGIN_RAD,arm.upper-JOINT_MARGIN_RAD
    low=np.r_[np.array(LOWER)+JOINT_MARGIN_RAD,np.tile(al,len(targets))]
    high=np.r_[np.array(UPPER)-JOINT_MARGIN_RAD,np.tile(au,len(targets))]
    # Task policy: waist yaw stays within either side of body-forward. Apply
    # after model-limit margins so the requested +/-90 degrees remain usable.
    low[3]=max(low[3],-np.pi/2)
    high[3]=min(high[3],np.pi/2)
    start=np.clip(np.r_[torso,np.tile(initial,len(targets))],low,high)
    span=high-low
    if automatic: axis,up=axes(principal_axis,orientation)
    rotations=[Rotation.from_quat(t.orientation_xyzw).as_matrix() for t in targets]

    def transforms(x):
        arm.mount=torso_in_body(x[:4])
        return [arm.forward(q) for q in x[4:].reshape(-1,len(initial))]

    def equations(x):
        result=[]
        for t,target,rotation in zip(transforms(x),targets,rotations):
            result.extend(t[:3,3]-target.position_m)
            if automatic:
                result.append(t[:3,1]@axis)
            else:
                result.extend(Rotation.from_matrix(rotation.T@t[:3,:3]).as_rotvec())
        return np.asarray(result)

    def constraints(x):
        q1,q2,q3,_=x[:4];pitch=q1+q2-q3
        # R1Pro URDF branch: forward knee, waist above knee, upright to forward.
        c=[np.sin(q1),(1 if forward_lower else -1)*np.sin(q1+q2),np.cos(q1+q2),pitch,np.pi/2-pitch]
        if automatic:
            c += [-t[:3,2]@up-DOWNWARD_EPSILON for t in transforms(x)]
        return np.asarray(c)

    def objective(x):
        delta=(x-start)/span
        arms=x[4:].reshape(-1,len(initial))
        return float(delta@delta+np.sum((np.diff(arms,axis=0)/(au-al))**2))

    def tilt_cost(x):
        # Pregrasp, contact and lift take priority; the lateral waypoint can tilt.
        return float(sum(1.+t[:3,2]@up for t in transforms(x)[1:]))

    # Seeds cover joint limits; no remembered scene poses or prescribed pitch.
    samples=qmc.Halton(4+len(initial),scramble=False).random(12)
    seeds=[start] if warm_start is None else [warm_start,start]
    for sample in samples:
        q=low[:4]+sample[:4]*span[:4]
        a=al+sample[4:]*(au-al)
        seeds.append(np.r_[q,np.tile(a,len(targets))])
    best=None
    for seed in seeds if seed_limit is None else seeds[:seed_limit]:
        checks=[{"type":"eq","fun":equations},{"type":"ineq","fun":constraints}]
        if automatic:
            fit=prefer_downward(seed,list(zip(low,high)),checks,tilt_cost,objective,maxiter=200)
        else:
            fit=minimize(objective,seed,method="SLSQP",bounds=list(zip(low,high)),
                constraints=checks,options={"maxiter":200,"ftol":1e-9})
        if not np.isfinite(fit.x).all() or constraints(fit.x).min() < -1e-6:
            continue
        ts=transforms(fit.x)
        errors=[np.linalg.norm(t[:3,3]-p.position_m) for t,p in zip(ts,targets)]
        angular=([abs(t[:3,1]@axis) for t in ts] if automatic else
                 [Rotation.from_matrix(r.T@t[:3,:3]).magnitude() for r,t in zip(rotations,ts)])
        if max(errors)>POSITION_TOLERANCE_M or max(angular)>np.sin(PLANE_TOLERANCE_RAD):
            continue
        value=(objective(fit.x),fit.x.copy(),errors,angular)
        if best is None or value[0]<best[0]:best=value
        # Keep the feasible downward-refined solution on the first reachable branch.
        if best is not None:break
    if best is None:
        raise ValueError("识别后的目标在当前站位未找到覆盖接近、抓取和抬升的操作姿态")
    _,q,errors,angular=best
    return q[:4].tolist(),{"mode":"joint_torso_arm_ik","positions_error_m":list(map(float,errors)),
        "downward_tilt_deg":([float(np.rad2deg(np.arccos(np.clip(-t[:3,2]@up,-1,1))))
                              for t in transforms(q)] if automatic else None),
        "orientation_priority":"downward_then_joint_motion" if automatic else "explicit",
        "orientation_residual":list(map(float,angular)),"target_joint_positions_rad":q[4:].reshape(-1,len(initial)).tolist(),
        "validation":"endpoint_fk; complete joint trajectory collision checked before submission"}


def validate_fixed_poses(plan,joints,side,*,sample_path=False):
    from scipy.optimize import least_squares
    observed=dict(zip(joints["names"],joints["positions_rad"]))
    arm=ArmModel(side,[observed[f"torso_joint{i}"] for i in range(1,5)])
    low,high=arm.lower+JOINT_MARGIN_RAD,arm.upper-JOINT_MARGIN_RAD
    q=np.clip([observed[n] for n in arm.names],low,high)
    samples=[];targets=[];previous=None
    for name in ("clearance","pregrasp","grasp","lift"):
        target=getattr(plan,name)
        count=max(1,int(np.ceil(np.linalg.norm(np.array(target.position_m)-previous.position_m)/.01))) if sample_path and previous else 1
        for i in range(1,count+1):
            point=(np.asarray(previous.position_m)+(np.asarray(target.position_m)-previous.position_m)*i/count
                   if previous else target.position_m)
            targets.append((name,target.model_copy(update={"position_m":list(point)})))
        previous=target
    for name,target in targets:
        rotation=Rotation.from_quat(target.orientation_xyzw).as_matrix()
        def residual(candidate):
            t=arm.forward(candidate)
            return np.r_[t[:3,3]-target.position_m,Rotation.from_matrix(rotation.T@t[:3,:3]).as_rotvec()]
        fit=least_squares(residual,q,bounds=(low,high),max_nfev=250)
        error=residual(fit.x)
        if np.linalg.norm(error[:3])>POSITION_TOLERANCE_M or np.linalg.norm(error[3:])>PLANE_TOLERANCE_RAD:
            raise ValueError(f"实测操作姿态下 {name} 末端目标不可达")
        if sample_path and samples and np.max(np.abs(fit.x-q))>np.deg2rad(12):
            raise ValueError(f"{name} 显式姿态路径出现关节跳变")
        q=fit.x
        samples.append({"key":name,"position_m":target.position_m,"joint_positions_rad":q.tolist()})
    return samples
