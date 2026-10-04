"""Natural head-camera viewing geometry; computes targets without motion."""
import itertools
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.spatial.transform import Rotation
from .torso_collision import gaps as torso_gaps

LOWER = np.array([-1.1345, -2.7925, -1.8326, -3.0543])
UPPER = np.array([1.8326, 2.5307, 1.5708, 3.0543])
TORSO_CLEARANCE_M = 0.01


def torso_matrix(q):
    q1, q2, q3, yaw = q
    pitch = q1+q2-q3
    t = np.eye(4)
    t[:3, 3] = [-.079032+.4*np.sin(q1)+.3*np.sin(q1+q2)+.09962*np.sin(pitch),
                 .00000035, .34265+.4*np.cos(q1)+.3*np.cos(q1+q2)+.09962*np.cos(pitch)]
    t[:3, :3] = Rotation.from_euler('y', pitch).as_matrix() @ Rotation.from_euler('z', yaw).as_matrix()
    return t


def natural_pose(bounds_world, world_body, current, body_camera_optical, intrinsic, shape):
    bounds = np.asarray(bounds_world, dtype=float)
    current = np.asarray(current, dtype=float)
    world_body, body_camera_optical = np.asarray(world_body), np.asarray(body_camera_optical)
    k = np.asarray(intrinsic, dtype=float)
    if (bounds.shape != (2,3) or current.shape != (4,) or k.shape != (3,3)
            or world_body.shape != (4,4) or body_camera_optical.shape != (4,4)
            or any(not np.isfinite(x).all() for x in (bounds,current,k,world_body,body_camera_optical))
            or np.any(bounds[1]<bounds[0]) or min(shape)<2 or min(k[0,0],k[1,1])<=0):
        raise ValueError("观察几何、关节或标定数据无效")
    shape = np.asarray(shape)
    size = shape[::-1]-1
    mount = np.linalg.inv(torso_matrix(current)) @ body_camera_optical
    values = np.vstack([bounds.mean(0), list(itertools.product(*bounds.T))])
    points = (values-world_body[:3,3]) @ world_body[:3,:3]
    # Geometry constants above are R1Pro URDF dimensions and hard joint limits.
    # Solver tolerances / start count below control numerical precision and cost.
    length = .4 + .3 + .09962
    joint_margin = np.deg2rad(5)
    lower, upper = LOWER+joint_margin, UPPER-joint_margin
    # The 5 degree joint margin matches the existing grasp path solver.
    candidates = []
    def project(x):
        q = x
        target = points
        camera = torso_matrix(q) @ mount
        optical = (target-camera[:3,3]) @ camera[:3,:3]
        z = optical[:,2]; pixel = optical @ k.T
        uv = pixel[:,:2] / np.maximum(z[:,None], np.finfo(float).eps) / size
        line = target[0]-camera[:3,3]
        ray = math.atan2(-line[2],np.linalg.norm(line[:2]))
        return uv,z,camera,ray

    def constraints(x):
        uv,z,c,ray = project(x)
        q1,q2,q3,_ = x[:4]; pitch=q1+q2-q3
        knee_x=.4*np.sin(q1); knee_z=.34265+.4*np.cos(q1)
        waist_x=knee_x+.3*np.sin(q1+q2)
        waist_z=knee_z+.3*np.cos(q1+q2)
        return np.r_[z-np.finfo(float).eps,uv.ravel(),(1-uv).ravel(),
            knee_x,knee_x-waist_x,waist_z-knee_z,pitch,np.pi/2-pitch,c[0,2],
            torso_gaps(x)-TORSO_CLEARANCE_M]

    def cost(x):
        uv,z,c,ray = project(x)
        q1,q2,q3,_ = x[:4]
        knee_z=.34265+.4*np.cos(q1)
        # Dimensionless geometric objectives: horizontal view, centered object,
        # limited joint displacement and upright upper body.
        return (np.sin(ray)**2 + np.sum((uv[0]-.5)**2)
                + np.sum(((x-current)/(upper-lower))**2)
                + np.sin(q1+q2-q3)**2)

    bounds_opt = list(zip(lower,upper))
    # Keep the knee-forward boundary clear without changing posture cost scales.
    bounds_opt[0] = (max(lower[0], np.deg2rad(8)), upper[0])
    starts = [current, (lower+upper)/2]
    starts += list(qmc.scale(qmc.Halton(4, scramble=False).random(12),lower,upper))
    for seed in starts:
        initial=np.clip(seed,*np.asarray(bounds_opt).T)
        fit=minimize(cost,initial,method='SLSQP',bounds=bounds_opt,
            constraints={'type':'ineq','fun':constraints}, options={'maxiter':250,'ftol':1e-9})
        if np.isfinite(fit.x).all() and constraints(fit.x).min()>=-1e-6:
            candidates.append(fit.x)
    if not candidates:
        raise ValueError("没有满足目标完整入镜、膝部朝前、上半身前倾和躯干 1 cm 自碰撞间距的候选；需要调整站位")
    chosen=min(candidates,key=lambda x:(cost(x),np.linalg.norm((x[:4]-current)/(upper-lower))))
    q=chosen
    uv,z,c,ray=project(chosen)
    return {"positions_rad":q.tolist(),
        "diagnostics":{"mode":"natural_view","upper_pitch_deg":float(np.rad2deg(q[0]+q[1]-q[2])),
            "view_ray_down_deg":float(np.rad2deg(ray)),
            "target_pixel":(uv[0]*size).tolist(),"minimum_image_margin":float(np.minimum(uv,1-uv).min()),
            "visibility_check":"frustum_only","manipulation_reach_checked":False,
            "torso_clearance_required_m":TORSO_CLEARANCE_M,
            "torso_clearance_min_m":float(torso_gaps(q).min())}}
