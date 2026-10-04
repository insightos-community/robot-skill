"""Task-independent grasp axes estimated from the observed whole-object cloud."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

DOWNWARD_EPSILON = 1e-5
TILT_COST_SLACK = 1-np.cos(np.deg2rad(.5))


def prefer_downward(seed, bounds, constraints, tilt_cost, secondary_cost, maxiter=250):
    """Lexicographic local optimization: downward alignment, then joint movement.

    The existing half-degree numerical angular allowance supplies cost slack.
    Feasibility is checked before choosing a secondary solution.
    """
    def feasible(x, checks):
        if not np.isfinite(x).all(): return False
        if any(value<lo-1e-7 or value>hi+1e-7 for value,(lo,hi) in zip(x,bounds)): return False
        for item in checks:
            value=np.atleast_1d(item["fun"](x))
            if not np.isfinite(value).all(): return False
            if item["type"]=="eq" and np.max(np.abs(value))>1e-6: return False
            if item["type"]=="ineq" and value.min() < -1e-7: return False
        return True

    options={"maxiter":maxiter,"ftol":1e-10}
    primary=minimize(tilt_cost,seed,method="SLSQP",bounds=bounds,constraints=constraints,options=options)
    if not feasible(primary.x,constraints):
        fallback=minimize(secondary_cost,seed,method="SLSQP",bounds=bounds,constraints=constraints,options=options)
        if not feasible(fallback.x,constraints): return fallback
        primary=minimize(tilt_cost,fallback.x,method="SLSQP",bounds=bounds,constraints=constraints,options=options)
        if not feasible(primary.x,constraints): primary=fallback
    ceiling=float(tilt_cost(primary.x))+TILT_COST_SLACK
    locked=[*constraints,{"type":"ineq","fun":lambda x:ceiling-tilt_cost(x)}]
    secondary=minimize(secondary_cost,primary.x,method="SLSQP",bounds=bounds,constraints=locked,options=options)
    return secondary if feasible(secondary.x,locked) else primary


def axes(principal_axis, orientation):
    if (not isinstance(principal_axis, dict) or principal_axis.get("frame_id") != "body"
            or principal_axis.get("source") != "visible_rgbd_pca"
            or principal_axis.get("reliable") is not True):
        raise ValueError("本次视觉点云无法可靠确定整体主轴，请调整观察位置")
    axis = np.asarray(principal_axis.get("direction"), dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError("整体主轴方向无效")
    up = Rotation.from_quat(orientation.body_orientation_xyzw).inv().apply([0., 0., 1.])
    return axis/np.linalg.norm(axis), up


def seed_quaternion(axis, up, side):
    # Prefer a downward tool. If the major axis is vertical, yaw stays free.
    x = axis-up*np.dot(axis, up)
    if np.linalg.norm(x) < 1e-8:
        candidate = np.eye(3)[np.argmin(np.abs(up))]
        x = candidate-up*np.dot(candidate, up)
    x /= np.linalg.norm(x)
    z = -up; y = np.cross(z, x)
    q = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat().tolist()
    return q
