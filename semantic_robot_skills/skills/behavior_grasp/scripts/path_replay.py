"""Offline replay of endpoint commands from the actual starting joint branch.

Uses the local Runtime default weighted J-PARSE and command rate. Assumes ideal
joint tracking and fixed torso/base. Collision, load and settling are unmodelled.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import ArmModel
from .native_ik import _weighted_ik_step, _jparse_compute_numpy, orientation_error
from .self_collision import require_joint_path


@dataclass(frozen=True)
class ReplaySettings:
    # Mirror atomic/backends/controller.py and Kinematics._ik_settings defaults.
    dt: float = 1/30
    timeout: float = 30.
    joint_speed: float = .5
    position_tolerance: float = .030
    orientation_tolerance: float = math.radians(4.)
    # Existing Skill path margin, with numerical FK / measured joint slack.
    minimum_margin: float = math.radians(5.)
    limit_slack: float = 1e-5


class PathReplayError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic['stage']} 从当前关节姿态不可连续到达: {diagnostic['reason']}; "
                         f"关节={diagnostic['nearest_joint']}, "
                         f"余量={diagnostic['minimum_joint_margin_deg']:.3f}°, "
                         f"位置误差={diagnostic['position_error_m']:.4f}m, "
                         f"姿态误差={diagnostic['orientation_error_deg']:.2f}°")


def replay_endpoint(arm, initial, pose, stage, settings=ReplaySettings()):
    q = np.asarray(initial, dtype=float).copy()
    if q.shape != arm.lower.shape or not np.isfinite(q).all():
        raise ValueError("连续 IK 验证需要完整有限的起始关节角")
    goal = np.eye(4)
    goal[:3, 3] = pose.position_m
    goal[:3, :3] = Rotation.from_quat(pose.orientation_xyzw).as_matrix()
    if pose.frame_id != 'body':
        raise ValueError("连续 IK 验证使用 body 目标")
    numerical_settings = (np.ones(6), .2, 100., .05)
    records = []
    worst_margin = float('inf')
    for step in range(math.ceil(settings.timeout/settings.dt)+1):
        actual = arm.forward(q)
        margins = np.minimum(q-arm.lower, arm.upper-q)
        index = int(np.argmin(margins))
        worst_margin = min(worst_margin, float(margins[index]))
        position_error = float(np.linalg.norm(actual[:3, 3]-goal[:3, 3]))
        angle_error = float(Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).magnitude())
        item = dict(stage=stage, step=step, elapsed_sim_s=step*settings.dt,
                    position_error_m=position_error, orientation_error_deg=math.degrees(angle_error),
                    minimum_joint_margin_deg=math.degrees(float(margins[index])),
                    nearest_joint=arm.names[index], joint_positions_rad=q.tolist(),
                    position_m=actual[:3, 3].tolist())
        records.append(item)
        if margins[index] < settings.minimum_margin-settings.limit_slack:
            raise PathReplayError(dict(item, reason='joint_margin', samples=records))
        if position_error <= settings.position_tolerance and angle_error <= settings.orientation_tolerance:
            return q, dict(item, worst_joint_margin_deg=math.degrees(worst_margin), samples=records)
        if step == math.ceil(settings.timeout/settings.dt):
            raise PathReplayError(dict(item, reason='not_converged', samples=records))
        error = np.r_[goal[:3, 3]-actual[:3, 3],
                      orientation_error(goal[None, :3, :3], actual[None, :3, :3])[0]]
        target, _ = _weighted_ik_step(q, arm.jacobian(q), error, arm.lower, arm.upper,
            np.ones(q.size, dtype=bool), arm.velocity, settings.dt, numerical_settings,
            lambda jac, err: _jparse_compute_numpy(jac[None])[0] @ err)
        increment = np.minimum(settings.joint_speed, arm.velocity)*settings.dt
        q = q + np.clip(target-q, -increment, increment)


def validate_endpoint_path(plan, joints, side, settings=ReplaySettings()):
    observed = dict(zip(joints['names'], joints['positions_rad']))
    arm = ArmModel(side, [observed[f'torso_joint{i}'] for i in range(1,5)])
    q = np.array([observed[name] for name in arm.names])
    segments = []
    for name in ('pregrasp', 'grasp', 'lift'):
        q, segment = replay_endpoint(arm, q, getattr(plan, name), 'grasp:'+name, settings)
        segment['self_collision'] = require_joint_path(
            {'names':list(observed),'positions_rad':list(observed.values())},arm.names,
            [p['joint_positions_rad'] for p in segment['samples']], 'grasp:'+name)
        observed.update(zip(arm.names,q))
        segments.append(segment)
    return dict(mode='native_default_ik_ideal_tracking', settings=asdict(settings), segments=segments,
                validation='sampled_native_self_collision_and_ideal_tracking; scene_load_settling_unverified')


def validate_single_endpoint(pose, joints, side, stage):
    observed=dict(zip(joints['names'],joints['positions_rad']))
    arm=ArmModel(side,[observed[f'torso_joint{i}'] for i in range(1,5)])
    _, segment=replay_endpoint(arm,np.array([observed[n] for n in arm.names]),pose,stage)
    return require_joint_path(joints,arm.names,[p['joint_positions_rad'] for p in segment['samples']],stage)
