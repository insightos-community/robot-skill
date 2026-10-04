"""Anchor the observed OBB, then express it in the measured closed-gripper frame."""
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .enclosure import transform
from .models import HeldGeometry, ObservationAnchor, RigidTransform


def _timestamp(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("抓持几何的观测时间需要时区")
    return parsed.timestamp()


def _base_sample(result):
    readings = [o for o in result.observations if o.kind == "robot.state"]
    if len(readings) != 1 or not readings[0].value:
        raise ValueError("识别锚定需要唯一的底盘状态观测")
    reading = readings[0]
    base = reading.value["base_pose"]
    pose = dict(position_m=base["position"], orientation_xyzw=base["orientation_xyzw"])
    return base["frame_id"], transform(pose), _timestamp(reading.observed_at)


def observation_anchor(located, before, after):
    """Bracket RGBD with base readings; only interpolate a stationary base.

    The Skill does not command base motion during perception. Small measured
    drift is interpolated at the RGBD timestamp; moving or stale data is rejected.
    """
    frame, a, start = _base_sample(before)
    end_frame, b, end = _base_sample(after)
    if frame not in ("world", "odom") or frame != end_frame:
        raise ValueError("识别前后底盘需要一致的 world 或 odom 坐标系")
    if located.pose.frame_id != "body" or located.pose.observed_at is None:
        raise ValueError("识别结果需要带时间的 body 位姿")
    observed = _timestamp(located.pose.observed_at)
    if not start <= observed <= end or end <= start:
        raise ValueError("RGBD 时间不在底盘采样区间内，不能复用过期识别")
    translation = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    angle = float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
    if translation > .001 or angle > np.deg2rad(.1):
        raise ValueError("识别期间底盘未保持静止，不能确定抓持几何")
    fraction = (observed - start) / (end - start)
    rotation = Slerp([0., 1.], Rotation.from_matrix(np.array([a[:3, :3], b[:3, :3]])))(fraction)
    return ObservationAnchor(
        frame_id=frame, observed_at=located.pose.observed_at,
        fixed_from_body=RigidTransform(
            position_m=((1 - fraction) * a[:3, 3] + fraction * b[:3, 3]).tolist(),
            orientation_xyzw=rotation.as_quat().tolist()),
        base_translation_drift_m=translation, base_rotation_drift_rad=angle)


def held_geometry(located, anchor, measured_eef, object_ref):
    if located.object_ref != object_ref:
        raise ValueError("抓持几何物体与当前抓取目标不一致")
    if measured_eef["frame_id"] != anchor.frame_id:
        raise ValueError("识别锚定与闭合后夹爪位姿的固定坐标系不一致")
    if measured_eef.get("observed_at") and _timestamp(measured_eef["observed_at"]) < _timestamp(anchor.observed_at):
        raise ValueError("闭合后夹爪观测早于物体识别，不能使用过期位姿")
    fixed_object = transform(anchor.fixed_from_body.model_dump()) @ transform(located.pose.model_dump())
    relative = np.linalg.inv(transform(measured_eef)) @ fixed_object
    return HeldGeometry(
        object_size_m=located.extent_m,
        eef_from_object=RigidTransform(position_m=relative[:3, 3].tolist(),
            orientation_xyzw=Rotation.from_matrix(relative[:3, :3]).as_quat().tolist()),
        fixed_frame_id=anchor.frame_id, recognition_observed_at=anchor.observed_at,
        eef_observed_at=measured_eef.get("observed_at"),
        recognition_revision=located.pose.revision, identity_confidence=located.identity_confidence)


def result_fields(state):
    """Legacy checkpoints remain readable without inventing missing geometry."""
    geometry = state.held_geometry
    return dict(object_size_m=geometry.object_size_m if geometry else None,
                eef_from_object=geometry.eef_from_object if geometry else None,
                held_geometry=geometry)
