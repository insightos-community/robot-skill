"""Target enclosure from observed OBB and native R1Pro gripper geometry.

Finger/target overlap is permitted. Palm, wrist and camera remain excluded.
This is a geometric precondition for closing, not contact or grasp confirmation.
"""
from functools import lru_cache
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .kinematics import torso_in_body

EPS = 1e-9  # numerical geometry tolerance, metres


class EnclosureError(ValueError):
    def __init__(self, diagnostic, phase):
        self.diagnostic = diagnostic
        self.phase = phase
        super().__init__("夹爪包围关系未满足：" + "；".join(diagnostic["failures"]))


def transform(pose):
    p = np.asarray(pose["position_m"], dtype=float)
    q = np.asarray(pose["orientation_xyzw"], dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
        raise ValueError("包围检查需要有限的位置和四元数")
    if not np.isclose(np.linalg.norm(q), 1., atol=1e-5, rtol=0):
        raise ValueError("包围检查需要单位四元数")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q).as_matrix()
    t[:3, 3] = p
    return t


def directions(v):
    norm = np.linalg.norm(v, axis=1)
    v = v[norm > EPS] / norm[norm > EPS, None]
    index = np.argmax(np.abs(v), axis=1)
    v *= np.where(v[np.arange(len(v)), index] < 0, -1., 1.)[:, None]
    return np.unique(np.round(v, 10), axis=0)


@lru_cache(maxsize=2)
def hand_model(side):
    if side not in ("left", "right"):
        raise ValueError("包围检查需要已选定的手侧")
    data = json.loads(Path(__file__).with_name("r1pro_gripper_geometry.json").read_text())
    hand = data["hands"][side]
    meshes = []
    for part in hand["nonfinger_parts"]:
        for piece in part["parts"]:
            vertices = np.asarray(piece["vertices"])
            hull = ConvexHull(vertices)
            tri = vertices[hull.simplices]
            edges = directions(np.concatenate([tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 1], tri[:, 0]-tri[:, 2]]))
            meshes.append((part["link"], vertices, directions(hull.equations[:, :3]), edges))
    return hand, meshes


def polygon(points):
    points = np.unique(np.asarray(points), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points-points[0], tol=EPS) < 2:
        return np.empty((0, 2))
    return points[ConvexHull(points).vertices]


def cross2(a, b):
    return a[0]*b[1]-a[1]*b[0]


def overlap_area(subject, boundary):
    """Clip two convex X/Z polygons; projections retain the observed OBB rotation."""
    result = list(subject)
    for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
        output = []
        if not result:
            return 0.
        previous = result[-1]
        before = cross2(b-a, previous-a)
        for current in result:
            after = cross2(b-a, current-a)
            if (after >= -EPS) != (before >= -EPS):
                output.append(previous + (current-previous)*before/(before-after))
            if after >= -EPS:
                output.append(current)
            previous, before = current, after
        result = output
    if len(result) < 3:
        return 0.
    points = np.asarray(result)
    return abs(float(np.sum(points[:, 0]*np.roll(points[:, 1], -1)-points[:, 1]*np.roll(points[:, 0], -1))))/2


def mesh_box_gap(mesh, box):
    _, vertices, normals, edges = mesh
    rotation, center, half = box
    crosses = np.cross(edges[:, None, :], rotation.T[None, :, :]).reshape(-1, 3)
    norm = np.linalg.norm(crosses, axis=1)
    crosses = crosses[norm > EPS] / norm[norm > EPS, None]
    axes = np.concatenate([rotation.T, normals, crosses])
    projection = (vertices-center) @ axes.T
    radius = np.abs(axes @ rotation) @ half
    return float(np.maximum(projection.min(0)-radius, -radius-projection.max(0)).max())


def enclosure(located, pose, side, positions):
    """Evaluate a planned or measured EEF pose, in the same body frame as recognition."""
    target = located.model_dump(mode="json") if hasattr(located, "model_dump") else located
    goal = pose.model_dump(mode="json") if hasattr(pose, "model_dump") else pose
    if target["pose"]["frame_id"] != "body" or goal["frame_id"] != "body":
        raise ValueError("包围检查需要 body 坐标的目标及末端位姿")
    extent = np.asarray(target["extent_m"], dtype=float)
    values = np.asarray(positions, dtype=float)
    if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("包围检查需要观测包围盒的三个正尺寸")
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("包围检查需要两根手指各自的实际或计划行程")
    hand, meshes = hand_model(side)
    object_eef = np.linalg.inv(transform(goal)) @ transform(target["pose"])
    rotation, center, half = object_eef[:3, :3], object_eef[:3, 3], extent/2
    corners = (np.asarray(list(product((-1., 1.), repeat=3))) * half) @ rotation.T + center
    target_projection = polygon(corners[:, [0, 2]])
    failures, fingers = [], []
    for finger, value in zip(hand["fingers"], values):
        if not finger["closed"]-EPS <= value <= finger["opened"]+EPS:
            raise ValueError(f"{finger['joint_name']} 行程超出模型范围")
        normal = np.asarray(finger["contact_face"]["plane"][:3])
        offset = float(finger["contact_face"]["plane"][3])
        axis = np.asarray(finger["axis"])
        margin = float(np.min(corners @ normal + offset - normal @ axis * value))
        closed_margin = float(np.min(corners @ normal + offset - normal @ axis * finger["closed"]))
        face = np.asarray(finger["contact_face"]["vertices"]) + value * axis
        patch = polygon(face[:, [0, 2]])
        area = overlap_area(target_projection, patch)
        if margin < -EPS:
            failures.append(f"{finger['link']} 内侧间距不足，目标超出 {abs(margin)*1000:.2f} mm")
        if closed_margin > EPS:
            failures.append(f"{finger['link']} 在模型闭合行程内无法接近目标")
        if area <= EPS*EPS:
            failures.append(f"{finger['link']} 有效夹持面与目标无投影重叠")
        fingers.append(dict(link=finger["link"], position_m=float(value),
            inner_clearance_m=margin, closed_travel_margin_m=closed_margin, projected_overlap_m2=area))
    gaps = {}
    for mesh in meshes:
        gap = mesh_box_gap(mesh, (rotation, center, half))
        name = mesh[0]
        gaps[name] = min(gaps.get(name, float('inf')), gap)
    for name, gap in gaps.items():
        if gap <= EPS:
            failures.append(f"{name} 与目标包围盒相交")
    return dict(valid=not failures, object_ref=target["object_ref"], side=side,
        end_effector_pose_body={k:goal[k] for k in ("frame_id","position_m","orientation_xyzw")},
        geometry_source="recognized_visible_obb_and_native_gripper_collision_geometry",
        fingers=fingers, nonfinger_separation_m=gaps, failures=failures,
        verification_level="geometric_closure_precondition")


def require_enclosure(located, pose, side, positions, phase="planned"):
    diagnostic = enclosure(located, pose, side, positions)
    if not diagnostic["valid"]:
        raise EnclosureError(diagnostic, phase)
    return diagnostic


def measured_pose_in_body(output, measured, side):
    entry = output["end_effector"]
    if entry["axis"] != side:
        raise ValueError("末端回读与抓取手不一致")
    t = transform(entry)
    frame = entry["frame_id"]
    if frame == "torso_link4":
        joints = dict(zip(measured["joints"]["names"], measured["joints"]["positions_rad"]))
        t = torso_in_body([joints[f'torso_joint{i}'] for i in range(1, 5)]) @ t
    elif frame not in ("body", "base", "base_link", "base_footprint"):
        base = measured["base_pose"]
        if frame != base["frame_id"]:
            raise ValueError(f"末端回读坐标系 {frame} 缺少 body 变换")
        t = np.linalg.inv(transform(dict(position_m=base["position"], orientation_xyzw=base["orientation_xyzw"]))) @ t
    return dict(frame_id="body", position_m=t[:3, 3].tolist(), orientation_xyzw=Rotation.from_matrix(t[:3, :3]).as_quat().tolist())


def measured_fingers(output, side):
    rows = [row for row in output["tools"] if row["side"] == side]
    if len(rows) != 1:
        raise ValueError("需要抓取手唯一的两指行程回读")
    values = np.asarray(rows[0]["positions_m"], dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("实际两指行程缺失或无效")
    return values.tolist()
