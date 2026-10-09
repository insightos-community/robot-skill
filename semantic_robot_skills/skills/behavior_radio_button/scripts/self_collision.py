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

"""Sampled native robot self-collision checks for Skill-planned joint paths.

Native collision mesh pieces are convexified offline. Fixed assemblies and
joint-adjacent bodies are excluded; other arm/body and arm/arm pairs are checked.
Finger meshes cover the complete stroke. Scene and held-object geometry are
outside this model. No simulator or robot command is created by this module.
"""
import json
from functools import lru_cache
from pathlib import Path

import fcl
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

SAMPLE_ANGLE_RAD = np.deg2rad(.5)
PENETRATION_EPSILON_M = 1e-5


class SelfCollisionError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        pair = diagnostic['links']
        super().__init__(f"{diagnostic['stage']} 自碰撞: {pair[0]} / {pair[1]}，"
                         f"穿入约 {diagnostic['penetration_m']*1000:.2f} mm")


def joint_map(joints):
    names, values = joints['names'], joints['positions_rad']
    if len(names) != len(values) or len(set(names)) != len(names) or not np.isfinite(values).all():
        raise ValueError('自碰撞检查需要完整有限且名称唯一的关节状态')
    return dict(zip(names, values))


def convex(vertices, faces=None):
    vertices = np.asarray(vertices, dtype=float)
    if faces is None:
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        for i, face in enumerate(faces):
            a,b,c=vertices[face]
            if np.dot(np.cross(b-a,c-a),hull.equations[i,:3])<0:faces[i]=face[::-1]
    else:
        faces = np.asarray(faces,dtype=np.int32)
    polygons=np.column_stack([np.full(len(faces),3),faces]).ravel().astype(np.int32)
    return fcl.Convex(vertices,len(faces),polygons)


class RobotCollision:
    def __init__(self):
        data=json.loads(Path(__file__).with_name('r1pro_collision_geometry.json').read_text())
        self.source_sha256=data['source_sha256']
        self.joints=[dict(j,a=np.array(j['local0']),b=np.linalg.inv(j['local1'])) for j in data['joints']]
        self.moving=[j['name'] for j in self.joints if j['kind']=='PhysicsRevoluteJoint']
        groups={name:name for name in data['links']}
        def group(name):
            while groups[name]!=name:name=groups[name]
            return name
        for j in self.joints:
            if j['kind']=='PhysicsFixedJoint':groups[group(j['child'])]=group(j['parent'])
        groups={n:group(n) for n in groups}
        adjacent={frozenset([groups[j['parent']],groups[j['child']]]) for j in self.joints}
        self.pieces={n:[convex(m['vertices'],m['faces']) for m in meshes] for n,meshes in data['links'].items() if meshes}
        self.broad={n:convex(np.concatenate([m['vertices'] for m in data['links'][n]])) for n in self.pieces}
        names=list(self.pieces)
        closing_pairs={frozenset([side+'_gripper_finger_link1',side+'_gripper_finger_link2']) for side in ('left','right')}
        disabled={frozenset(pair) for pair in data['disabled_collision_pairs']}
        self.pairs=[(a,b) for i,a in enumerate(names) for b in names[i+1:]
            if groups[a]!=groups[b] and frozenset([groups[a],groups[b]]) not in adjacent
            and frozenset([a,b]) not in closing_pairs|disabled]
        self.collision_request=fcl.CollisionRequest()
        self.distance_request=fcl.DistanceRequest(enable_signed_distance=True)

    def fk(self, values):
        missing=set(self.moving)-values.keys()
        if missing:raise ValueError('自碰撞检查缺少关节: '+', '.join(sorted(missing)))
        ts={'base_link':np.eye(4)}
        for j in self.joints:
            motion=np.eye(4)
            if j['kind']=='PhysicsRevoluteJoint':
                axis=np.zeros(3);axis['XYZ'.index(j['axis'])]=values[j['name']]
                motion[:3,:3]=Rotation.from_rotvec(axis).as_matrix()
            # Finger stroke is already included in each collision shape.
            ts[j['child']]=ts[j['parent']]@j['a']@motion@j['b']
        return ts

    @lru_cache(maxsize=512)
    def _collision(self, positions):
        ts=self.fk(dict(zip(self.moving,positions)))
        transforms={n:fcl.Transform(ts[n][:3,:3],ts[n][:3,3]) for n in self.broad}
        broad={n:fcl.CollisionObject(shape,transforms[n]) for n,shape in self.broad.items()}
        pieces={}
        for a,b in self.pairs:
            if not fcl.collide(broad[a],broad[b],self.collision_request,fcl.CollisionResult()):continue
            for n in (a,b):
                if n not in pieces:pieces[n]=[fcl.CollisionObject(shape,transforms[n]) for shape in self.pieces[n]]
            for left in pieces[a]:
                for right in pieces[b]:
                    if not fcl.collide(left,right,self.collision_request,fcl.CollisionResult()):continue
                    distance=fcl.distance(left,right,self.distance_request,fcl.DistanceResult())
                    if distance < -PENETRATION_EPSILON_M:
                        return {'links':[a,b],'penetration_m':float(-distance)}
        return None

    def check(self, values, stage, sample_index):
        missing=set(self.moving)-values.keys()
        if missing:raise ValueError('自碰撞检查缺少关节: '+', '.join(sorted(missing)))
        q=tuple(float(values[n]) for n in self.moving)
        if not np.isfinite(q).all():raise ValueError('自碰撞检查遇到非有限关节角')
        collision=self._collision(q)
        if collision:
            raise SelfCollisionError(dict(collision,stage=stage,sample_index=sample_index,
                joint_positions_rad=dict(zip(self.moving,q)),geometry_sha256=self.source_sha256))

    def distances(self, values, pairs):
        ts=self.fk(values)
        transforms={n:fcl.Transform(ts[n][:3,:3],ts[n][:3,3]) for n in self.broad}
        broad={n:fcl.CollisionObject(shape,transforms[n]) for n,shape in self.broad.items()}
        pieces={};result=[]
        for a,b in pairs:
            distance=fcl.distance(broad[a],broad[b],self.distance_request,fcl.DistanceResult())
            if distance<.025:
                for n in (a,b):
                    if n not in pieces:pieces[n]=[fcl.CollisionObject(shape,transforms[n]) for shape in self.pieces[n]]
                distance=min(fcl.distance(x,y,self.distance_request,fcl.DistanceResult()) for x in pieces[a] for y in pieces[b])
            result.append(distance)
        return np.asarray(result)


@lru_cache(maxsize=1)
def robot_collision():
    return RobotCollision()


def require_joint_path(joints, names, positions, stage):
    """Check every supplied station and interpolate each gap at <= 0.5 deg."""
    values=joint_map(joints);model=robot_collision()
    if len(set(names))!=len(names) or any(n not in values for n in names):
        raise ValueError('自碰撞路径关节名与当前状态不匹配')
    previous=np.array([values[n] for n in names],dtype=float);sample=0
    model.check(values,stage,sample)
    for row in positions:
        target=np.asarray(row,dtype=float)
        if target.shape!=previous.shape or not np.isfinite(target).all():raise ValueError('自碰撞路径目标维度错误')
        count=max(1,int(np.ceil(np.max(np.abs(target-previous))/SAMPLE_ANGLE_RAD)))
        for point in np.linspace(previous,target,count+1)[1:]:
            sample+=1;values.update(zip(names,point));model.check(values,stage,sample)
        previous=target
    return {'validation':'sampled_native_robot_self_collision','samples':sample+1,
            'sample_angle_rad':float(SAMPLE_ANGLE_RAD),'geometry_sha256':model.source_sha256}


def require_transition(joints, names, target, stage):
    return require_joint_path(joints,names,[target],stage)


def require_retraction(joints, plan):
    values=joint_map(joints);reports=[]
    def state():return {'names':list(values),'positions_rad':list(values.values())}
    for arm in plan['arms']:
        reports.append(require_transition(state(),arm['joint_names'],arm['positions_rad'],'grasp:retract:'+arm['arm_side']))
        values.update(zip(arm['joint_names'],arm['positions_rad']))
    names=[f'torso_joint{i}' for i in range(1,5)]
    reports.append(require_transition(state(),names,plan['torso_target_rad'],'grasp:operation-torso'))
    return reports
