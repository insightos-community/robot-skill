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

"""Torso/base self-clearance in the same sphere model used by the SDK planner."""
import json,itertools
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation

@lru_cache(maxsize=1)
def model():
    root=Path(__file__).parent;cfg=json.loads((root/'r1pro_torso_collision.json').read_text())
    geometry=json.loads((root/'r1pro_collision_geometry.json').read_text())
    joints=[dict(j,a=np.array(j['local0']),b=np.linalg.inv(j['local1'])) for j in geometry['joints']]
    spheres={n:np.array([[*s['center'],s['radius']+cfg['self_collision_buffer'].get(n,0)] for s in values]) for n,values in cfg['collision_spheres'].items()}
    names=list(spheres);pairs=[(a,b) for i,a in enumerate(names) for b in names[i+1:] if b not in cfg['self_collision_ignore'].get(a,[]) and a not in cfg['self_collision_ignore'].get(b,[])]
    return joints,spheres,pairs

def gaps(q):
    joints,spheres,pairs=model();values={f'torso_joint{i+1}':v for i,v in enumerate(q)};frames={'base_link':np.eye(4)}
    for j in joints:
        if j['parent'] not in frames:continue
        m=np.eye(4)
        if j['kind']=='PhysicsRevoluteJoint':
            if j['name'] not in values:continue
            r=np.zeros(3);r['XYZ'.index(j['axis'])]=values[j['name']];m[:3,:3]=Rotation.from_rotvec(r).as_matrix()
        elif j['kind']=='PhysicsPrismaticJoint':continue
        frames[j['child']]=frames[j['parent']]@j['a']@m@j['b']
    centers={n:s[:,:3]@frames[n][:3,:3].T+frames[n][:3,3] for n,s in spheres.items()}
    return np.concatenate([(np.linalg.norm(centers[a][:,None]-centers[b][None,:],axis=2)-spheres[a][:,3,None]-spheres[b][None,:,3]).ravel() for a,b in pairs])
