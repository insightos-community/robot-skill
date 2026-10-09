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

"""Fit the held box through explicit region geometry and compute EEF goals."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from .models import Input,Transform,Waypoint,Plan
EPS=1e-8

def matrix(p):
    t=np.eye(4);t[:3,:3]=Rotation.from_quat(p.orientation_xyzw).as_matrix();t[:3,3]=p.position_m
    return t

def pose(t):return Transform(position_m=t[:3,3].tolist(),orientation_xyzw=Rotation.from_matrix(t[:3,:3]).as_quat().tolist())
def corners(size):return np.asarray(list(itertools.product(*[(-x/2,x/2) for x in size])))
def points(t,p):return p@t[:3,:3].T+t[:3,3]

def release_rotation(current):
    """Minimal wrist rotation to horizontal or downward EEF +Z extension."""
    z=current[:,2];up=np.array([0.,0.,1.])
    horizontal=z-up*(z@up)
    if np.linalg.norm(horizontal)<EPS:
        horizontal=current[:,0]-up*(current[:,0]@up)
    horizontal/=np.linalg.norm(horizontal)
    candidates=[]
    for name,goal in (("horizontal",horizontal),("downward",-up)):
        cross=np.cross(z,goal);length=np.linalg.norm(cross)
        angle=float(np.arctan2(length,np.clip(z@goal,-1.,1.)))
        if length>EPS:axis=cross/length
        else:axis=current[:,0]
        candidates.append((angle,name,Rotation.from_rotvec(axis*angle).as_matrix()@current))
    _,name,rotation=min(candidates,key=lambda c:c[0])
    return rotation,name


def drop_plan(inputs, body_region, held, shape, local):
    """Release above an aperture without assuming an interior floor or volume."""
    opening=inputs.target.region.opening
    aperture=matrix(opening.region_from_opening)
    if not np.allclose(aperture[:3,2],[0,0,1],atol=EPS):
        raise ValueError("顶部开口法向须朝区域上方")
    release=np.linalg.inv(aperture)@local
    eef_rotation=release[:3,:3]@held[:3,:3].T
    eef_rotation,orientation=release_rotation(eef_rotation)
    release[:3,:3]=eef_rotation@held[:3,:3]
    if inputs.target.placement_pose_hint is None:
        release[:2,3]=0.
    # Retain the lateral hint while aligning the EEF horizontally/downward.
    # The aperture plane and projected object thickness set the release height.
    release[2,3]=0.
    projected=points(release,shape);gap=inputs.clearance_m
    if (np.any(np.linalg.norm(projected[:,:2],axis=1)>min(opening.extent_m)/2-gap+EPS)
            if opening.shape=="circle" else
            np.any(np.abs(projected[:,:2])>np.array(opening.extent_m)/2-gap+EPS)):
        raise ValueError("物体无法通过开口")
    release[2,3]=-projected[:,2].min()+gap
    exterior=release.copy();exterior[2,3]+=np.ptp(projected[:,2])
    release_body=body_region@aperture@release
    exterior_body=body_region@aperture@exterior
    def waypoint(key,purpose,obj):
        return Waypoint(key="place:"+key,purpose=purpose,body_from_eef=pose(obj@np.linalg.inv(held)))
    # Drop has no prescribed resting pose inside the unknown interior.
    return Plan(target_object_body=pose(release_body),release_object_body=pose(release_body),
        approach=[waypoint("preplace","clearance",exterior_body),waypoint("release-pose","place",release_body)],
        retreat=[waypoint("retreat","retreat",exterior_body)],release_orientation=orientation)


def build_plan(inputs:Input,current_eef:Transform):
    region=inputs.target.region;body_region=matrix(region.body_from_region)
    held=matrix(inputs.eef_from_object);shape=corners(inputs.object_size_m)
    local=np.linalg.inv(body_region)@matrix(current_eef)@held
    if inputs.target.placement_pose_hint is not None:local=matrix(inputs.target.placement_pose_hint)
    if inputs.release_mode=="drop":return drop_plan(inputs,body_region,held,shape,local)
    if inputs.target.placement_pose_hint is None:
        rotated=shape@local[:3,:3].T
        local[:3,3]=[0.,0.,-rotated[:,2].min()]
    box=points(local,shape);width,depth,height=region.extent_m;gap=inputs.clearance_m
    if np.any(np.abs(box[:,:2])>np.array([width,depth])/2-gap+EPS):
        raise ValueError("物体超出放置区域")
    if abs(box[:,2].min())>EPS:raise ValueError("放置终点须接触区域支撑平面")
    if inputs.target.type!="support_surface" and box[:,2].max()>height-gap+EPS:
        raise ValueError("物体超出容器内部高度")
    release=local.copy()
    if inputs.target.type=="support_surface":
        pre=local.copy();pre[2,3]+=np.ptp(box[:,2])+gap
        path=[("preplace","clearance",pre),("seat","place",release)];exits=[("retreat",pre)]
    else:
        aperture=matrix(region.opening.region_from_opening);normal=aperture[:3,2]
        if inputs.target.type=="top_open_container":
            if not np.allclose(normal,[0,0,1],atol=EPS) or abs(aperture[2,3]-height)>EPS:
                raise ValueError("顶部开口须位于区域顶部且法向朝外")
        elif abs(normal[2])>EPS:raise ValueError("侧方开口法向须水平")
        entering=local.copy()
        if inputs.target.type=="side_open_container":
            # Center the object in available vertical slack during insertion.
            slack=height-box[:,2].max()-gap
            if slack<=EPS:raise ValueError("容器缺少抬起及撤手空间")
            entering[2,3]+=slack/2
        aperture_object=np.linalg.inv(aperture)@entering
        projected=points(aperture_object,shape)
        if (np.any(np.linalg.norm(projected[:,:2],axis=1)>min(region.opening.extent_m)/2-gap+EPS)
                if region.opening.shape=="circle" else
                np.any(np.abs(projected[:,:2])>np.array(region.opening.extent_m)/2-gap+EPS)):
            raise ValueError("物体无法通过开口")
        if projected[:,2].max()>EPS:raise ValueError("目标须位于开口内侧")
        half=np.max(np.abs(shape@aperture_object[:3,:3].T),axis=0)[2]
        exterior=aperture_object.copy();exterior[2,3]=half+np.ptp(projected[:,2])+gap;exterior=aperture@exterior
        if inputs.target.type=="top_open_container":
            path=[("preplace","clearance",exterior),("seat","place",release)];exits=[("retreat",exterior)]
        else:
            path=[("preplace","clearance",exterior),("enter","insert",entering),("seat","place",release)]
            exits=[("retreat:clear",entering),("retreat:exit",exterior)]
    def waypoint(key,purpose,obj):return Waypoint(key="place:"+key,purpose=purpose,body_from_eef=pose(body_region@obj@np.linalg.inv(held)))
    return Plan(target_object_body=pose(body_region@local),release_object_body=pose(body_region@release),
        approach=[waypoint(*p) for p in path],retreat=[waypoint(k,"retreat",p) for k,p in exits])
