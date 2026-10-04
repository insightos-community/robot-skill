"""Compute a retreat along EEF extension from observed OBB and open finger meshes."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from .enclosure import hand_model,transform
from .compact_spheres import compact_mesh_spheres


def offset_for(located,grasp,side,opening_m):
    relative=np.linalg.inv(transform(grasp.model_dump(mode='json'))) @ transform(located.pose.model_dump(mode='json'))
    corners=np.array(list(product((-1.,1.),repeat=3)))*np.asarray(located.extent_m)/2
    corners=corners@relative[:3,:3].T+relative[:3,3]
    hand,_=hand_model(side);front=[]
    # Same compact cover and 2 mm world buffer as the SDK planner.
    for finger in hand['fingers']:
        for part in finger['parts']:
            vertices=np.asarray(part['vertices'])+np.asarray(finger['axis'])*opening_m
            front.extend(sphere['center'][2]+sphere['radius']+.002
                         for sphere in compact_mesh_spheres(vertices))
    distance=max(0.,max(front)-corners[:,2].min())+.002
    direction=Rotation.from_quat(grasp.orientation_xyzw).apply([0.,0.,1.])
    return (-direction*distance).tolist(),{'distance_m':float(distance),'source':'recognized_obb_and_open_finger_geometry'}
