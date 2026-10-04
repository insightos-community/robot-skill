"""wxnicr handle alignment in world coordinates, expressed in the Robot body.

Calibration: radio collision mesh 6 is long along object +Y. Native EEF +Z
extends out of the fingers; EEF +Y is the opening axis. EEF +X follows the
horizontal handle direction. Wrist camera orientation remains free.
No numerical or engine dependency is needed in the standalone Skill.
"""
from math import sqrt

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def unit(v):
    length = sqrt(dot(v, v))
    if length < 1e-8:
        raise ValueError("把手长轴的水平投影无法确定抓取朝向")
    return [x / length for x in v]


def rotate(q, v):
    # Unit quaternion xyzw, active local-to-parent rotation.
    t = [2*x for x in cross(q[:3], v)]
    c = cross(q[:3], t)
    return [v[i] + q[3]*t[i] + c[i] for i in range(3)]


def inverse(q):
    return [-q[0], -q[1], -q[2], q[3]]


def basis_quaternion(x, y, z):
    m = [[x[i], y[i], z[i]] for i in range(3)]
    trace = sum(m[i][i] for i in range(3))
    if trace > 0:
        s = 2 * sqrt(1 + trace)
        q = [(m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s, s/4]
    else:
        i = max(range(3), key=lambda k: m[k][k])
        j, k = (i+1)%3, (i+2)%3
        s = 2 * sqrt(1 + m[i][i] - m[j][j] - m[k][k])
        q = [0., 0., 0., (m[k][j]-m[j][k])/s]
        q[i], q[j], q[k] = s/4, (m[j][i]+m[i][j])/s, (m[k][i]+m[i][k])/s
    q = unit(q)
    return [-v for v in q] if q[3] < 0 else q


def automatic_grasp_geometry(orientation, grasp_offset, pregrasp_offset, lift_offset):
    object_world = orientation.object_orientation_xyzw
    world_body = inverse(orientation.body_orientation_xyzw)
    handle = rotate(object_world, [0., 1., 0.])
    x_world = unit([handle[0], handle[1], 0.])
    z_world = [0., 0., -1.]
    y_world = cross(z_world, x_world)
    quaternion = basis_quaternion(*(rotate(world_body, axis) for axis in (x_world, y_world, z_world)))
    # Grasp offset is object-local; pregrasp/lift vectors use world axes.
    return (quaternion, rotate(world_body, rotate(object_world, grasp_offset)),
            rotate(world_body, pregrasp_offset), rotate(world_body, lift_offset))
