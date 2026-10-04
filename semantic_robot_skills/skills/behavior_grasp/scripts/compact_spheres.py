"""Conservative oriented-box slab cover for convex collision mesh parts."""
import numpy as np

MESH_MARGIN_M = .001


def compact_mesh_spheres(vertices):
    """Cover an entire mesh OBB with spheres along its longest dimension.

    Each sphere circumscribes one OBB slab. Slab length is at most the second
    longest OBB dimension, so the count follows geometry without a volume grid.
    The union contains the whole OBB, including triangle interiors.
    """
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Collision vertices must be a finite nonempty Nx3 array')
    origin = points.mean(axis=0)
    centered = points-origin
    _, basis = np.linalg.eigh(centered.T @ centered)
    basis = basis[:, ::-1]
    local = centered @ basis
    lower, upper = local.min(axis=0), local.max(axis=0)
    extent = upper-lower
    axis = int(np.argmax(extent))
    cross = max(float(np.sort(extent)[-2]), 2*MESH_MARGIN_M)
    count = max(1, int(np.ceil(extent[axis]/cross)))
    width = extent.copy()
    width[axis] /= count
    radius = float(np.linalg.norm(width/2)+MESH_MARGIN_M)
    spheres = []
    for index in range(count):
        center = (lower+upper)/2
        center[axis] = lower[axis]+(index+.5)*width[axis]
        spheres.append(dict(center=(origin+basis@center).tolist(), radius=radius))
    return spheres



# Palm subdivision caps the enclosing radius at 22 mm before the mesh margin.
# Clipping includes all hull/plane intersections, so each convex subvolume is covered.
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _enclosing_ball(v):
 origin=v.mean(0);scale=max(np.ptp(v,axis=0).max(),1e-6);p=(v-origin)/scale
 start=np.r_[np.zeros(3),np.linalg.norm(p,axis=1).max()]
 def fun(x):return x[3]
 def cons(x):return x[3]**2-np.sum((p-x[:3])**2,axis=1)
 def jac(x):return np.column_stack([2*(p-x[:3]),np.full(len(p),2*x[3])])
 fit=minimize(fun,start,jac=lambda x:np.array([0,0,0,1.]),constraints={'type':'ineq','fun':cons,'jac':jac},bounds=[(None,None)]*3+[(0,None)],method='SLSQP',options={'ftol':1e-10,'maxiter':60})
 c=origin+fit.x[:3]*scale;return c,float(np.linalg.norm(v-c,axis=1).max())

def palm_mesh_spheres(v, limit=.022):
 v=np.array(v);c,r=_enclosing_ball(v)
 if r<=limit+1e-9:return [dict(center=c.tolist(),radius=r+MESH_MARGIN_M)]
 hull=ConvexHull(v);v=v[hull.vertices];h=ConvexHull(v);edges=np.unique(np.sort(np.vstack([h.simplices[:,[0,1]],h.simplices[:,[1,2]],h.simplices[:,[2,0]]]),axis=1),axis=0)
 axis=np.argmax(np.ptp(v,axis=0));mid=(v[:,axis].min()+v[:,axis].max())/2
 a,b=v[edges[:,0]],v[edges[:,1]];mask=(a[:,axis]-mid)*(b[:,axis]-mid)<0;a,b=a[mask],b[mask];cross=a+(b-a)*((mid-a[:,axis])/(b[:,axis]-a[:,axis]))[:,None]
 return palm_mesh_spheres(np.vstack([v[v[:,axis]<=mid],cross]),limit)+palm_mesh_spheres(np.vstack([v[v[:,axis]>=mid],cross]),limit)
