"""Resolve a visible held-object box in the same frame as the measured EEF."""
import numpy as np
from typing import Literal
from pydantic import Field
from .models import Model,Transform,Vector,HeldGeometry,Region,Opening
from .geometry import matrix,pose,corners,points


class ObservedPose(Transform):
    frame_id:Literal["body"]
    revision:str=Field(min_length=1)
    observed_at:str|None=None


class LocatedObject(Model):
    object_ref:str
    pose:ObservedPose
    extent_m:Vector
    identity_confidence:float=Field(ge=0,le=1)


def held_geometry(inputs, verification, current_eef):
    located=LocatedObject.model_validate(verification)
    if located.object_ref!=inputs.object_ref:raise ValueError("识别物体与持物目标不匹配")
    if located.identity_confidence<inputs.minimum_confidence:raise ValueError("持物识别置信度不足")
    if min(located.extent_m)<=0:raise ValueError("持物点云不足以估计三维包围盒")
    # Size and orientation come from the same observed box coordinate system.
    relative=np.linalg.inv(matrix(current_eef))@matrix(located.pose)
    return HeldGeometry(object_size_m=located.extent_m,eef_from_object=pose(relative),
        source="visible_rgbd_bbox",identity_confidence=located.identity_confidence)


def rim_top_region(inputs, output):
    """Fit a horizontal circular rim from measured points; explicit regions retain other shapes."""
    from scipy.optimize import least_squares
    located=LocatedObject.model_validate(output["verification"])
    if located.object_ref!=inputs.target.object_ref:raise ValueError("识别容器与投放目标不匹配")
    if located.identity_confidence<inputs.minimum_confidence:raise ValueError("容器识别置信度不足")
    samples=output.get("visible_upper_surface")
    if not samples or samples.get("frame_id")!="body":raise ValueError("识别结果缺少桶沿采样点，需要支持上缘点云的识别能力")
    xyz=np.asarray(samples["points_m"],dtype=float)
    up=np.asarray(samples["up_direction"],dtype=float)
    pixel=float(samples["pixel_footprint_m"]);height=float(samples["upper_height_m"])
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<6 or not np.isfinite(xyz).all()
            or up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-8
            or not np.isfinite(pixel) or pixel<=0 or not np.isfinite(height)):
        raise ValueError("桶沿采样无效")
    up/=np.linalg.norm(up)
    axis=np.eye(3)[np.argmin(np.abs(up))];x=axis-up*(axis@up);x/=np.linalg.norm(x)
    axes=np.column_stack([x,np.cross(up,x),up]);cloud=xyz@axes
    fits=[]
    for band in (1.,2.,3.):
        q=cloud[cloud[:,2]>=height-band*pixel,:2]
        if len(q)<6:raise ValueError("可见桶沿采样不足，请提供开口区域或调整观察位置")
        design=np.column_stack([2*q,np.ones(len(q))])
        seed,_,rank,_=np.linalg.lstsq(design,np.sum(q*q,axis=1),rcond=None)
        radius2=seed[2]+seed[:2]@seed[:2]
        if rank<3 or radius2<=0:raise ValueError("桶沿无法确定圆形开口")
        fit=least_squares(lambda v:np.linalg.norm(q-v[:2],axis=1)-v[2],
            np.r_[seed[:2],np.sqrt(radius2)],loss="soft_l1",f_scale=pixel)
        angles=np.sort(np.mod(np.arctan2(q[:,1]-fit.x[1],q[:,0]-fit.x[0]),2*np.pi))
        arc=2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))
        rms=float(np.sqrt(np.mean(fit.fun**2)))
        if not fit.success or fit.x[2]<=pixel or rms>pixel or arc<np.pi:
            raise ValueError("可见上缘未通过圆形开口拟合，请提供开口区域或调整观察位置")
        fits.append({"center":fit.x[:2],"radius":float(fit.x[2]),"rms_m":rms,
                     "arc_rad":float(arc),"points":len(q),"band_pixels":band})
    selected=fits[1];spread=max(float(np.linalg.norm(f["center"]-selected["center"])) for f in fits)
    if spread>pixel:raise ValueError("桶沿中心估计不稳定，请提供开口区域或调整观察位置")
    # One image-space sampling footprint is retained as a boundary margin.
    radius=min(f["radius"] for f in fits)-pixel
    top=np.eye(4);top[:3,:3]=axes;top[:3,3]=axes@np.r_[selected["center"],height]
    region=Region(body_from_region=pose(top),opening=Opening(shape="circle",
        region_from_opening=Transform(position_m=[0,0,0]),extent_m=[2*radius,2*radius]))
    return region,{"source":"visible_circular_rim","region":region.model_dump(mode="json"),
        "pixel_footprint_m":pixel,"center_spread_m":spread,"fit_rms_m":selected["rms_m"],
        "visible_arc_rad":selected["arc_rad"],"rim_radius_m":selected["radius"],
        "point_count":selected["points"],"unobserved_wall_thickness":True}
