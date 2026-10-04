"""Task-supplied placement geometry; normal body-frame execution."""
from typing import Annotated, Literal
from pydantic import BaseModel,ConfigDict,Field,field_validator,model_validator
from semantic_robot_skill_sdk import ActionResult
Vector=Annotated[list[float],Field(min_length=3,max_length=3)]
class Model(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
class Transform(Model):
    position_m:Vector
    orientation_xyzw:list[float]=Field(default_factory=lambda:[0.,0.,0.,1.],min_length=4,max_length=4)
    @field_validator("orientation_xyzw")
    @classmethod
    def unit(cls,value):
        norm=sum(x*x for x in value)**.5
        if norm<1e-8:raise ValueError("四元数不能为零")
        return [x/norm for x in value]
class Opening(Model):
    shape:Literal["rectangle","circle"]="rectangle"
    # Local XY spans the aperture; local +Z points out of the container.
    region_from_opening:Transform
    extent_m:list[float]=Field(min_length=2,max_length=2)
    @field_validator("extent_m")
    @classmethod
    def positive(cls,v):
        if min(v)<=0:raise ValueError("开口尺寸须为正")
        return v
class Region(Model):
    body_from_region:Transform
    extent_m:Vector|None=None
    opening:Opening|None=None
class Target(Model):
    object_ref:str=Field(min_length=1)
    type:Literal["support_surface","top_open_container","side_open_container"]
    region:Region|None=None
    prompt:str|None=Field(default=None,min_length=1)
    perception_profile:str|None=None
    placement_pose_hint:Transform|None=None
    @model_validator(mode="after")
    def geometry(self):
        if self.region is None:return self
        size=self.region.extent_m
        if size is not None and (min(size[:2])<=0 or size[2]<0):raise ValueError("区域尺寸无效")
        if self.type=="support_surface":
            if self.region.opening is not None:raise ValueError("支撑面不使用开口")
        elif self.region.opening is None:raise ValueError("容器需要开口几何")
        elif size is not None and size[2]<=0:raise ValueError("容器内部高度须为正")
        return self
    @property
    def relation(self):return "on" if self.type=="support_surface" else "inside"
class Input(Model):
    object_ref:str=Field(min_length=1)
    side:Literal["left","right","auto"]="auto"
    object_size_m:Vector|None=None
    eef_from_object:Transform|None=None
    prompt:str|None=Field(default=None,min_length=1)
    perception_profile:str|None=None
    minimum_confidence:float=Field(default=.3,ge=0,le=1)
    target:Target
    release_mode:Literal["place","drop"]="place"
    clearance_m:float=Field(default=0.,ge=0)
    release_opening_m:float=Field(gt=0)
    maximum_force_n:float=Field(gt=0)
    timeout_seconds:float=Field(default=180,gt=0)
    @model_validator(mode="after")
    def valid(self):
        if (self.object_size_m is None)!=(self.eef_from_object is None):
            raise ValueError("物体尺寸及抓持变换须一起提供，或一起通过识别获取")
        if self.object_size_m is None and self.prompt is None:
            raise ValueError("自动获取持物几何需要 prompt")
        if self.object_size_m is not None and min(self.object_size_m)<=0:raise ValueError("物体尺寸须为正")
        if self.object_ref==self.target.object_ref:raise ValueError("物体和目标须不同")
        if self.release_mode=="drop" and self.target.type!="top_open_container":raise ValueError("drop 需要顶部开口")
        if self.target.region is None and (self.release_mode!="drop" or self.target.prompt is None):
            raise ValueError("省略目标区域时需要顶部 drop 模式及 target.prompt")
        if self.release_mode=="place" and self.target.region.extent_m is None:
            raise ValueError("place 模式需要区域尺寸；容器须提供内部可用尺寸")
        return self
class Waypoint(Model):
    key:str
    purpose:Literal["clearance","insert","place","retreat"]
    body_from_eef:Transform
class Plan(Model):
    target_object_body:Transform
    release_object_body:Transform
    approach:list[Waypoint]
    retreat:list[Waypoint]
    release_orientation:Literal["horizontal","downward"]|None=None
class HeldGeometry(Model):
    object_size_m:Vector
    eef_from_object:Transform
    source:Literal["task_input","visible_rgbd_bbox"]
    identity_confidence:float|None=None

class State(Model):
    hand_selection:dict|None=None
    release_verification:dict|None=None
    stage:str="plan"
    plan:Plan|None=None
    object_geometry:HeldGeometry|None=None
    target_geometry:dict|None=None
    results:dict[str,ActionResult]=Field(default_factory=dict)
    evidence_refs:list[str]=Field(default_factory=list)
class Result(Model):
    holding_side:Literal["left","right"]
    hand_selection:dict
    object_ref:str
    target_ref:str
    requested_relation:Literal["on","inside"]
    release_mode:Literal["place","drop"]
    target_object_body:Transform
    release_object_body:Transform
    object_geometry:HeldGeometry|None=None
    target_geometry:dict|None=None
    release_orientation:Literal["horizontal","downward"]|None=None
    release_verification:dict
    verification_level:Literal["release_and_retreat_motion"]="release_and_retreat_motion"
    evidence_refs:list[str]=Field(default_factory=list)
