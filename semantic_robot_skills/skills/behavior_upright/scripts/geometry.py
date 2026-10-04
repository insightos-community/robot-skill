"""Fixed upright, natural carry and init goals derived from R1Pro geometry."""
import json,math
from pathlib import Path
INITIAL_TORSO_RAD=(1.025,-1.45,-.47,0.)
VERIFY_TOLERANCE_RAD=.02

def fixed_posture(holding_side):
    target={f"torso_joint{i+1}":v for i,v in enumerate(INITIAL_TORSO_RAD)}
    carry=json.loads(Path(__file__).with_name("carry_postures.json").read_text())
    pitch=INITIAL_TORSO_RAD[0]+INITIAL_TORSO_RAD[1]-INITIAL_TORSO_RAD[2]
    for side in ("left","right"):
        if side==holding_side:target.update(carry[side])
        else:target.update({f"{side}_arm_joint{i+1}":v for i,v in enumerate(
            [0.,.6 if side=="left" else -.6,0.,-(math.pi/2+pitch),0.,0.,0.])})
    return target

def holding(tools):
    by_side={t['side']:t for t in tools}
    if set(by_side)!={'left','right'} or len(tools)!=2 or any('held_object_ref' not in t for t in tools):
        raise ValueError('需要双手原生持物身份反馈')
    selected=[(side,t['held_object_ref']) for side,t in by_side.items() if t['held_object_ref']]
    if len(selected)>1:raise ValueError('双手均持物，请先明确恢复目标')
    return selected[0] if selected else (None,None)
