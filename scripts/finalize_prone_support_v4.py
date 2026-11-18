"""Bound hip CAD clearance and retime the separately optimized v4 keyposes."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco
from scipy.ndimage import maximum_filter1d, gaussian_filter1d
from scipy.spatial.transform import Rotation
from prepare_prone_support_v4 import ROOT, OUT, XML, SOURCE, mix, export, load_mesh_model, MeshGeometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keys",type=Path,default=OUT/"attempt_c/optimized_keyposes.npz")
    p.add_argument("--out",type=Path,default=OUT/"final_candidate")
    p.add_argument("--max-joint-speed",type=float,default=1.3)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    z=np.load(args.keys)
    keys=z["qpos"].copy();indices=z["source_frames"]
    names=np.load(SOURCE/"prone_1_grounded_start_tracking.npz")["joint_names"]
    m=load_mesh_model(XML);g=MeshGeometry(m);d=mujoco.MjData(m)
    key_delta=np.max(np.abs(np.diff(keys[:,7:],axis=0)),axis=1)
    r=Rotation.from_quat(keys[:,3:7],scalar_first=True)
    angle=(r[1:]*r[:-1].inv()).magnitude()
    intervals=np.maximum.reduce([np.diff(indices)*.02*1.7,key_delta/args.max_joint_speed,angle/1.0])
    # Each key timestamp lands on a 50 Hz sample. The geometric path is preserved.
    intervals=np.ceil(intervals/.02)*.02
    times=np.r_[0.,np.cumsum(intervals)]
    duration=times[-1]
    poses=[]
    for t in np.arange(round((duration+1.5)/.02)+1)*.02:
        k=min(max(0,np.searchsorted(times,t)-1),len(keys)-2)
        w=np.clip((t-times[k])/(times[k+1]-times[k]),0,1)
        poses.append(mix(keys[k],keys[k+1],w))
    poses=np.asarray(poses)
    corrections=[]
    pelvis=int(g.geoms[g.names.index("pelvis")])
    for side,sign in [("left",1.),("right",-1.)]:
        adr=m.jnt_qposadr[m.joint(side+"_hip_roll_joint").id]
        hip=int(g.geoms[g.names.index(side+"_hip_roll_link")])
        required=np.zeros(len(poses))
        def gap(q,offset):
            d.qpos[:]=q;d.qpos[adr]+=sign*offset
            mujoco.mj_kinematics(m,d)
            return mujoco.mj_geomDistance(m,d,pelvis,hip,.001,None)
        for i,q in enumerate(poses):
            if gap(q,0)>=.0006:continue
            low,high=0.,.05
            if gap(q,high)<.0006:raise RuntimeError(f"{side} frame {i} exceeds small hip correction")
            for _ in range(20):
                mid=(low+high)/2
                if gap(q,mid)<.0006:low=mid
                else:high=mid
            required[i]=high
        # A local upper envelope followed by smoothing maintains clearance without
        # abrupt per-frame corrections. Preserve the first supported pose exactly.
        offset=gaussian_filter1d(maximum_filter1d(required,size=31,mode="nearest"),sigma=5)
        offset=np.maximum(offset,required+.001*(required>0))
        offset[:10]=0
        poses[:,adr]+=sign*offset
        corrections.append({"joint":side+"_hip_roll_joint","maximum_offset_rad":float(offset.max())})
    for q in poses:
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        q[2]-=g.floor_heights(d).min()
    report=export(m,g,poses,names,args.out,"prone_1_support_v4_tracking")
    report.update({"source_keyposes":str(args.keys),"method":"Continuation contact IK, bounded hip CAD clearance correction, adaptive timing with <=1.3 rad/s keyframe slope and 1.5 s final hold", "keyframe_max_joint_speed_limit_rad_s":args.max_joint_speed,"corrections":corrections,"prepended_free_fall_frames":0,"dynamic_getup_certified":False})
    (args.out/"preparation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
