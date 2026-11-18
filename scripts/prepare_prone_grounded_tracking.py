"""Build a 50 Hz full-body tracking reference with an already-supported start.

This is a reference for policy learning, not a certified dynamic get-up rollout.
No drop/settling simulation is prepended. Original models/motions are untouched.
"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
OTHER = ROOT.parent / "DroidUpE1_mjlab"
sys.path.insert(0, str(OTHER / "tools/amp"))
from audit_prone_mesh import MeshGeometry, load_mesh_model, motion_qpos

AUDIT = OTHER / "dataset/wbot_20dof/amp/roban_getup/getup/prone_1_physics_audit"
OUT = ROOT / "src/assets/motions/wbot_20dof/prone_grounded_start"
XML = ROOT / "src/assets/robots/wbot/xmls/wbot_20dof.xml"


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*x*(10+x*(-15+6*x))


def mix(a,b,w):
    pose = (1-w)*a+w*b
    rots = Rotation.from_quat(np.stack([a[3:7],b[3:7]]),scalar_first=True)
    pose[3:7] = Slerp([0,1],rots)(w).as_quat(scalar_first=True)
    return pose


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source = AUDIT / "mesh_repaired/prone_1_MESH_REPAIRED_GEOMETRY_ONLY.npz"
    z = dict(np.load(source))
    model = load_mesh_model(XML)
    geometry = MeshGeometry(model)
    d = mujoco.MjData(model)
    original = motion_qpos(model,z)
    initial = np.load(AUDIT / "support_v3/hold_validation/refined_prone.npz")["qpos"][-1]
    final = np.load(AUDIT / "support_v3/hold_validation/redesigned_stand.npz")["qpos"][-1]
    # Keep the same world origin as the reference for both endpoint holds.
    final[:2] += original[-1,:2]-final[:2]
    dt=.02; delay=.20; duration=(len(original)-1)/float(z["fps"])
    timeline=np.arange(round((duration+delay+1.)/dt)+1)*dt
    poses=[]
    for t in timeline:
        source_t=np.clip(t-delay,0,duration)
        x=source_t*float(z["fps"]);i=min(int(x),len(original)-2)
        pose=mix(original[i],original[i+1],x-i)
        if source_t<2.:
            pose=mix(initial,pose,float(smooth(source_t/2.)))
        if source_t>duration-1.2:
            pose=mix(pose,final,float(smooth((source_t-duration+1.2)/1.2)))
        d.qpos[:]=pose;mujoco.mj_kinematics(model,d)
        pose[2]-=float(geometry.floor_heights(d).min())
        poses.append(pose)
    poses=np.asarray(poses)
    for i in range(1,len(poses)):
        if poses[i,3:7]@poses[i-1,3:7]<0:poses[i,3:7]*=-1
    names=z["joint_names"]
    adr=[model.jnt_qposadr[model.joint(str(n)).id] for n in names]
    body_ids=list(range(1,model.nbody))
    bodies=np.array([model.body(i).name for i in body_ids])
    data={"fps":np.array(50.),"joint_names":names,"body_names":bodies,
          "joint_pos":poses[:,adr].astype(np.float32)}
    data["joint_vel"]=np.gradient(data["joint_pos"],dt,axis=0,edge_order=2)
    bp=[];bq=[];minimum=[];self_pen=[]
    for pose in poses:
        d.qpos[:]=pose;mujoco.mj_forward(model,d)
        bp.append(d.xpos[body_ids].copy());bq.append(d.xquat[body_ids].copy())
        minimum.append(float(geometry.floor_heights(d).min()))
        self_pen.append(min([float(c.dist) for c in d.contact]+[0.]))
    data["body_pos_w"]=np.asarray(bp,dtype=np.float32)
    data["body_quat_w"]=np.asarray(bq,dtype=np.float32)
    data["body_lin_vel_w"]=np.gradient(data["body_pos_w"],dt,axis=0,edge_order=2)
    angular=np.zeros((len(poses),len(body_ids),3))
    for j in range(len(body_ids)):
        rotations=Rotation.from_quat(data["body_quat_w"][:,j],scalar_first=True)
        edges=(rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt
        angular[:,j]=np.vstack((edges[0],(edges[:-1]+edges[1:])/2,edges[-1]))
    data["body_ang_vel_w"]=angular.astype(np.float32)
    for key in ["joint_vel","body_lin_vel_w","body_ang_vel_w"]:
        data[key][0]=0;data[key][-1]=0
    path=OUT / "prone_1_grounded_start_tracking.npz"
    np.savez_compressed(path,**data)
    np.save(OUT / "reference_qpos.npy",poses)
    report={"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "output":str(path),"output_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "frames":len(poses),"fps":50,"body_count":len(body_ids),"duration_s":float(timeline[-1]),
            "first_frame_source":"Last pose of independently validated 3 s prone support hold",
            "prepended_free_fall_frames":0,"initial_supported_hold_s":delay,
            "first_frame_velocities_zero":True,"maximum_source_frame_ground_penetration_m":max(0,-min(minimum)),
            "primitive_self_penetration_m":max(0,-min(self_pen)),
            "max_joint_step_rad":float(np.max(np.abs(np.diff(poses[:,adr],axis=0)))),
            "dynamic_getup_certified":False,"purpose":"Experimental full-body imitation training reference; learned getup must be tested"}
    (OUT/"preparation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
