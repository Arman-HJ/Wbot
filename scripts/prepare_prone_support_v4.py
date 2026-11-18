"""Experimental support-aware continuation IK; original data and XML are read only.

Floor extrema use actual meshes. The output must pass separate geometry and
contact-wrench checks; IK success is never treated as dynamic certification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
OTHER = ROOT.parent / "DroidUpE1_mjlab"
sys.path.insert(0, str(OTHER / "tools/amp"))
from audit_prone_mesh import load_mesh_model, MeshGeometry

SOURCE = ROOT / "src/assets/motions/wbot_20dof/prone_grounded_start"
OUT = ROOT / "src/assets/motions/wbot_20dof/prone_support_v4"
XML = ROOT / "src/assets/robots/wbot/xmls/wbot_20dof.xml"


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10+x*(-15+6*x))


def mix(a, b, w):
    q = (1-w)*a+w*b
    q[3:7] = Slerp([0, 1], Rotation.from_quat([a[3:7], b[3:7]], scalar_first=True))(w).as_quat(scalar_first=True)
    return q


def export(model, geometry, poses, names, out, stem):
    dt = .02
    adr = [model.jnt_qposadr[model.joint(str(n)).id] for n in names]
    d = mujoco.MjData(model)
    ids = list(range(1, model.nbody))
    bp, bq, contacts, lows = [], [], [], []
    support_names = ["left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_link", "right_elbow_link", "left_knee_link", "right_knee_link"]
    support_ids = [geometry.names.index(n) for n in support_names]
    for pose in poses:
        d.qpos[:] = pose
        mujoco.mj_forward(model, d)
        heights = geometry.floor_heights(d)
        bp.append(d.xpos[ids].copy())
        bq.append(d.xquat[ids].copy())
        contacts.append(heights[support_ids] <= .002)
        lows.append(heights.min())
    bp, bq = np.asarray(bp), np.asarray(bq)
    angular = np.zeros((len(poses), len(ids), 3))
    for j in range(len(ids)):
        r = Rotation.from_quat(bq[:, j], scalar_first=True)
        edges = (r[1:]*r[:-1].inv()).as_rotvec()/dt
        angular[:, j] = np.vstack((edges[0], (edges[:-1]+edges[1:])/2, edges[-1]))
    z = dict(fps=np.array(50.), joint_names=names, body_names=np.array([model.body(i).name for i in ids]),
             joint_pos=poses[:, adr].astype(np.float32), joint_vel=np.gradient(poses[:, adr], dt, axis=0, edge_order=2).astype(np.float32),
             body_pos_w=bp.astype(np.float32), body_quat_w=bq.astype(np.float32),
             body_lin_vel_w=np.gradient(bp, dt, axis=0, edge_order=2).astype(np.float32), body_ang_vel_w=angular.astype(np.float32))
    for k in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        z[k][0] = 0
        z[k][-1] = 0
    np.savez_compressed(out / f"{stem}.npz", **z)
    np.save(out / "reference_qpos.npy", poses)
    np.savez_compressed(out / "support_annotations.npz", body_names=support_names, contact_flags=np.asarray(contacts), minimum_floor_height_m=lows)
    return {"frames":len(poses), "duration_s":(len(poses)-1)*dt, "maximum_floor_penetration_m":max(0., -min(lows)),
            "max_joint_step_rad":float(np.abs(np.diff(poses[:,adr], axis=0)).max()), "output_sha256":hashlib.sha256((out/f"{stem}.npz").read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--nfev", type=int, default=80)
    p.add_argument("--duration-scale", type=float, default=1.5)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--reuse", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = np.load(SOURCE / "reference_qpos.npy")
    names = np.load(SOURCE / "prone_1_grounded_start_tracking.npz")["joint_names"]
    m = load_mesh_model(XML)
    d = mujoco.MjData(m)
    geom = MeshGeometry(m)
    jids = [m.joint(str(n)).id for n in names]
    adr = m.jnt_qposadr[jids]
    lo, hi = m.jnt_range[jids].T
    feet = [geom.names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    hands = [geom.names.index(n) for n in ("left_elbow_link", "right_elbow_link")]
    knees = [geom.names.index(n) for n in ("left_knee_link", "right_knee_link")]
    foot_bodies = [m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    # Keep the final posture facing the motion heading, removing the original
    # endpoint blend's unnecessary ~90 degree yaw rotation while standing up.
    end = source[-1].copy()
    yaw = Rotation.from_euler("z", 1.65)
    end[3:7] = (yaw*Rotation.from_quat(end[3:7], scalar_first=True)).as_quat(scalar_first=True)
    refs = source.copy()
    for i in range(300, len(refs)):
        refs[i] = mix(refs[i], end, smooth((i-300)/100))
    indices = sorted(set([0, 10, 400] + list(range(10, 401, args.stride))))
    results, stats = [], []
    if args.reuse:
        z = np.load(args.out / "optimized_keyposes.npz")
        results, indices = z["qpos"], z["source_frames"]
    else:
        prev = refs[0].copy()
        for frame in indices:
            if frame <= 10:
                results.append(source[0].copy())
                prev = source[0].copy()
                continue
            ref = refs[frame]
            base = Rotation.from_quat(ref[3:7], scalar_first=True)
            delta = (base.inv()*Rotation.from_quat(prev[3:7], scalar_first=True)).as_rotvec()
            x0 = np.r_[prev[2], delta, prev[adr]]
            hand_weight = 1.-smooth((frame-155)/90.)
            foot_flat = smooth((frame-130)/110.)
            foot_weight = smooth((frame-10)/45.)
            # Continuation starts from the preceding solution, avoiding IK branch
            # jumps. Its regularizer is deliberately stronger than pose imitation.
            def forward(x):
                d.qpos[:] = ref
                d.qpos[2] = x[0]
                d.qpos[3:7] = (base*Rotation.from_rotvec(x[1:4])).as_quat(scalar_first=True)
                d.qpos[adr] = x[4:]
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)
                heights, pts = [], []
                for g, vertices in zip(geom.geoms, geom.vertices):
                    R = d.geom_xmat[g].reshape(3, 3)
                    world = vertices@R.T+d.geom_xpos[g]
                    h = world[:,2].min()
                    heights.append(h)
                    # The COM residual uses the projected hull; restricting this
                    # to vertices within a hard z threshold creates a discontinuous
                    # objective and can freeze finite-difference IK. Independent
                    # wrench audit still uses only real vertices within 2 mm.
                    pts.append(world)
                return np.array(heights), pts

            def residual(x):
                heights, pts = forward(x)
                active = feet+(hands if hand_weight>.5 else [])
                xy = np.concatenate([pts[i][:,:2] for i in active])
                if len(xy) >= 3:
                    try:
                        hull = ConvexHull(xy)
                        violation = np.maximum(hull.equations[:,:2]@d.subtree_com[1,:2]+hull.equations[:,2]+.006, 0).max()
                    except Exception:
                        violation = np.linalg.norm(d.subtree_com[1,:2]-xy.mean(axis=0))
                else:
                    violation = np.linalg.norm(d.subtree_com[1,:2]-xy.mean(axis=0))
                self_depth = np.zeros(m.ngeom)
                for c in d.contact:
                    self_depth[c.geom1] = min(self_depth[c.geom1], c.dist)
                    self_depth[c.geom2] = min(self_depth[c.geom2], c.dist)
                normals = np.concatenate([d.xmat[b].reshape(3,3)[2,:2] for b in foot_bodies])
                # Hip/pelvis CAD overlap is excluded in the source XML. Explicit
                # close-distance constraints cover it in this experiment.
                mesh_gap = geom.self_distances(d)
                target_weight = .1 + smooth((frame-210)/110.)*.8
                r = [x[1:4]*target_weight, (x[4:]-ref[adr])*target_weight,
                     (x[4:]-prev[adr])*.12, (x[1:4]-delta)*.12,
                     np.minimum(heights,0)*1800., heights[feet]*900.*foot_weight,
                     heights[hands]*500.*hand_weight, normals*foot_flat*15.,
                     self_depth*350., np.minimum(mesh_gap,0)*300., [violation*80.]]
                # Do not lift knees abruptly during the first support transfer.
                r.append(heights[knees]*50.*(1.-smooth((frame-10)/45.)))
                return np.concatenate(r)

            lower = np.r_[.02, [-1.7]*3, lo+1e-5]
            upper = np.r_[.8, [1.7]*3, hi-1e-5]
            opt = least_squares(residual, np.clip(x0,lower+1e-7,upper-1e-7), bounds=(lower,upper), max_nfev=args.nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7, x_scale=np.r_[.2,np.ones(23)])
            heights, pts = forward(opt.x)
            pose = d.qpos.copy()
            pose[2] -= heights.min()
            results.append(pose)
            prev = pose.copy()
            row = {"frame":frame, "cost":float(opt.cost), "nfev":opt.nfev, "feet_height_m":(heights[feet]-heights.min()).tolist(), "hand_height_m":(heights[hands]-heights.min()).tolist(), "minimum_before_projection_m":float(heights.min())}
            stats.append(row)
            if frame%25 == 0:
                print(json.dumps(row), flush=True)
                np.savez_compressed(args.out/"optimized_keyposes_progress.npz", qpos=results, source_frames=indices[:len(results)])
        results = np.asarray(results)
        np.savez_compressed(args.out/"optimized_keyposes.npz", qpos=results, source_frames=indices)
        (args.out/"optimization.json").write_text(json.dumps(stats, indent=2))
    # Slow the transition while keeping the loader at 50 Hz; 1.5 s final hold.
    duration = 8.*args.duration_scale
    poses = []
    for t in np.arange(round((duration+1.5)/.02)+1)*.02:
        f = min(t/args.duration_scale*50, 400.)
        k = min(max(0,np.searchsorted(indices,f)-1),len(indices)-2)
        w = np.clip((f-indices[k])/(indices[k+1]-indices[k]),0,1)
        q = mix(results[k],results[k+1],w)
        d.qpos[:] = q
        mujoco.mj_kinematics(m,d)
        q[2] -= geom.floor_heights(d).min()
        poses.append(q)
    poses=np.asarray(poses)
    report=export(m, geom, poses, names, args.out, "prone_1_support_v4_tracking")
    report.update({"source":str(SOURCE), "duration_scale":args.duration_scale, "dynamic_getup_certified":False, "prepended_free_fall_frames":0, "method":"Continuation IK: exact mesh floor, two-foot/hand contact schedule, flattened soles, COM support-polygon residual, primitive and convex mesh self-clearance. Independent audits required."})
    (args.out/"preparation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
