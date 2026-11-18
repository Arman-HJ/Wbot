# WBOT 20DoF VelStyle motions (paper protocol)

Paper: **Driving the Style** (arXiv:2611.20888).

## How the paper builds motion data

1. **Capture**: authors record human motions with a mocap system (private; not released).
2. **Retarget**: [GMR](https://github.com/YanjieZe/GMR) maps human traj → robot kinematics (joint limits).
3. **Per clip**: full-body reference at every frame  
   `joint_pos/vel` + `body_{pos,quat,lin_vel,ang_vel}_w` (world).  
   **Pelvis** = anchor / base frame.
4. **Five semantic clips**:

| Role | Paper use |
| --- | --- |
| `random_walk` | Main train clip for generalization (Table IV/VI) |
| `backward` | In-dist + held-out eval |
| `side` | In-dist + held-out eval |
| `slow` | In-dist + held-out eval |
| `jog` | In-dist only (not held-out in Table IV) |

**Generalization protocol**: train **only** on `random_walk`; at eval, feed the actor the **velocity profile** of a held-out clip (`backward` / `side` / `slow`) and score against that clip’s joint reference.

Ablations:

- **B1**: also train on left–right mirrored `random_walk`.
- **S (IV-F)**: Savitzky–Golay (window 100) on base position + frame-to-frame base rotation; joints unchanged.

## WBOT stand-in (this folder)

Original Agile Robots mocap is unavailable. We map existing WBOT AMP locomotion clips to the same **roles**:

```bash
python scripts/prepare_vel_style_paper_motions.py --also-mirror --also-smoothed
```

| Paper role | WBOT source | Notes |
| --- | --- | --- |
| `random_walk` | `walk_run_amp/walk.npz` | Direct; multi-direction planar vel |
| `side` | `side_l.npz` | Direct |
| `jog` | `run.npz` | Direct |
| `backward` | time-reverse `walk_straight_a` | **Approximation** |
| `slow` | 2× stretch `walk_straight_b` | **Approximation** |
| `random_walk_mirror` | `walk_mirror.npz` | Paper B1 |
| `random_walk_smoothed` | SG on random_walk base | Paper IV-F |

Manifest: `dataset.yaml`.

## Train (paper generalization)

```bash
./scripts/train_tracking_vel_style_wbot_20dof.sh
# uses random_walk_tracking.npz by default
```
