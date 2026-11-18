# Wbot-20Dof-Tracking-VelStyle

Scaffold of *Driving the Style* (arXiv:2611.20888) on WBOT 20DoF.

| Piece | Behavior |
| --- | --- |
| Rewards | BeyondMimic dense whole-body tracking (no AMP disc) |
| Actor `command` | `(vx, vy, wz)` from reference pelvis (yaw frame) |
| Actor obs | command · anchor ori · ang vel · projected gravity · joints · last action |
| Critic | privileged body / EE + same command (3-D vel) |
| Curriculum | standing-start body-reward ramp (`p=0.3`, stab 30, trans 60) |
| Runner | plain `MjlabOnPolicyRunner` (MLP, no history encoder) |

Existing Kick / e1_mimic / AMP tasks are untouched; this package is additive.

## Motions (paper protocol)

See `src/assets/motions/wbot_20dof/vel_style/README.md` and `dataset.yaml`.

```bash
python scripts/prepare_vel_style_paper_motions.py --also-mirror --also-smoothed
```

Default train clip = `random_walk_tracking.npz` (paper generalization).

## Train

```bash
./scripts/train_tracking_vel_style_wbot_20dof.sh

# or
python scripts/train.py Wbot-20Dof-Tracking-VelStyle \
  --motion-file src/assets/motions/wbot_20dof/vel_style/random_walk_tracking.npz \
  --env.scene.num-envs=4096
```

## Play

```bash
python scripts/play.py Wbot-20Dof-Tracking-VelStyle --viewer wasm --run <run>
```

## Still TODO (beyond scaffold)

- Held-out eval driven by another clip's velocity profile
- Gamepad deployment path (command from sticks, upright yaw target)
- Optional Distill baseline (Teacher → student with same actor obs)
- Per-env reward scale (curriculum currently uses mean scale globally)
