# GMR retargeted motion — joint order

- Robot: `pnd_adam_sp`
- Model: `/home/songhat/code_workspace/Motion_retargeting/GMR/assets/pnd_adam_sp/scene_adam_sp.xml`
- Generated: 2026-06-26T03:45:37+00:00

## Pickle layout

| Field | Shape / type | Description |
|-------|----------------|-------------|
| `fps` | scalar | Frame rate |
| `root_pos` | (T, 3) | Pelvis position, world frame, **meters** |
| `root_rot` | (T, 4) | Pelvis orientation **xyzw** |
| `dof_pos` | (T, 29) | Hinge angles, **radians** |
| `joint_names` | list[str] | Name for each `dof_pos` column (see below) |

## `dof_pos` column order (use this when reading clips)

Columns follow **`mujoco_qpos`** order: `qpos[7:]` from the GMR MuJoCo model
(compiled hinge `qpos` addresses). This matches **MimicKit / Isaac Lab** body-tree
hinge order for `x2_ultra_simple.xml`, **not** the `<actuator><motor>` list order.

Leg indices 0–14 match motor order. Indices **15–30** differ (head vs arms).

```
   0  left_hip_pitch_joint
   1  left_hip_roll_joint
   2  left_hip_yaw_joint
   3  left_knee_joint
   4  left_ankle_pitch_joint
   5  left_ankle_roll_joint
   6  right_hip_pitch_joint
   7  right_hip_roll_joint
   8  right_hip_yaw_joint
   9  right_knee_joint
  10  right_ankle_pitch_joint
  11  right_ankle_roll_joint
  12  waist_roll_joint
  13  waist_pitch_joint
  14  waist_yaw_joint
  15  left_shoulder_pitch_joint
  16  left_shoulder_roll_joint
  17  left_shoulder_yaw_joint
  18  left_elbow_joint
  19  left_wrist_yaw_joint
  20  left_wrist_pitch_joint
  21  left_wrist_roll_joint
  22  right_shoulder_pitch_joint
  23  right_shoulder_roll_joint
  24  right_shoulder_yaw_joint
  25  right_elbow_joint
  26  right_wrist_yaw_joint
  27  right_wrist_pitch_joint
  28  right_wrist_roll_joint
```

## Motor / actuator order (Do not index `dof_pos` by this list)

Used by unitree `joint_names` in npz metadata and XML `<motor joint=...>`.
Re-map by name, not by shared index, when converting to other conventions.

```
   0  left_hip_pitch_joint
   1  left_hip_roll_joint
   2  left_hip_yaw_joint
   3  left_knee_joint
   4  left_ankle_pitch_joint
   5  left_ankle_roll_joint
   6  right_hip_pitch_joint
   7  right_hip_roll_joint
   8  right_hip_yaw_joint
   9  right_knee_joint
  10  right_ankle_pitch_joint
  11  right_ankle_roll_joint
  12  waist_roll_joint
  13  waist_pitch_joint
  14  waist_yaw_joint
  15  left_shoulder_pitch_joint
  16  left_shoulder_roll_joint
  17  left_shoulder_yaw_joint
  18  left_elbow_joint
  19  left_wrist_yaw_joint
  20  left_wrist_pitch_joint
  21  left_wrist_roll_joint
  22  right_shoulder_pitch_joint
  23  right_shoulder_roll_joint
  24  right_shoulder_yaw_joint
  25  right_elbow_joint
  26  right_wrist_yaw_joint
  27  right_wrist_pitch_joint
  28  right_wrist_roll_joint
```

## Downstream notes

- **MimicKit** (`amp_npy_to_mimickit`): use `joint_names` in the pickle, or treat
  source columns as `mujoco_qpos` / MJCF DFS — **not** motor order.
- **unitree AMP npz**: after `pkl_to_npz`, `joint_pos` columns must match
  `joint_names` (motor names); use `robot_joint_indexes` when exporting.
