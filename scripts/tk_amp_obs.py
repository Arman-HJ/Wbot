"""TienKung-style AMP observation (joint pos/vel + EE in root frame)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.entity import Entity

# Match TienKung-Lab adam_sp_env.get_amp_obs_for_expert_trans joint blocks.
RIGHT_ARM_JOINTS = (
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
)
LEFT_ARM_JOINTS = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
)
RIGHT_LEG_JOINTS = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)
LEFT_LEG_JOINTS = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
)
WAIST_JOINTS = ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")

HAND_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

AMP_JOINT_POS_SIZE = 29
AMP_JOINT_VEL_SIZE = 29
AMP_EE_POS_SIZE = 12
AMP_OBS_DIM = AMP_JOINT_POS_SIZE + AMP_JOINT_VEL_SIZE + AMP_EE_POS_SIZE


class TkAmpObsHelper:
  """Caches joint/body indices and builds TienKung AMP observations."""

  def __init__(self, env: ManagerBasedRlEnv, asset_name: str = "robot") -> None:
    self.env = env
    self.asset: Entity = env.scene[asset_name]
    device = env.device

    def _joint_ids(names: tuple[str, ...]) -> list[int]:
      ids, matched = self.asset.find_joints(names, preserve_order=True)
      if len(matched) != len(names):
        raise ValueError(f"Missing joints for AMP obs: expected {names}, got {matched}")
      return list(ids)

    def _body_ids(names: tuple[str, ...]) -> list[int]:
      ids, matched = self.asset.find_bodies(names, preserve_order=True)
      if len(matched) != len(names):
        raise ValueError(f"Missing bodies for AMP obs: expected {names}, got {matched}")
      return list(ids)

    self.right_arm_ids = _joint_ids(RIGHT_ARM_JOINTS)
    self.left_arm_ids = _joint_ids(LEFT_ARM_JOINTS)
    self.right_leg_ids = _joint_ids(RIGHT_LEG_JOINTS)
    self.left_leg_ids = _joint_ids(LEFT_LEG_JOINTS)
    self.waist_ids = _joint_ids(WAIST_JOINTS)
    self.hand_body_ids = _body_ids(HAND_BODIES)
    self.feet_body_ids = _body_ids(FOOT_BODIES)

    self.left_arm_local_vec = torch.tensor(
      [0.0, 0.0, -0.25], device=device, dtype=torch.float32
    )
    self.right_arm_local_vec = torch.tensor(
      [0.0, 0.0, -0.25], device=device, dtype=torch.float32
    )

  def compute(self) -> torch.Tensor:
    robot = self.asset
    root_pos = robot.data.root_link_pos_w
    root_quat = robot.data.root_link_quat_w
    body_pos = robot.data.body_link_pos_w
    body_quat = robot.data.body_link_quat_w
    n = root_pos.shape[0]

    left_hand = (
      body_pos[:, self.hand_body_ids[0], :]
      - root_pos
      + quat_apply(
        body_quat[:, self.hand_body_ids[0], :],
        self.left_arm_local_vec.expand(n, -1),
      )
    )
    right_hand = (
      body_pos[:, self.hand_body_ids[1], :]
      - root_pos
      + quat_apply(
        body_quat[:, self.hand_body_ids[1], :],
        self.right_arm_local_vec.expand(n, -1),
      )
    )
    left_hand = quat_apply_inverse(root_quat, left_hand)
    right_hand = quat_apply_inverse(root_quat, right_hand)

    left_foot = quat_apply_inverse(
      root_quat, body_pos[:, self.feet_body_ids[0], :] - root_pos
    )
    right_foot = quat_apply_inverse(
      root_quat, body_pos[:, self.feet_body_ids[1], :] - root_pos
    )

    jp = robot.data.joint_pos
    jv = robot.data.joint_vel
    return torch.cat(
      (
        jp[:, self.right_arm_ids],
        jp[:, self.left_arm_ids],
        jp[:, self.right_leg_ids],
        jp[:, self.left_leg_ids],
        jp[:, self.waist_ids],
        jv[:, self.right_arm_ids],
        jv[:, self.left_arm_ids],
        jv[:, self.right_leg_ids],
        jv[:, self.left_leg_ids],
        jv[:, self.waist_ids],
        left_hand,
        right_hand,
        left_foot,
        right_foot,
      ),
      dim=-1,
    )
