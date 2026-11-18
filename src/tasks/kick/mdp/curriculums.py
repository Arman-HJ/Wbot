"""Kick curricula: synced push, terminations, and RSI magnitude."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class PushTerminationStage(TypedDict, total=False):
  step: int
  velocity_range: dict[str, tuple[float, float]]
  anchor_pos_threshold: float
  anchor_ori_threshold: float
  ee_body_pos_threshold: float
  # RSI on motion command (applied at resample / reset).
  rsi_pose_range: dict[str, tuple[float, float]]
  rsi_velocity_range: dict[str, tuple[float, float]]
  rsi_joint_position_range: tuple[float, float]


class push_termination_curriculum:
  """Sync push magnitude, tracking terminations, and RSI ranges.

  Interval of ``push_robot`` is left unchanged. Stages are keyed by
  ``env.common_step_counter`` (control steps = iteration * steps_per_env).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    stages: list[PushTerminationStage] = cfg.params["stages"]
    if not stages:
      raise ValueError("push_termination_curriculum requires non-empty stages")
    for i in range(1, len(stages)):
      if stages[i]["step"] < stages[i - 1]["step"]:
        raise ValueError("Curriculum stages must be in nondecreasing step order")
    self._stages = stages
    self._push_cfg = env.event_manager.get_term_cfg("push_robot")
    self._anchor_pos_cfg = env.termination_manager.get_term_cfg("anchor_pos")
    self._anchor_ori_cfg = env.termination_manager.get_term_cfg("anchor_ori")
    self._ee_cfg = env.termination_manager.get_term_cfg("ee_body_pos")
    self._motion = env.command_manager.get_term("motion")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stages: list[PushTerminationStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, stages
    applied: PushTerminationStage | dict[str, Any] = {}
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        applied = stage
      else:
        break

    if "velocity_range" in applied:
      self._push_cfg.params["velocity_range"] = applied["velocity_range"]
    if "anchor_pos_threshold" in applied:
      self._anchor_pos_cfg.params["threshold"] = float(applied["anchor_pos_threshold"])
    if "anchor_ori_threshold" in applied:
      self._anchor_ori_cfg.params["threshold"] = float(applied["anchor_ori_threshold"])
    if "ee_body_pos_threshold" in applied:
      self._ee_cfg.params["threshold"] = float(applied["ee_body_pos_threshold"])

    motion_cfg = self._motion.cfg
    if "rsi_pose_range" in applied:
      motion_cfg.pose_range = dict(applied["rsi_pose_range"])
    if "rsi_velocity_range" in applied:
      motion_cfg.velocity_range = dict(applied["rsi_velocity_range"])
    if "rsi_joint_position_range" in applied:
      motion_cfg.joint_position_range = tuple(applied["rsi_joint_position_range"])

    vr = self._push_cfg.params["velocity_range"]
    pose = motion_cfg.pose_range or {}
    jpr = motion_cfg.joint_position_range
    return {
      "step": torch.tensor(float(env.common_step_counter)),
      "push_x_max": torch.tensor(float(vr["x"][1])),
      "push_yaw_max": torch.tensor(float(vr["yaw"][1])),
      "anchor_pos_threshold": torch.tensor(
        float(self._anchor_pos_cfg.params["threshold"])
      ),
      "anchor_ori_threshold": torch.tensor(
        float(self._anchor_ori_cfg.params["threshold"])
      ),
      "ee_body_pos_threshold": torch.tensor(float(self._ee_cfg.params["threshold"])),
      "rsi_yaw_max": torch.tensor(float(pose.get("yaw", (0.0, 0.0))[1])),
      "rsi_joint_max": torch.tensor(float(jpr[1] if jpr is not None else 0.0)),
    }
