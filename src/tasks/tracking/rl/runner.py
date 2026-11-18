import os
from typing import cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand

from .history_actor import HistoryEncoderMLPModel


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def export_motion_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export ``(obs, time_step) -> (actions, motion reference)`` to ONNX.

    Deployment observation contract (all float32, natural joint order, 50 Hz):

    Plain MLP actor: ``obs = actor``.

    History-encoder actor (``HistoryEncoderMLPModel``, e.g. Wbot-20Dof-Kick):
    ``obs = [actor | actor_history]``, shape ``(1, A + H*F)`` (kick: 112 + 50*66 = 3412).

    - ``actor`` (A): terms of the env ``actor`` group in order. Kick (112):
      ``command`` 40 (ref joint_pos 20 + ref joint_vel 20 at ``time_step``),
      ``motion_anchor_ori_b`` 6, ``base_ang_vel`` 3, ``projected_gravity`` 3,
      ``joint_pos`` 20 (encoder − default_joint_pos), ``joint_vel`` 20,
      ``actions`` 20 (previous raw policy output, before action_scale).
    - ``actor_history`` (H*F): time-major, oldest → newest, H frames of F values.
      Kick frame (66) = ``base_ang_vel`` 3, ``projected_gravity`` 3,
      ``joint_pos`` 20, ``joint_vel`` 20, ``actions`` 20 — same definitions as the
      matching ``actor`` terms, so the newest frame equals ``actor[46:112]``.
      Push one frame per policy step before inference; on start/reset fill all H
      frames with the first frame.

    Normalizers and the history encoder are inside the graph (feed raw values);
    the privileged decoder is training-only and not exported. Layout is also
    stored in ONNX metadata (``policy_obs_layout``, ``actor_history_*``).
    """
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def _history_obs_metadata(self) -> dict:
    """ONNX metadata describing ``obs = [actor | actor_history]`` (empty for plain MLP)."""
    policy = self.alg.get_policy()
    if not isinstance(policy, HistoryEncoderMLPModel):
      return {}
    obs_manager = self.env.unwrapped.observation_manager
    group = policy.history_obs_group
    return {
      "policy_obs_layout": f"actor({policy.obs_dim}) | {group}("
      f"{policy.history_length}x{policy.history_frame_dim}, time-major, oldest->newest)",
      "actor_obs_dim": float(policy.obs_dim),
      "actor_history_length": float(policy.history_length),
      "actor_history_frame_dim": float(policy.history_frame_dim),
      "actor_history_terms": list(obs_manager.active_terms[group]),
    }

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = policy_path.split("/")[-2] + ".onnx"
    self.export_motion_policy_to_onnx(policy_path, filename)
    self.export_policy_to_onnx(policy_path, "policy.onnx")
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    motion_term = cast(
      MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
    )
    metadata.update(
      {
        "anchor_body_name": motion_term.cfg.anchor_body_name,
        "body_names": list(motion_term.cfg.body_names),
      }
    )
    metadata.update(self._history_obs_metadata())
    attach_metadata_to_onnx(os.path.join(policy_path, filename), metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)  # type: ignore
        self.registry_name = None
