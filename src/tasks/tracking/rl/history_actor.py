"""Actor with long-history encoder + privileged decoder (football / HumanoidKick style).

Actor input = ``cat(normalize(obs[actor]), encoder(normalize(obs[actor_history])))``.
The decoder reconstructs ``obs[privileged]`` from the history embedding; its MSE is
added to the PPO loss by :class:`PPOWithReconstruction`. The decoder is training-only
and is not exported.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState


class HistoryEncoderMLPModel(MLPModel):
  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    history_obs_group: str = "actor_history",
    privileged_obs_group: str = "privileged",
    history_length: int = 50,
    encoder_hidden_dims: tuple[int, ...] | list[int] = (1024, 128),
    embedding_dim: int = 64,
    decoder_hidden_dims: tuple[int, ...] | list[int] = (128, 128),
  ) -> None:
    self.embedding_dim = embedding_dim
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )
    self.history_obs_group = history_obs_group
    self.privileged_obs_group = privileged_obs_group
    self.history_length = history_length
    history_dim = obs[history_obs_group].shape[-1]
    if history_dim % history_length != 0:
      raise ValueError(
        f"'{history_obs_group}' dim {history_dim} is not divisible by "
        f"history_length={history_length}."
      )
    self.history_frame_dim = history_dim // history_length
    self.privileged_dim = obs[privileged_obs_group].shape[-1]

    if obs_normalization:
      self.history_normalizer = EmpiricalNormalization(self.history_frame_dim)
    else:
      self.history_normalizer = nn.Identity()
    self.encoder = MLP(history_dim, embedding_dim, encoder_hidden_dims, activation)
    self.decoder = MLP(
      embedding_dim, self.privileged_dim, decoder_hidden_dims, activation
    )
    self.privileged_estimate: torch.Tensor | None = None

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self.embedding_dim

  def encode_history(self, history: torch.Tensor) -> torch.Tensor:
    frames = history.reshape(-1, self.history_length, self.history_frame_dim)
    frames = self.history_normalizer(frames)
    return self.encoder(frames.flatten(start_dim=1))

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    latent = super().get_latent(obs, masks, hidden_state)
    embedding = self.encode_history(obs[self.history_obs_group])
    self.privileged_estimate = self.decoder(embedding)
    return torch.cat((latent, embedding), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    super().update_normalization(obs)
    if self.obs_normalization:
      newest = obs[self.history_obs_group][..., -self.history_frame_dim :]
      self.history_normalizer.update(newest)  # type: ignore[operator]

  def as_jit(self) -> nn.Module:
    return _ExportHistoryMLPModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _ExportHistoryMLPModel(self, verbose)


class _ExportHistoryMLPModel(nn.Module):
  """Deterministic actor for export; decoder dropped.

  Single flat input ``obs = [actor (A) | actor_history (H*F)]``; see
  ``MotionTrackingOnPolicyRunner.export_motion_policy_to_onnx`` for the layout.
  """

  is_recurrent: bool = False

  def __init__(self, model: HistoryEncoderMLPModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_normalizer = copy.deepcopy(model.history_normalizer)
    self.encoder = copy.deepcopy(model.encoder)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.actor_obs_dim = model.obs_dim
    self.history_length = model.history_length
    self.history_frame_dim = model.history_frame_dim
    self.input_size = model.obs_dim + model.history_length * model.history_frame_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    current = self.obs_normalizer(x[:, : self.actor_obs_dim])
    frames = x[:, self.actor_obs_dim :].reshape(
      -1, self.history_length, self.history_frame_dim
    )
    embedding = self.encoder(self.history_normalizer(frames).flatten(start_dim=1))
    out = self.mlp(torch.cat((current, embedding), dim=-1))
    return self.deterministic_output(out)

  @torch.jit.export
  def reset(self) -> None:
    pass

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
