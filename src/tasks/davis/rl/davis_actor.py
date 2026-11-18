"""DAVIS actor: depth + proprio history + gated aux geometry → joint residuals."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState

from src.tasks.davis.rl.light_depth_encoder import LightDepthEncoder


class HIMEstimator(nn.Module):
  """History-informed latent estimator → 3-D (paper Table IX).

  论文引用 HIM [45]，未给出本任务中的层宽；暂用 2 层 MLP。
  """

  def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ELU(inplace=True),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ELU(inplace=True),
      nn.Linear(hidden_dim, output_dim),
    )

  def forward(self, proprio_hist: torch.Tensor) -> torch.Tensor:
    return self.net(proprio_hist)


class AuxGeometryHead(nn.Module):
  """Per-object geometry (3) + visibility logit (1) from depth latent."""

  def __init__(self, latent_dim: int = 32, hidden_dim: int = 64) -> None:
    # 论文未给出 aux 头隐层；暂用单隐层 MLP。
    super().__init__()
    self.geo = nn.Sequential(
      nn.Linear(latent_dim, hidden_dim),
      nn.ELU(inplace=True),
      nn.Linear(hidden_dim, 3),
    )
    self.vis = nn.Sequential(
      nn.Linear(latent_dim, hidden_dim),
      nn.ELU(inplace=True),
      nn.Linear(hidden_dim, 1),
    )

  def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return self.geo(z), self.vis(z).squeeze(-1)


class GTAnnealer:
  """GT→prediction √ ramp (paper App. D-B Eq. 11), ratcheted & EMA-gated."""

  def __init__(
    self,
    k0: int = 1000,
    k1: int = 8000,
    ema_threshold: float = 0.5,
    grace_update: int = 500,
  ) -> None:
    # k0/k1/ema_threshold/grace：论文未给出数值 → 可配置合理默认。
    self.k0 = k0
    self.k1 = k1
    self.ema_threshold = ema_threshold
    self.grace_update = grace_update
    self.beta = 0.0
    self.update_idx = 0
    self.aux_ema: float | None = None
    self._frozen_until = 0

  def step(self, aux_loss_value: float | None = None) -> float:
    self.update_idx += 1
    if aux_loss_value is not None:
      if self.aux_ema is None:
        self.aux_ema = float(aux_loss_value)
      else:
        self.aux_ema = 0.95 * self.aux_ema + 0.05 * float(aux_loss_value)
      if self.aux_ema > self.ema_threshold and self.update_idx > self.grace_update:
        self._frozen_until = max(self._frozen_until, self.update_idx + 1)
    if self.update_idx <= self._frozen_until:
      return self.beta
    if self.update_idx <= self.k0:
      target = 0.0
    elif self.update_idx >= self.k1:
      target = 1.0
    else:
      t = (self.update_idx - self.k0) / max(self.k1 - self.k0, 1)
      target = math.sqrt(max(0.0, min(1.0, t)))
    self.beta = max(self.beta, target)
    return self.beta


class DavisActorModel(MLPModel):
  """Deployable actor path with train-time GT annealing.

  Observation groups expected under obs_set ``actor``:
    - ``actor_proprio``: (B, proprio_dim) with history already flattened by env
    - ``actor_depth``: (B, 1, 80, 168)
    - ``aux_gt``: (B, 8) = ball_xyz(3)+goal_xyz(3)+ball_vis(1)+goal_vis(1)  [train only]
  ``aux_gt`` is NEVER concatenated raw into the MLP; only used for annealing mixture.
  """

  is_recurrent = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    depth_group: str = "actor_depth",
    proprio_group: str = "actor_proprio",
    aux_gt_group: str = "aux_gt",
    depth_latent_dim: int = 32,
    him_dim: int = 3,
    anneal_k0: int = 1000,
    anneal_k1: int = 8000,
    anneal_ema_threshold: float = 0.5,
    aux_geo_weight_ball: float = 1.0,
    aux_geo_weight_goal: float = 1.0,
    aux_vis_weight: float = 0.2,
    deployment_beta: float | None = None,
    **kwargs: Any,
  ) -> None:
    del kwargs
    # Resolve dims from sample obs before Module.__init__.
    proprio = obs[proprio_group]
    assert proprio.ndim == 2
    proprio_dim = int(proprio.shape[-1])
    depth = obs[depth_group]
    assert depth.ndim == 4 and depth.shape[1] == 1
    depth_hw = (int(depth.shape[2]), int(depth.shape[3]))
    aux_feat_dim = 8  # 2 objects × (3 + 1)
    latent_dim = depth_latent_dim + aux_feat_dim + proprio_dim + him_dim

    # Parent MLP uses proprio-only for normalizer; latent dim overridden below.
    object.__setattr__(self, "_latent_dim", latent_dim)
    object.__setattr__(self, "proprio_dim", proprio_dim)
    object.__setattr__(self, "depth_hw", depth_hw)
    object.__setattr__(self, "depth_group", depth_group)
    object.__setattr__(self, "proprio_group", proprio_group)
    object.__setattr__(self, "aux_gt_group", aux_gt_group)
    object.__setattr__(self, "depth_latent_dim", depth_latent_dim)
    object.__setattr__(self, "him_dim", him_dim)
    object.__setattr__(self, "aux_geo_weight_ball", aux_geo_weight_ball)
    object.__setattr__(self, "aux_geo_weight_goal", aux_geo_weight_goal)
    object.__setattr__(self, "aux_vis_weight", aux_vis_weight)
    object.__setattr__(self, "deployment_beta", deployment_beta)
    object.__setattr__(self, "_aux_feat_dim", aux_feat_dim)

    super().__init__(
      obs,
      {obs_set: [proprio_group]},
      obs_set,
      output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    self.depth_encoder = LightDepthEncoder(latent_dim=depth_latent_dim)
    self.ball_head = AuxGeometryHead(depth_latent_dim)
    self.goal_head = AuxGeometryHead(depth_latent_dim)
    self.him = HIMEstimator(self.proprio_dim, output_dim=him_dim)
    self.annealer = GTAnnealer(k0=anneal_k0, k1=anneal_k1, ema_threshold=anneal_ema_threshold)
    self.obs_groups_all = list(obs_groups[obs_set])
    self.last_aux_loss: torch.Tensor | None = None
    self.last_aux_logs: dict[str, float] = {}

  def _get_latent_dim(self) -> int:
    return self._latent_dim

  def get_beta(self) -> float:
    if self.deployment_beta is not None:
      return float(self.deployment_beta)
    return float(self.annealer.beta)

  def encode_aux(
    self, depth: torch.Tensor, aux_gt: torch.Tensor | None
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Returns gated_aux features, aux_loss, and debug tensors."""
    z = self.depth_encoder(depth)
    ball_y_hat, ball_q = self.ball_head(z)
    goal_y_hat, goal_q = self.goal_head(z)
    ball_p = torch.sigmoid(ball_q)
    goal_p = torch.sigmoid(goal_q)

    beta = self.get_beta()
    if aux_gt is None:
      y_ball_e, y_goal_e = ball_y_hat, goal_y_hat
      p_ball_e, p_goal_e = ball_p, goal_p
      v_ball = torch.ones_like(ball_p)
      v_goal = torch.ones_like(goal_p)
      y_ball_gt = ball_y_hat.detach()
      y_goal_gt = goal_y_hat.detach()
    else:
      y_ball_gt = aux_gt[:, 0:3]
      y_goal_gt = aux_gt[:, 3:6]
      v_ball = aux_gt[:, 6]
      v_goal = aux_gt[:, 7]
      y_ball_e = (1.0 - beta) * y_ball_gt + beta * ball_y_hat
      y_goal_e = (1.0 - beta) * y_goal_gt + beta * goal_y_hat
      p_ball_e = (1.0 - beta) * v_ball + beta * ball_p
      p_goal_e = (1.0 - beta) * v_goal + beta * goal_p

    # Detached confidence gate (paper Eq. 5 / 13).
    c_ball = torch.clamp(p_ball_e.detach(), 0.05, 1.0)
    c_goal = torch.clamp(p_goal_e.detach(), 0.05, 1.0)
    gated = torch.cat(
      [
        c_ball.unsqueeze(-1) * y_ball_e,
        c_ball.unsqueeze(-1),
        c_goal.unsqueeze(-1) * y_goal_e,
        c_goal.unsqueeze(-1),
      ],
      dim=-1,
    )

    # Aux losses (paper Eq. 10): geo only when visible; BCE on logits.
    if aux_gt is not None:
      geo_ball = (v_ball * torch.sum((ball_y_hat - y_ball_gt) ** 2, dim=-1)).mean()
      geo_goal = (v_goal * torch.sum((goal_y_hat - y_goal_gt) ** 2, dim=-1)).mean()
      bce = nn.functional.binary_cross_entropy_with_logits
      vis_ball = bce(ball_q, v_ball)
      vis_goal = bce(goal_q, v_goal)
      aux_loss = (
        self.aux_geo_weight_ball * geo_ball
        + self.aux_geo_weight_goal * geo_goal
        + self.aux_vis_weight * vis_ball
        + self.aux_vis_weight * vis_goal
      )
    else:
      aux_loss = z.new_zeros(())
      geo_ball = geo_goal = vis_ball = vis_goal = z.new_zeros(())

    extras = {
      "z_depth": z,
      "geo_ball": geo_ball.detach(),
      "geo_goal": geo_goal.detach(),
      "vis_ball": vis_ball.detach() if torch.is_tensor(vis_ball) else z.new_zeros(()),
      "vis_goal": vis_goal.detach() if torch.is_tensor(vis_goal) else z.new_zeros(()),
      "beta": z.new_tensor(beta),
    }
    return gated, aux_loss, extras

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> torch.Tensor:
    del masks, hidden_state
    proprio = obs[self.proprio_group]
    proprio_n = self.obs_normalizer(proprio) if self.obs_normalization else proprio
    depth = obs[self.depth_group]
    aux_gt = obs.get(self.aux_gt_group, None)
    gated, aux_loss, extras = self.encode_aux(depth, aux_gt)
    him = self.him(proprio_n)
    self.last_aux_loss = aux_loss
    self.last_aux_logs = {
      "aux_total": float(aux_loss.detach().item()) if aux_loss.ndim == 0 else float(aux_loss.detach().mean().item()),
      "beta": float(extras["beta"].item()),
    }
    # Actor features: depth latent + gated aux + proprio history + HIM (paper Fig. 2 / App. D-A).
    return torch.cat([extras["z_depth"], gated, proprio_n, him], dim=-1)

  def compute_aux_loss(self, obs: TensorDict) -> torch.Tensor:
    depth = obs[self.depth_group]
    aux_gt = obs[self.aux_gt_group]
    _, aux_loss, _ = self.encode_aux(depth, aux_gt)
    return aux_loss

  def as_jit(self) -> nn.Module:
    return _TorchDavisActor(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxDavisActor(self, verbose)


class _TorchDavisActor(nn.Module):
  def __init__(self, model: DavisActorModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.depth_encoder = copy.deepcopy(model.depth_encoder)
    self.ball_head = copy.deepcopy(model.ball_head)
    self.goal_head = copy.deepcopy(model.goal_head)
    self.him = copy.deepcopy(model.him)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(self, proprio: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    proprio_n = self.obs_normalizer(proprio)
    z = self.depth_encoder(depth)
    ball_y, ball_q = self.ball_head(z)
    goal_y, goal_q = self.goal_head(z)
    c_ball = torch.clamp(torch.sigmoid(ball_q).detach(), 0.05, 1.0)
    c_goal = torch.clamp(torch.sigmoid(goal_q).detach(), 0.05, 1.0)
    gated = torch.cat(
      [c_ball.unsqueeze(-1) * ball_y, c_ball.unsqueeze(-1), c_goal.unsqueeze(-1) * goal_y, c_goal.unsqueeze(-1)],
      dim=-1,
    )
    him = self.him(proprio_n)
    latent = torch.cat([z, gated, proprio_n, him], dim=-1)
    return self.deterministic_output(self.mlp(latent))


class _OnnxDavisActor(nn.Module):
  def __init__(self, model: DavisActorModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.inner = _TorchDavisActor(model)
    self.proprio_dim = model.proprio_dim
    self.depth_hw = model.depth_hw

  def forward(self, proprio: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    return self.inner(proprio, depth)

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = self.depth_hw
    return torch.zeros(1, self.proprio_dim), torch.zeros(1, 1, h, w)

  @property
  def input_names(self) -> list[str]:
    return ["proprio", "depth"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
