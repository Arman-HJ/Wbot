"""PPO + privileged-reconstruction loss for :class:`HistoryEncoderMLPModel` actors.

``loss = surrogate + value_coef * value - entropy_coef * entropy
        + reconstruction_coef * MSE(decoder(embedding), obs[privileged])``
(same as ``football/utils/runner.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms import PPO

from .history_actor import HistoryEncoderMLPModel


class PPOWithReconstruction(PPO):
  actor: HistoryEncoderMLPModel

  def __init__(self, *args, reconstruction_coef: float = 1.0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if not isinstance(self._raw_actor, HistoryEncoderMLPModel):
      raise TypeError(
        "PPOWithReconstruction requires a HistoryEncoderMLPModel actor, got "
        f"{type(self._raw_actor).__name__}."
      )
    self.reconstruction_coef = reconstruction_coef

  def update(self) -> dict[str, float]:
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_reconstruction_loss = 0
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None
    privileged_group = self._raw_actor.privileged_obs_group

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (  # type: ignore
            batch.advantages.std() + 1e-8  # type: ignore
          )

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      privileged_estimate = self._raw_actor.privileged_estimate
      assert privileged_estimate is not None
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
      values = self.critic(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params,  # type: ignore
            distribution_params,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          self.set_learning_rate(self.learning_rate)

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
      surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(  # type: ignore
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()  # type: ignore

      reconstruction_loss = F.mse_loss(
        privileged_estimate[:original_batch_size],
        batch.observations[privileged_group][:original_batch_size],  # type: ignore
      )

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
        + self.reconstruction_coef * reconstruction_loss
      )

      rnd_loss = (
        self.rnd.compute_loss(batch.observations[:original_batch_size])  # type: ignore
        if self.rnd
        else None
      )

      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      self.optimizer.zero_grad()
      if self.muon_optimizer is not None:
        self.muon_optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()  # type: ignore

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.muon_optimizer is not None:
        self.muon_optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_reconstruction_loss += reconstruction_loss.item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()  # type: ignore
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()  # type: ignore

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "reconstruction": mean_reconstruction_loss / num_updates,
    }
    if mean_rnd_loss is not None:
      loss_dict["rnd"] = mean_rnd_loss / num_updates
    if mean_symmetry_loss is not None:
      loss_dict["symmetry"] = mean_symmetry_loss / num_updates

    self.storage.clear()
    return loss_dict
