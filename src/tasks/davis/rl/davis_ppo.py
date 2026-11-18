"""PPO + visibility-aware auxiliary geometry loss + GT annealing schedule."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO

from src.tasks.davis.rl.davis_actor import DavisActorModel


class DavisPPO(PPO):
  """Asymmetric actor-critic PPO with DAVIS aux losses (no AMP).

  Production shooting train uses ``DavisAmpPPO`` (AMP + aux). This class remains
  for ablation / debugging without a discriminator.
  """

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_aux_loss = 0.0
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None

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
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
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

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

      # DAVIS auxiliary geometry / visibility (paper Eq. 3 / App. D-B).
      aux_loss = batch.observations.new_zeros(())
      if isinstance(self._raw_actor, DavisActorModel):
        aux_loss = self._raw_actor.compute_aux_loss(batch.observations[:original_batch_size])
        loss = loss + aux_loss

      rnd_loss = None
      if self.rnd:
        rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size])

      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss
      else:
        symmetry_loss = None

      self.optimizer.zero_grad()
      if self.muon_optimizer is not None:
        self.muon_optimizer.zero_grad()
      loss.backward()
      if self.rnd and rnd_loss is not None:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()

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
      mean_aux_loss += float(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else 0.0
      if mean_rnd_loss is not None and rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None and symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_aux_loss /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    # Advance GT→prediction annealing once per PPO update (paper: measured in policy updates).
    if isinstance(self._raw_actor, DavisActorModel):
      self._raw_actor.annealer.step(aux_loss_value=mean_aux_loss)

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "aux": mean_aux_loss,
    }
    if isinstance(self._raw_actor, DavisActorModel):
      loss_dict["anneal_beta"] = float(self._raw_actor.annealer.beta)
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss

    self.storage.clear()
    return loss_dict
