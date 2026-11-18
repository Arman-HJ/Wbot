"""DAVIS PPO + least-squares AMP (shooting motions; no twist command)."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.amp_constants import AMP_LABEL_DIM, AMP_LABEL_NAMES
from rsl_rl.algorithms.amp_ppo import AmpPPO
from rsl_rl.algorithms.ppo import PPO

from src.tasks.davis.rl.davis_actor import DavisActorModel


class DavisAmpPPO(AmpPPO):
  """AmpPPO + DAVIS aux / annealing; fixed soccer-style AMP labels (no twist).

  HumanoidSoccer retarget clips are all labeled ``walk`` so the discriminator
  samples a single style bucket; policy labels are the same constant one-hot.
  Head/neck DoFs are excluded from AMP via ``amp_joint_names`` (20-DoF body).
  """

  # All soccer-standard clips share this label bucket.
  FIXED_AMP_LABEL: str = "walk"

  def act(self, obs):
    if self._env is None:
      raise RuntimeError("AMP environment was not attached to the algorithm.")
    self._current_amp_state = obs["amp"].detach().clone()
    n = self._current_amp_state.shape[0]
    label_id = AMP_LABEL_NAMES.index(self.FIXED_AMP_LABEL)
    self._current_motion_label = torch.nn.functional.one_hot(
      torch.full((n,), label_id, device=self.device, dtype=torch.long),
      num_classes=AMP_LABEL_DIM,
    ).to(dtype=torch.float32)
    # Skip AmpPPO.act (twist-conditioned labels); use base PPO.act.
    return PPO.act(self, obs)

  def update(self) -> dict[str, float]:
    """DAVIS aux PPO update, then AmpPPO discriminator training."""
    # Capture before PPO path clears storage (same order as AmpPPO.update).
    amp_batch_size = (
      self.storage.num_envs
      * self.storage.num_transitions_per_env
      // self.num_mini_batches
    )
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

    # --- AmpPPO discriminator update (same as AmpPPO.update after PPO) ---
    self._enforce_minimum_std()
    if self.amp_replay.size == 0:
      raise RuntimeError("No non-terminal AMP transitions were collected.")
    amp_data = self._ensure_amp_data()

    totals = {
      "amp": 0.0,
      "amp_grad_pen": 0.0,
      "amp_policy_pred": 0.0,
      "amp_expert_pred": 0.0,
    }
    attn_totals: dict[str, float] = {}
    attn_counts: dict[str, float] = {}
    updates = self.num_learning_epochs * self.num_mini_batches
    for _ in range(updates):
      policy_state_raw, policy_next_raw, policy_motion_label = self.amp_replay.sample(
        amp_batch_size
      )
      expert_state_raw, expert_next_raw, expert_motion_label = amp_data.sample(
        policy_motion_label
      )
      policy_state = torch.cat(
        (
          self.amp_normalizer.normalize(policy_state_raw[:, :-AMP_LABEL_DIM]),
          policy_state_raw[:, -AMP_LABEL_DIM:],
        ),
        dim=1,
      )
      policy_next = torch.cat(
        (
          self.amp_normalizer.normalize(policy_next_raw[:, :-AMP_LABEL_DIM]),
          policy_next_raw[:, -AMP_LABEL_DIM:],
        ),
        dim=1,
      )
      expert_state = torch.cat(
        (self.amp_normalizer.normalize(expert_state_raw), expert_motion_label),
        dim=1,
      )
      expert_next = torch.cat(
        (self.amp_normalizer.normalize(expert_next_raw), expert_motion_label),
        dim=1,
      )

      policy_prediction = self.discriminator(policy_state, policy_next)
      expert_prediction = self.discriminator(expert_state, expert_next)
      self._accumulate_attention_logs(expert_motion_label, attn_totals, attn_counts)
      policy_loss = torch.mean(torch.square(policy_prediction + 1.0))
      expert_loss = torch.mean(torch.square(expert_prediction - 1.0))
      amp_loss = 0.5 * (policy_loss + expert_loss)
      grad_penalty = self.discriminator.gradient_penalty(
        expert_state, expert_next, self.amp_gradient_penalty
      )
      disc_loss = amp_loss + grad_penalty
      self.discriminator_optimizer.zero_grad()
      disc_loss.backward()
      if self.is_multi_gpu:
        grads = [
          parameter.grad.reshape(-1)
          for parameter in self.discriminator.parameters()
          if parameter.grad is not None
        ]
        joined = torch.cat(grads)
        torch.distributed.all_reduce(joined)
        joined /= self.gpu_world_size
        offset = 0
        for parameter in self.discriminator.parameters():
          if parameter.grad is not None:
            count = parameter.numel()
            parameter.grad.copy_(joined[offset : offset + count].view_as(parameter))
            offset += count
      torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
      self.discriminator_optimizer.step()
      self.amp_normalizer.update(policy_state_raw[:, :-AMP_LABEL_DIM])
      self.amp_normalizer.update(expert_state_raw)
      totals["amp"] += amp_loss.item()
      totals["amp_grad_pen"] += grad_penalty.item()
      totals["amp_policy_pred"] += policy_prediction.mean().item()
      totals["amp_expert_pred"] += expert_prediction.mean().item()

    loss_dict.update({name: value / updates for name, value in totals.items()})
    for label_name, fraction in zip(
      AMP_LABEL_NAMES,
      amp_data.last_label_fractions.tolist(),
      strict=True,
    ):
      loss_dict[f"amp_label_fraction/{label_name}"] = float(fraction)
    loss_dict.update(
      {
        f"amp_motion_sample/{name}": float(fraction)
        for name, fraction in zip(
          amp_data.motion_names,
          amp_data.last_motion_fractions.tolist(),
          strict=True,
        )
      }
    )
    for key, value in attn_totals.items():
      count = max(attn_counts.get(key, 0.0), 1.0)
      loss_dict[key] = value / count
    return loss_dict
