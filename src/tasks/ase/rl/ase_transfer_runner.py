"""ASE phase-2 hierarchical transfer: high-level ω(z̄|s,g) + frozen low-level π(a|s,z)."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from tensordict import TensorDict

from rsl_rl.models import ASEDiscriminatorEncoder

from src.tasks.ase.mdp.commands import SphericalSkillCommand


def _unwrap_mjlab_env(env: VecEnv):
  unwrapped = getattr(env, "unwrapped", env)
  if hasattr(unwrapped, "env"):
    unwrapped = unwrapped.env.unwrapped
  return unwrapped


class AseHierarchicalVecEnv:
  """VecEnv adapter: HL actions are unnormalized latents; LL produces joint targets."""

  def __init__(
    self,
    env: VecEnv,
    *,
    low_level_actor: MLPModel,
    latent_dim: int,
    latent_repeat: int = 5,
    discriminator: ASEDiscriminatorEncoder | None = None,
    task_style_lerp: float = 0.5,
    device: str = "cpu",
  ) -> None:
    self.env = env
    self.low_level_actor = low_level_actor
    self.latent_dim = int(latent_dim)
    self.latent_repeat = max(1, int(latent_repeat))
    self.discriminator = discriminator
    self.task_style_lerp = float(task_style_lerp)
    self.device = torch.device(device) if not isinstance(device, torch.device) else device
    self.num_envs = env.num_envs
    self.num_actions = self.latent_dim
    self.max_episode_length = env.max_episode_length
    self.cfg = getattr(env, "cfg", None)
    self._held_latent = torch.zeros(self.num_envs, self.latent_dim, device=self.device)
    self._steps_held = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.low_level_actor.eval()
    for p in self.low_level_actor.parameters():
      p.requires_grad_(False)
    if self.discriminator is not None:
      self.discriminator.eval()
      self.discriminator.requires_grad_(False)

  @property
  def unwrapped(self):
    return getattr(self.env, "unwrapped", self.env)

  @property
  def episode_length_buf(self):
    return self.env.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value):
    self.env.episode_length_buf = value

  def _strip_skill(self, obs: TensorDict) -> TensorDict:
    out = obs.clone()
    if "skill" in out.keys():
      del out["skill"]
    return out

  def _inject_latent(
    self, z_bar: torch.Tensor, obs: TensorDict
  ) -> tuple[TensorDict, torch.Tensor]:
    z = torch.nn.functional.normalize(z_bar, dim=-1)
    mj_env = _unwrap_mjlab_env(self.env)
    term = mj_env.command_manager.get_term("skill")
    assert isinstance(term, SphericalSkillCommand)
    term.set_latents(z)
    full = obs.clone()
    full["skill"] = z
    return full, z

  def get_observations(self) -> TensorDict:
    obs = self.env.get_observations()
    return self._strip_skill(obs)

  def step(self, actions: torch.Tensor):
    """``actions`` are unnormalized latents z̄ from the high-level policy."""
    actions = actions.to(self.device)
    # Resample held latent when the repeat horizon expires.
    need_new = self._steps_held <= 0
    if torch.any(need_new):
      ids = need_new.nonzero(as_tuple=False).squeeze(-1)
      self._held_latent[ids] = actions[ids]
      self._steps_held[ids] = self.latent_repeat
    self._steps_held -= 1

    base_obs = self.env.get_observations().to(self.device)
    full_obs, _z = self._inject_latent(self._held_latent, base_obs)

    with torch.inference_mode():
      joint_actions = self.low_level_actor(full_obs, stochastic_output=False)

    obs, rewards, dones, extras = self.env.step(joint_actions.to(self.env.device))
    obs = obs.to(self.device)
    rewards = rewards.to(self.device)
    dones = dones.to(self.device)

    if self.discriminator is not None and "disc" in obs.keys():
      disc_obs = self.discriminator.get_disc_obs(obs, flatten_history_dim=False)
      style, _score = self.discriminator.predict_style_reward(
        disc_obs, dt=float(_unwrap_mjlab_env(self.env).step_dt)
      )
      rewards = (
        self.task_style_lerp * rewards
        + (1.0 - self.task_style_lerp) * style
      )

    # Reset held latents on episode ends so the next HL action is applied immediately.
    done_mask = dones.to(dtype=torch.bool)
    if torch.any(done_mask):
      self._steps_held[done_mask] = 0

    return self._strip_skill(obs), rewards, dones, extras

  def close(self) -> None:
    if hasattr(self.env, "close"):
      self.env.close()


class RslRlAseTransferOnPolicyRunner(OnPolicyRunner):
  """Train high-level skill selector on top of a frozen ASE low-level policy."""

  _is_mjlab_ase_transfer_runner = True
  _uses_rsl_rl_amp_env = True

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self._raw_env = env
    self.cfg = train_cfg
    self.device = device
    ll_ckpt = train_cfg.get("low_level_checkpoint") or ""
    if not ll_ckpt:
      raise ValueError(
        "ASE transfer requires low_level_checkpoint (phase-1 ASE model_*.pt path)."
      )
    latent_dim = int(train_cfg.get("latent_dim", 64))
    latent_repeat = int(train_cfg.get("latent_repeat", 5))
    task_style_lerp = float(
      train_cfg.get("algorithm", {}).get("task_style_lerp", 0.5)
    )

    ll_actor, disc = self._load_low_level(env, ll_ckpt, latent_dim, device)
    hier_env = AseHierarchicalVecEnv(
      env,
      low_level_actor=ll_actor,
      latent_dim=latent_dim,
      latent_repeat=latent_repeat,
      discriminator=disc if train_cfg.get("freeze_discriminator", True) else None,
      task_style_lerp=task_style_lerp,
      device=device,
    )
    # Override actor output dim to latent_dim via env.num_actions before construct.
    super().__init__(hier_env, train_cfg, log_dir, device)
    self.low_level_actor = ll_actor
    self.ase_discriminator = disc

  def _load_low_level(
    self,
    env: VecEnv,
    checkpoint: str,
    latent_dim: int,
    device: str,
  ) -> tuple[MLPModel, ASEDiscriminatorEncoder | None]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    obs = env.get_observations()
    # Build LL actor with skill in policy groups (phase-1 layout).
    ll_obs_groups = {
      "policy": ["actor", "skill"],
      "critic": ["critic", "skill"],
      "discriminator": ["disc"],
      "discriminator_demonstration": ["disc_demo"],
    }
    # Ensure skill key exists for construction.
    if "skill" not in obs.keys():
      raise RuntimeError(
        "ASE transfer env must expose a 'skill' observation group "
        "(call attach_skill_latent_command)."
      )
    actor_cfg = dict(self.cfg.get("low_level_actor") or self.cfg["actor"])
    actor_cfg.pop("class_name", None)
    # LL action dim = underlying joint actions.
    joint_dim = int(env.num_actions)
    actor = MLPModel(obs, ll_obs_groups, "actor", joint_dim, **actor_cfg).to(device)
    state = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict")
    if state is None and "actor" in ckpt:
      state = ckpt["actor"]
    # rsl_rl PPO save format
    if state is None:
      # OnPolicyRunner.save stores under alg.save() keys
      for key in ("actor_state_dict", "policy_state_dict"):
        if key in ckpt:
          state = ckpt[key]
          break
    if state is None:
      # Fallback: alg.save() nested
      raise KeyError(
        f"Could not find actor weights in {checkpoint}. Keys={list(ckpt.keys())}"
      )
    actor.load_state_dict(state, strict=False)
    actor.eval()

    disc = None
    if self.cfg.get("freeze_discriminator", True) and "amp_discriminator_state_dict" in ckpt:
      from rsl_rl.models.discriminator_model import LossType, resolve_amp_config

      # Minimal disc rebuild from checkpoint dims.
      disc_obs = obs["disc"]
      disc_obs_dim = int(disc_obs.shape[-1])
      disc_obs_steps = int(disc_obs.shape[1])
      disc = ASEDiscriminatorEncoder(
        disc_obs_dim=disc_obs_dim,
        disc_obs_steps=disc_obs_steps,
        obs_groups=ll_obs_groups,
        latent_dim=latent_dim,
        loss_type=LossType.WGAN,
        device=device,
        hidden_dims=tuple(
          self.cfg.get("algorithm", {})
          .get("amp_cfg", {})
          .get("amp_discriminator", {})
          .get("hidden_dims", (512, 256))
        ),
      ).to(device)
      disc.load_state_dict(ckpt["amp_discriminator_state_dict"], strict=False)
      if "amp_discriminator_normalizer_state_dict" in ckpt:
        disc.disc_obs_normalizer.load_state_dict(
          ckpt["amp_discriminator_normalizer_state_dict"], strict=False
        )
      disc.eval()
    return actor, disc
