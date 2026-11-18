"""RL configuration for decoupled (dual-agent) velocity tracking tasks.

Extends the standard ``RslRlOnPolicyRunnerCfg`` with a second actor/critic pair so
``PPODecoupled`` can train a lower-body agent (gait) and an upper-body agent
(stability / angular-momentum balance) with independent optimizers and rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


@dataclass
class RslRlDecoupledRunnerCfg(RslRlBaseRunnerCfg):
  """Dual-agent on-policy runner config (``PPODecoupled``)."""

  class_name: str = "MjlabOnPolicyRunner"
  """The runner class name. Default is MjlabOnPolicyRunner."""

  num_actions_lower: int = 12
  """Action dimension of the lower-body agent (legs)."""

  num_actions_upper: int = 17
  """Action dimension of the upper-body agent (torso + arms)."""

  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor_lower": ("actor",),
      "actor_upper": ("actor",),
      "critic_lower": ("critic",),
      "critic_upper": ("critic",),
    },
  )

  actor_lower: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    )
  )
  """Lower-body actor (legs / gait)."""

  actor_upper: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    )
  )
  """Upper-body actor (torso + arms / stability)."""

  critic_lower: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    )
  )
  """Lower-body critic."""

  critic_upper: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    )
  )
  """Upper-body critic."""

  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      class_name="PPODecoupled",
      learning_rate=1.0e-3,
      schedule="adaptive",
      desired_kl=0.01,
    )
  )
  """Shared algorithm defaults. ``class_name`` must be ``PPODecoupled``.

  Per-agent fields in ``algorithm_lower`` / ``algorithm_upper`` override these.
  """

  algorithm_lower: RslRlPpoAlgorithmCfg | None = None
  """Lower-body PPO hyperparameters (overrides ``algorithm``). ``class_name`` ignored."""

  algorithm_upper: RslRlPpoAlgorithmCfg | None = None
  """Upper-body PPO hyperparameters (overrides ``algorithm``). ``class_name`` ignored."""
