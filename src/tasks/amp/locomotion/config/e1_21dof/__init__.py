"""E1 AMP configs kept for AmpPpoAlgorithmCfg / runner base classes.

E1 assets are not shipped in this repo, so the E1 task is not registered.
"""

from .rl_cfg import (
  AmpPpoAlgorithmCfg as AmpPpoAlgorithmCfg,
  E121DofWalkAmpRunnerCfg as E121DofWalkAmpRunnerCfg,
  e1_21dof_walk_amp_runner_cfg as e1_21dof_walk_amp_runner_cfg,
)
