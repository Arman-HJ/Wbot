"""HIL runner config (re-exports AMP v2 ``rsl_rl`` train builder)."""

from src.tasks.amp_v2.rl.rl_cfg import AmpV2RunnerCfg, build_rsl_rl_amp_train_cfg

__all__ = ["AmpV2RunnerCfg", "build_rsl_rl_amp_train_cfg"]
