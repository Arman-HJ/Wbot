"""Runner cfg pieces for the history-encoder actor + reconstruction PPO."""

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from src.tasks.tracking.history_obs import HISTORY_LENGTH


@dataclass
class HistoryActorCfg(RslRlModelCfg):
  """Actor = MLP(cat(actor obs, encoder(actor_history))); decoder → privileged."""

  class_name: str = "src.tasks.tracking.rl.history_actor:HistoryEncoderMLPModel"
  history_obs_group: str = "actor_history"
  privileged_obs_group: str = "privileged"
  history_length: int = HISTORY_LENGTH
  encoder_hidden_dims: tuple[int, ...] = (1024, 128)
  embedding_dim: int = 64
  decoder_hidden_dims: tuple[int, ...] = (128, 128)


@dataclass
class ReconPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "src.tasks.tracking.rl.ppo_recon:PPOWithReconstruction"
  reconstruction_coef: float = 1.0
