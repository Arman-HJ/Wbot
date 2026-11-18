import os

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Expose actor obs normalizer to symmetry augmentation (norm-equivariant mirror)."""
    symmetry = getattr(self.alg, "symmetry", None)
    if symmetry is not None:
      self.env.unwrapped._symmetry_actor_normalizer = self.alg.actor.obs_normalizer
    try:
      super().learn(
        num_learning_iterations=num_learning_iterations,
        init_at_random_ep_len=init_at_random_ep_len,
      )
    finally:
      if symmetry is not None:
        self.env.unwrapped._symmetry_actor_normalizer = None

  @staticmethod
  def _get_run_name(logger_type: str, log_dir: str | None) -> str:
    if logger_type == "wandb":
      import wandb

      return wandb.run.name if wandb.run else "local"  # type: ignore[union-attr]
    if logger_type == "trackio":
      from trackio import context_vars

      run = context_vars.current_run.get()
      return run.name if run is not None else "local"
    return os.path.split(log_dir)[-1] if log_dir else "local"

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name = self._get_run_name(self.logger.logger_type, self.logger.log_dir)
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type == "wandb":
      import wandb

      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
    elif self.logger.logger_type == "trackio":
      import trackio

      trackio.save(onnx_path)
