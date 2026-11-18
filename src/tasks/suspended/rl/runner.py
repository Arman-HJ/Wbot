import os

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class SuspendedOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  @staticmethod
  def _get_run_name(logger_type: str, log_dir: str | None) -> str:
    if logger_type == "wandb":
      import wandb

      return wandb.run.name if wandb.run else "local"  # type: ignore[union-attr]
    return os.path.split(log_dir)[-1] if log_dir else "local"

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name = self._get_run_name(self.logger.logger_type, self.logger.log_dir)
    onnx_path = os.path.join(policy_path, filename)
    attach_metadata_to_onnx(onnx_path, get_base_metadata(self.env.unwrapped, run_name))
    if self.logger.logger_type == "wandb":
      import wandb

      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
