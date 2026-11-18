"""Runner for the decoupled (dual-agent) velocity task.

Overrides ``save`` to skip per-checkpoint ONNX export: the dual-agent policy has
two actors/critics and the standard exporter assumes a single ``joint_pos`` action
term. ONNX export for the decoupled policy can be added later once the dual-agent
inference interface is finalized.
"""

from mjlab.rl import MjlabOnPolicyRunner


class VelocityDecoupledOnPolicyRunner(MjlabOnPolicyRunner):
  def save(self, path: str, infos=None):
    super().save(path, infos)
    # Skip ONNX export + metadata for the dual-agent policy.
