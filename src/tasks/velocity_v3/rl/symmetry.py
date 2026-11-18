"""Left-right symmetry augmentation for velocity-v3 actor obs/actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict


@dataclass
class _MirrorCache:
  n_action: int
  has_phase: bool
  history_len: int
  source_idx: torch.Tensor
  sign: torch.Tensor
  ang_vel_sign: torch.Tensor
  gravity_sign: torch.Tensor
  command_sign: torch.Tensor
  ang_vel_sign_hist: torch.Tensor
  gravity_sign_hist: torch.Tensor
  command_sign_hist: torch.Tensor
  joint_source_idx_hist: torch.Tensor
  joint_sign_hist: torch.Tensor


def _get_action_joint_names(env) -> list[str]:
  """Get action-space joint names in policy action order."""
  term = env.unwrapped.action_manager.get_term("joint_pos")
  if hasattr(term, "target_names"):
    return list(term.target_names)
  robot = env.unwrapped.scene["robot"]
  return list(robot.joint_names)


def _find_lr_peer(name: str, name_to_idx: dict[str, int]) -> str | None:
  """Return the opposite-side joint name, if present in ``name_to_idx``."""
  if name.startswith("left_"):
    peer = "right_" + name[len("left_") :]
  elif name.startswith("right_"):
    peer = "left_" + name[len("right_") :]
  elif name.endswith("_Left"):
    peer = name[: -len("_Left")] + "_Right"
  elif name.endswith("_Right"):
    peer = name[: -len("_Right")] + "_Left"
  else:
    return None
  return peer if peer in name_to_idx else None


def _is_yaw_or_roll_joint_name(name: str) -> bool:
  """True for roll/yaw DoFs across Isaac (left_hip_roll) and PND (hipRoll_Left)."""
  if ("_roll_" in name) or ("_yaw_" in name):
    return True
  if name.endswith("_roll_joint") or name.endswith("_yaw_joint"):
    return True
  # PND / camelCase: hipRoll_Left, shoulderYaw_Right, ankleRoll_Left, ...
  stem, _, side = name.rpartition("_")
  if side not in ("Left", "Right"):
    stem = name
  return ("Roll" in stem) or ("Yaw" in stem)


def _build_joint_mirror_cache(env, n_action: int, has_phase: bool, history_len: int) -> _MirrorCache:
  names = _get_action_joint_names(env)
  if len(names) != n_action:
    raise RuntimeError(
      f"Symmetry augmentation expects {n_action} controlled joints, got {len(names)} action joints."
    )

  name_to_idx = {n: i for i, n in enumerate(names)}
  source_idx = list(range(n_action))
  sign = [1.0] * n_action

  visited: set[str] = set()
  for name in names:
    if name in visited:
      continue
    peer = _find_lr_peer(name, name_to_idx)
    if peer is not None:
      li = name_to_idx[name]
      ri = name_to_idx[peer]
      source_idx[li] = ri
      source_idx[ri] = li
      visited.add(name)
      visited.add(peer)

  for i, name in enumerate(names):
    # Reflection keeps pitch-like DOFs, flips yaw/roll-like DOFs.
    # This also correctly handles centerline joints (e.g. waist_yaw/waist_roll).
    if _is_yaw_or_roll_joint_name(name):
      sign[i] = -1.0

  device = env.unwrapped.device
  source_idx_t = torch.tensor(source_idx, device=device, dtype=torch.long)
  sign_t = torch.tensor(sign, device=device, dtype=torch.float32)
  ang_vel_sign = torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=torch.float32)
  gravity_sign = torch.tensor([1.0, -1.0, 1.0], device=device, dtype=torch.float32)
  command_sign = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=torch.float32)
  n_hist_joint = 3 * history_len
  joint_source_idx_hist = (
    torch.arange(n_hist_joint, device=device, dtype=torch.long).unsqueeze(1) * n_action
    + source_idx_t.unsqueeze(0)
  ).reshape(-1)

  return _MirrorCache(
    n_action=n_action,
    has_phase=has_phase,
    history_len=history_len,
    source_idx=source_idx_t,
    sign=sign_t,
    ang_vel_sign=ang_vel_sign,
    gravity_sign=gravity_sign,
    command_sign=command_sign,
    ang_vel_sign_hist=ang_vel_sign.repeat(history_len),
    gravity_sign_hist=gravity_sign.repeat(history_len),
    command_sign_hist=command_sign.repeat(history_len),
    joint_source_idx_hist=joint_source_idx_hist,
    joint_sign_hist=sign_t.repeat(n_hist_joint),
  )


def _infer_obs_layout(obs_dim: int, n_action: int) -> tuple[bool, int]:
  base_with_phase = 11 + 3 * n_action
  if obs_dim % base_with_phase == 0:
    return True, obs_dim // base_with_phase

  base_without_phase = 9 + 3 * n_action
  if obs_dim % base_without_phase == 0:
    return False, obs_dim // base_without_phase

  raise RuntimeError(
    f"Cannot infer actor obs layout for dim={obs_dim}, n_action={n_action}. "
    "Expected (11+3N)*H (with phase) or (9+3N)*H (without phase)."
  )


def _switch_joint_tensor(x: torch.Tensor, cache: _MirrorCache) -> torch.Tensor:
  source_idx = cache.source_idx if cache.source_idx.device == x.device else cache.source_idx.to(x.device)
  sign = cache.sign
  if sign.device != x.device or sign.dtype != x.dtype:
    sign = sign.to(device=x.device, dtype=x.dtype)
  y = x.index_select(dim=-1, index=source_idx)
  return y * sign


def _get_actor_obs_normalizer(env: Any) -> Any | None:
  """Actor EmpiricalNormalization injected by VelocityOnPolicyRunner during learn()."""
  env_u = getattr(env, "unwrapped", env)
  normalizer = getattr(env_u, "_symmetry_actor_normalizer", None)
  if normalizer is None or isinstance(normalizer, torch.nn.Identity):
    return None
  return normalizer


def _apply_mirror_obs_tensor(obs_sym: torch.Tensor, cache: _MirrorCache) -> None:
  """Apply left-right symmetry to actor obs rows (in-place).

  Works in raw or normalized space; caller must ensure the space matches how the
  actor will consume the augmented batch.
  """
  idx = 0
  dev = obs_sym.device
  dtype = obs_sym.dtype
  h = cache.history_len
  n = cache.n_action

  if __debug__:
    prefix_dim = (9 + (2 if cache.has_phase else 0)) * h
    joint_dim = 3 * h * n
    expected_dim = prefix_dim + joint_dim
    if expected_dim != obs_sym.shape[-1]:
      raise RuntimeError(
        f"Symmetry layout mismatch: expected {expected_dim} from cache, got {obs_sym.shape[-1]}."
      )

  ang_vel_sign = cache.ang_vel_sign_hist
  gravity_sign = cache.gravity_sign_hist
  command_sign = cache.command_sign_hist
  joint_source_idx = cache.joint_source_idx_hist
  joint_sign = cache.joint_sign_hist
  if ang_vel_sign.device != dev or ang_vel_sign.dtype != dtype:
    ang_vel_sign = ang_vel_sign.to(device=dev, dtype=dtype)
    gravity_sign = gravity_sign.to(device=dev, dtype=dtype)
    command_sign = command_sign.to(device=dev, dtype=dtype)
    joint_sign = joint_sign.to(device=dev, dtype=dtype)
  if joint_source_idx.device != dev:
    joint_source_idx = joint_source_idx.to(device=dev)

  # torso_ang_vel (axial vector): [-x, y, -z]
  torso_end = idx + 3 * h
  obs_sym[:, idx:torso_end].mul_(ang_vel_sign.view(1, -1))
  idx = torso_end

  # projected_gravity (polar vector): [x, -y, z]
  grav_end = idx + 3 * h
  obs_sym[:, idx:grav_end].mul_(gravity_sign.view(1, -1))
  idx = grav_end

  # command [vx, vy, wz] -> [vx, -vy, -wz]
  cmd_end = idx + 3 * h
  obs_sym[:, idx:cmd_end].mul_(command_sign.view(1, -1))
  idx = cmd_end

  # phase [sin, cos] mirrors as phi -> phi + 0.5 cycle:
  # [sin(2pi*phi), cos(2pi*phi)] -> [-sin(2pi*phi), -cos(2pi*phi)].
  # This keeps left/right gait semantics consistent after leg swap.
  if cache.has_phase:
    phase_end = idx + 2 * h
    obs_sym[:, idx:phase_end].mul_(-1.0)
    idx = phase_end

  # joint_pos / joint_vel / actions: mirror once on the joint dimension.
  joint_total = 3 * h * n
  joint_end = idx + joint_total
  joint_block = obs_sym[:, idx:joint_end]
  joint_sym = joint_block.index_select(dim=1, index=joint_source_idx)
  joint_sym.mul_(joint_sign.view(1, joint_total))
  joint_block.copy_(joint_sym)
  idx = joint_end

  if idx != obs_sym.shape[-1]:
    raise RuntimeError(f"Symmetry obs parse mismatch: consumed {idx}, total {obs_sym.shape[-1]}")


def _mirror_actor_obs_batch(
  actor_obs: torch.Tensor,
  cache: _MirrorCache,
  normalizer: Any | None,
) -> torch.Tensor:
  """Mirror actor obs so that actor(obs_normalizer(x)) matches T(norm(x)).

  RSL-RL augments raw rollout obs, then the actor applies per-dimension
  EmpiricalNormalization. Left/right joint dims have different running stats on
  PND, so mirroring raw values and then normalizing is NOT equivariant and
  pushes the policy toward symmetric hopping. Fix: mirror in normalized space,
  then inverse-normalize back to raw for storage.
  """
  if normalizer is None:
    obs_sym = actor_obs.clone()
    _apply_mirror_obs_tensor(obs_sym, cache)
    return obs_sym

  eps = float(getattr(normalizer, "eps", 1e-2))
  mean = normalizer._mean
  std = normalizer._std + eps
  if mean.device != actor_obs.device:
    mean = mean.to(actor_obs.device)
    std = std.to(actor_obs.device)
  if mean.dtype != actor_obs.dtype:
    mean = mean.to(dtype=actor_obs.dtype)
    std = std.to(dtype=actor_obs.dtype)

  obs_norm = (actor_obs - mean) / std
  obs_sym_norm = obs_norm.clone()
  _apply_mirror_obs_tensor(obs_sym_norm, cache)
  return obs_sym_norm * std + mean


def _get_actor_obs_tensor(obs: TensorDict | torch.Tensor) -> tuple[str | None, torch.Tensor]:
  """Extract actor/policy observation tensor from TensorDict or Tensor input."""
  if isinstance(obs, TensorDict):
    keys = list(obs.keys())
    if "actor" in keys:
      return "actor", obs["actor"]
    if "policy" in keys:
      return "policy", obs["policy"]
    if len(keys) == 1:
      k = keys[0]
      return k, obs[k]
    raise RuntimeError(f"Cannot infer actor observation key from TensorDict keys={keys}")
  return None, obs


@torch.no_grad()
def compute_symmetric_states(
  env,
  obs: TensorDict | torch.Tensor | None = None,
  actions: torch.Tensor | None = None,
):
  """Return concatenated [original; mirrored] obs/actions for PPO symmetry extension."""
  if obs is None and actions is None:
    return None, None

  if actions is not None:
    n_action = actions.shape[-1]
  elif hasattr(env, "num_actions"):
    n_action = int(env.num_actions)
  else:
    n_action = len(_get_action_joint_names(env))

  has_phase = True
  history_len = 1
  if obs is not None:
    _, actor_obs = _get_actor_obs_tensor(obs)
    if actor_obs.ndim < 2:
      raise RuntimeError(
        f"Expected actor observation tensor with at least 2 dims, got shape={tuple(actor_obs.shape)}."
      )
    has_phase, history_len = _infer_obs_layout(int(actor_obs.shape[-1]), n_action)

  cache_key = "_x2_v3_mirror_cache"
  cache: _MirrorCache | None = getattr(env, cache_key, None)
  if (
    cache is None
    or cache.n_action != n_action
    or cache.has_phase != has_phase
    or cache.history_len != history_len
  ):
    cache = _build_joint_mirror_cache(env, n_action, has_phase, history_len)
    setattr(env, cache_key, cache)

  obs_aug = None
  if obs is not None:
    actor_key, actor_obs = _get_actor_obs_tensor(obs)
    batch_size = actor_obs.shape[0]
    normalizer = _get_actor_obs_normalizer(env)
    obs_sym = _mirror_actor_obs_batch(actor_obs, cache, normalizer)

    if isinstance(obs, TensorDict):
      obs_aug = obs.repeat(2)
      assert actor_key is not None
      obs_aug[actor_key][batch_size : 2 * batch_size] = obs_sym
    else:
      obs_aug = torch.cat([obs, obs_sym], dim=0)

  actions_aug = None
  if actions is not None:
    actions_sym = _switch_joint_tensor(actions, cache)
    actions_aug = torch.cat([actions, actions_sym], dim=0)

  return obs_aug, actions_aug


if __name__ == "__main__":
  class _DummyTerm:
    def __init__(self, target_names: list[str]) -> None:
      self.target_names = target_names

  class _DummyActionManager:
    def __init__(self, target_names: list[str]) -> None:
      self._term = _DummyTerm(target_names)

    def get_term(self, _name: str):
      return self._term

  class _DummyEnv:
    def __init__(self, target_names: list[str], device: str = "cpu") -> None:
      self.device = device
      self.unwrapped = self
      self.action_manager = _DummyActionManager(target_names)
      self.scene = {"robot": object()}
      self.num_actions = len(target_names)

  def _run_case(joint_names: list[str], src_idx: list[int], sign: list[float], label: str) -> None:
    env = _DummyEnv(joint_names)
    batch_size = 2
    n_action = len(joint_names)
    obs_dim = 11 + 3 * n_action
    actor_obs = torch.arange(batch_size * obs_dim, dtype=torch.float32).reshape(batch_size, obs_dim)
    actions = torch.arange(batch_size * n_action, dtype=torch.float32).reshape(batch_size, n_action)
    obs_td = TensorDict({"actor": actor_obs.clone()}, batch_size=[batch_size])

    src_idx_t = torch.tensor(src_idx, dtype=torch.long)
    sign_t = torch.tensor(sign, dtype=torch.float32)

    obs_aug, actions_aug = compute_symmetric_states(env=env, obs=obs_td, actions=actions)
    assert obs_aug is not None and actions_aug is not None
    expected_actions_sym = actions.index_select(dim=1, index=src_idx_t) * sign_t.view(1, -1)
    assert torch.allclose(actions_aug[batch_size:], expected_actions_sym), f"{label}: actions mirror mismatch"

    expected_obs_sym = actor_obs.clone()
    expected_obs_sym[:, 0:3] *= torch.tensor([-1.0, 1.0, -1.0])
    expected_obs_sym[:, 3:6] *= torch.tensor([1.0, -1.0, 1.0])
    expected_obs_sym[:, 6:9] *= torch.tensor([1.0, -1.0, -1.0])
    expected_obs_sym[:, 9:11] *= -1.0
    joint_block = expected_obs_sym[:, 11:].view(batch_size, 3, n_action)
    joint_block = joint_block.index_select(dim=2, index=src_idx_t) * sign_t.view(1, 1, -1)
    expected_obs_sym[:, 11:] = joint_block.reshape(batch_size, -1)
    assert torch.allclose(obs_aug["actor"][batch_size:], expected_obs_sym), f"{label}: obs mirror mismatch"
    print(f"{label}: passed")

  x2_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
  ]
  _run_case(
    x2_joint_names,
    src_idx=[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13],
    sign=[1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1],
    label="x2 prefix naming",
  )

  pnd_joint_names = [
    "hipPitch_Left",
    "hipRoll_Left",
    "hipYaw_Left",
    "kneePitch_Left",
    "anklePitch_Left",
    "ankleRoll_Left",
    "hipPitch_Right",
    "hipRoll_Right",
    "hipYaw_Right",
    "kneePitch_Right",
    "anklePitch_Right",
    "ankleRoll_Right",
  ]
  _run_case(
    pnd_joint_names,
    src_idx=[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5],
    sign=[1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1],
    label="pnd suffix naming",
  )

  # Normalization-aware mirror: actor(norm(x)) must see T(norm(x)), not norm(T_raw(x)).
  class _DummyNorm:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-2) -> None:
      self._mean = mean
      self._std = std
      self.eps = eps

  env = _DummyEnv(pnd_joint_names)
  n_action = len(pnd_joint_names)
  obs_dim = 11 + 3 * n_action
  mean = torch.zeros(1, obs_dim)
  std = torch.ones(1, obs_dim)
  # Asymmetric running stats on left vs right hip pitch (indices 11 and 17).
  mean[0, 11] = 0.2
  mean[0, 17] = -0.1
  std[0, 11] = 1.5
  std[0, 17] = 0.8
  env._symmetry_actor_normalizer = _DummyNorm(mean, std)

  actor_obs = torch.randn(2, obs_dim)
  obs_td = TensorDict({"actor": actor_obs.clone()}, batch_size=[2])
  obs_aug, _ = compute_symmetric_states(env=env, obs=obs_td, actions=None)
  assert obs_aug is not None

  eps = env._symmetry_actor_normalizer.eps
  obs_norm = (actor_obs - mean) / (std + eps)
  expected_sym_norm = obs_norm.clone()
  cache = getattr(env, "_x2_v3_mirror_cache")
  _apply_mirror_obs_tensor(expected_sym_norm, cache)
  expected_sym_raw = expected_sym_norm * (std + eps) + mean
  assert torch.allclose(obs_aug["actor"][2:], expected_sym_raw, atol=1e-5), "norm-aware obs mirror mismatch"

  # Raw mirror without normalizer must differ when L/R stats differ.
  env_no_norm = _DummyEnv(pnd_joint_names)
  obs_td2 = TensorDict({"actor": actor_obs.clone()}, batch_size=[2])
  obs_aug2, _ = compute_symmetric_states(env=env_no_norm, obs=obs_td2, actions=None)
  assert obs_aug2 is not None
  assert not torch.allclose(obs_aug2["actor"][2:], expected_sym_raw, atol=1e-5)
  print("normalization-aware mirror: passed")

  print("symmetry self-test passed.")


__all__ = ["compute_symmetric_states"]
