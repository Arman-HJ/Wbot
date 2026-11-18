"""Contact-aware get-up learning with continuous reference progression."""
from dataclasses import dataclass
import math
import hashlib
from pathlib import Path

import numpy as np
import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply


class ReferenceResidualAction(JointPositionAction):
    """Predict a bounded position correction to the reference, in radians."""

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        command = self._env.command_manager.get_term("motion")
        reference = command.joint_pos[:, self._target_ids]
        correction = (actions * self._scale).clamp(-0.75, 0.75)
        limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
        self._processed_actions = torch.clamp(
            reference + correction, limits[..., 0], limits[..., 1]
        )


@dataclass(kw_only=True)
class ReferenceResidualActionCfg(JointPositionActionCfg):
    def build(self, env):
        return ReferenceResidualAction(self, env)


class ContinuousGetupCommand(MotionCommand):
    """Reset only between episodes; hold the final reference instead of teleporting."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.end_hold_steps = torch.zeros_like(self.time_steps)
        self.stand_steps = torch.zeros_like(self.time_steps)
        self.started_at_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fresh_reset = torch.zeros_like(self.started_at_zero)
        self.completed_start_episodes = 0
        self.successful_start_episodes = 0
        self.contact_targets = None
        self.contact_names = []
        self.valid_reset_frames = torch.arange(self.motion.time_step_total, device=self.device)
        with np.load(cfg.motion_file) as data:
            names = [str(n) for n in data["joint_names"]]
            if names != list(self.robot.joint_names):
                raise ValueError("Reference joint order must match robot joint order")
            if not math.isclose(float(data["fps"]), 1.0 / env.step_dt):
                raise ValueError("Reference FPS must match the policy control frequency")
            if "support_contact" in data:
                self.contact_targets = torch.as_tensor(
                    data["support_contact"], dtype=torch.float32, device=self.device
                )
                self.contact_names = [str(n) for n in data["support_body_names"]]
        annotations = Path(cfg.motion_file).with_name("support_annotations.npz")
        if self.contact_targets is None and annotations.exists():
            with np.load(annotations) as data:
                if "motion_sha256" in data:
                    digest = hashlib.sha256(Path(cfg.motion_file).read_bytes()).hexdigest()
                    if str(data["motion_sha256"].item()) != digest:
                        raise ValueError("Support annotations belong to a different reference file")
                self.contact_targets = torch.as_tensor(data["contact_flags"], dtype=torch.float32, device=self.device)
                self.contact_names = [str(n) for n in data["body_names"]]
                if "force_feasible" in data:
                    valid = np.asarray(data["force_feasible"], dtype=bool)
                    if valid.shape != (self.motion.time_step_total,) or not valid.any():
                        raise ValueError("Invalid force-feasibility reset mask")
                    self.valid_reset_frames = torch.as_tensor(np.flatnonzero(valid), device=self.device)
            if len(self.contact_targets) != self.motion.time_step_total:
                raise ValueError("Support annotations do not match reference length")
        for key in ("standing_hold_s", "reference_progress", "from_start_success_rate",
                    "from_start_completed_episodes"):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    def _adaptive_sampling(self, env_ids):
        # Count true failures, then explicitly bound hard-bin sampling to 15%.
        failed = self._env.termination_manager.terminated[env_ids]
        bins = (self.time_steps[env_ids] * self.bin_count // self.motion.time_step_total).clamp(
            max=self.bin_count - 1
        )
        self._current_bin_failed += torch.bincount(bins[failed], minlength=self.bin_count)
        hard = self.bin_failed_count + 1e-3
        hard = hard / hard.sum()
        # Random reference initialization uses only force-feasible poses. Full
        # episodes still traverse the entire clip and must learn its transitions.
        valid = self.valid_reset_frames
        valid_bins = (valid * self.bin_count // self.motion.time_step_total).clamp(max=self.bin_count - 1)
        counts = torch.bincount(valid_bins, minlength=self.bin_count).clamp(min=1)
        hard_frame = hard[valid_bins] / counts[valid_bins]
        hard_frame /= hard_frame.sum()
        p_frame = .7 / len(valid) + .3 * hard_frame
        frames = valid[torch.multinomial(p_frame, len(env_ids), replacement=True)].clone()
        frames[torch.rand(len(env_ids), device=self.device) < self.cfg.start_fraction] = 0
        self.time_steps[env_ids] = frames
        p = torch.zeros_like(hard).scatter_add_(0, valid_bins, p_frame)
        overall = (1 - self.cfg.start_fraction) * p
        overall[0] += self.cfg.start_fraction
        self.metrics["sampling_entropy"][:] = (-(overall * overall.clamp(min=1e-12).log()).sum() / math.log(self.bin_count)
                                                if self.bin_count > 1 else 1.)
        prob, idx = overall.max(0)
        self.metrics["sampling_top1_prob"][:] = prob
        self.metrics["sampling_top1_bin"][:] = idx.float() / self.bin_count

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.end_hold_steps[env_ids] = 0
        self.stand_steps[env_ids] = 0
        self.started_at_zero[env_ids] = self.time_steps[env_ids] == 0
        self._fresh_reset[env_ids] = True

    def _update_command(self):
        active = ~self._fresh_reset
        at_end = self.time_steps >= self.motion.time_step_total - 1
        self.end_hold_steps += (active & at_end).long()
        stand = standing_state(self._env)
        self.stand_steps = torch.where(active & at_end & stand, self.stand_steps + 1, 0)
        self.time_steps = (self.time_steps + active.long()).clamp(max=self.motion.time_step_total - 1)
        self._fresh_reset[:] = False
        self.update_relative_body_poses()
        self.bin_failed_count.lerp_(self._current_bin_failed, self.cfg.adaptive_alpha)
        self._current_bin_failed.zero_()
        self.metrics["standing_hold_s"][:] = self.stand_steps * self._env.step_dt
        self.metrics["reference_progress"][:] = self.time_steps / max(1, self.motion.time_step_total - 1)

    def reset(self, env_ids):
        # Count only completed, non-initialization episodes that began at frame zero.
        eligible = self.started_at_zero[env_ids] & (self._env.episode_length_buf[env_ids] > 0)
        required = round(self.cfg.final_hold_s / self._env.step_dt)
        success = (self.stand_steps[env_ids] >= required) & ~self._env.termination_manager.terminated[env_ids]
        success &= standing_state(self._env)[env_ids]
        self.completed_start_episodes += int(eligible.sum().item())
        self.successful_start_episodes += int((eligible & success).sum().item())
        result = super().reset(env_ids)
        result["from_start_success_rate"] = self.successful_start_episodes / max(1, self.completed_start_episodes)
        result["from_start_completed_episodes"] = self.completed_start_episodes
        return result


@dataclass(kw_only=True)
class ContinuousGetupCommandCfg(MotionCommandCfg):
    start_fraction: float = 0.5
    final_hold_s: float = 2.0

    def build(self, env):
        return ContinuousGetupCommand(self, env)


def _foot_forces(env):
    sensor = env.scene["support_contacts"]
    names = sensor.primary_names
    indices = [next(i for i, n in enumerate(names) if n.endswith(side + "_ankle_roll_link"))
               for side in ("left", "right")]
    return sensor.data.force[:, indices].norm(dim=-1)


def standing_state(env):
    robot = env.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    up = quat_apply(robot.data.root_link_quat_w, torch.tensor(
        [0., 0., 1.], device=env.device).expand(env.num_envs, -1))[:, 2]
    speed = robot.data.root_link_lin_vel_w.norm(dim=-1)
    omega = robot.data.root_link_ang_vel_w.norm(dim=-1)
    feet = (_foot_forces(env) > 5.).all(dim=-1)
    return (z > .48) & (up > .9) & (speed < .15) & (omega < .5) & feet


def motion_complete(env, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    return cmd.end_hold_steps >= round(cmd.cfg.final_hold_s / env.step_dt)


def stable_standing_reward(env):
    cmd = env.command_manager.get_term("motion")
    late = cmd.time_steps >= int(.8 * cmd.motion.time_step_total)
    return (standing_state(env) & late).float()


def support_contact_reward(env):
    """Reward loaded requested supports; allow additional recovery contacts."""
    cmd = env.command_manager.get_term("motion")
    if cmd.contact_targets is None:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene["support_contacts"]
    indices = [next(i for i, n in enumerate(sensor.primary_names) if n.endswith(name))
               for name in cmd.contact_names]
    loaded = (sensor.data.force[:, indices].norm(dim=-1) / 10.).clamp(0, 1)
    desired = cmd.contact_targets[cmd.time_steps]
    return (loaded * desired).sum(-1) / desired.sum(-1).clamp(min=1)


def loaded_foot_slip(env):
    robot = env.scene["robot"]
    indices = robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)[0]
    speed2 = robot.data.body_link_lin_vel_w[:, indices, :2].square().sum(-1)
    return (speed2 * (_foot_forces(env) > 5.)).sum(-1)


def residual_rate_cost(env):
    # Measure changes of the residual in radians, independent of motor-specific scale.
    term = env.action_manager.get_term("joint_pos")
    diff = (env.action_manager.action - env.action_manager.prev_action) * term.scale
    return diff.square().sum(-1)
