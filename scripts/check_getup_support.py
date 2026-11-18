"""Small GPU integration checks for the continuous supported get-up task."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
    cfg = load_env_cfg("Wbot-20DoF-Getup-Tracking-Support-V4")
    cfg.scene.num_envs = 4
    cfg.commands["motion"].sampling_mode = "start"
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset()
        cmd = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("joint_pos")
        zeros = torch.zeros((4, 20), device=env.device)
        action.process_actions(zeros)
        assert torch.allclose(action._processed_actions, cmd.joint_pos, atol=1e-6)
        assert torch.all(cmd.time_steps == 0)
        for _ in range(30):
            obs, reward, _, _, _ = env.step(zeros)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(x).all() for x in obs.values())
        ids = torch.arange(4, device=env.device)
        for _ in range(25):
            cmd._adaptive_sampling(ids)
            valid = torch.isin(cmd.time_steps, cmd.valid_reset_frames) | (cmd.time_steps == 0)
            assert valid.all(), "Random initialization selected a force-infeasible frame"
        # Updating an exhausted reference must never write physical state.
        cmd.time_steps[:] = cmd.motion.time_step_total - 1
        cmd._fresh_reset[:] = False
        before = cmd.robot.data.root_link_pos_w.clone()
        cmd._update_command()
        assert torch.all(cmd.time_steps == cmd.motion.time_step_total - 1)
        assert torch.equal(cmd.robot.data.root_link_pos_w, before)
        # Position corrections are clipped to actual joint limits.
        action.process_actions(torch.full_like(zeros, 1e6))
        limits = cmd.robot.data.soft_joint_pos_limits
        assert (action._processed_actions <= limits[..., 1]).all()
        assert (action._processed_actions >= limits[..., 0]).all()
        # A physics failure cannot count as a successful completed get-up.
        cmd.started_at_zero[:] = True
        cmd.stand_steps[:] = 100
        env.episode_length_buf[:] = 200
        env.termination_manager._terminated_buf[:] = True
        previous = cmd.successful_start_episodes
        cmd.reset(ids)
        assert cmd.successful_start_episodes == previous
        print("SUPPORT_V4_CHECKS_PASSED " + json.dumps({
            "first_frame_exact": True, "zero_action_is_reference": True,
            "finite_steps": 30, "endpoint_no_teleport": True,
            "hard_joint_limits": True, "failure_not_success": True,
            "force_feasible_random_resets": True,
        }))
    finally:
        env.close()


if __name__ == "__main__":
    main()
