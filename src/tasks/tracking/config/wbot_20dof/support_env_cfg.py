"""Supported get-up v4: residual control, bounded sampling, continuous episodes."""
from dataclasses import fields
from pathlib import Path

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg

from .grounded_env_cfg import grounded_start_tracking_env_cfg
from .support_mdp import (
    ContinuousGetupCommandCfg, ReferenceResidualActionCfg, loaded_foot_slip,
    motion_complete, residual_rate_cost, stable_standing_reward, support_contact_reward,
)

MOTION = (Path(__file__).resolve().parents[4] / "assets/motions/wbot_20dof"
          / "prone_support_v4/final_candidate/prone_1_support_v4_tracking.npz")


def support_tracking_env_cfg(play=False):
    cfg = grounded_start_tracking_env_cfg(play=play)
    previous = cfg.commands["motion"]
    kwargs = {f.name: getattr(previous, f.name) for f in fields(previous)}
    cfg.commands["motion"] = ContinuousGetupCommandCfg(**kwargs)
    command = cfg.commands["motion"]
    command.motion_file = str(MOTION)
    command.start_fraction = .5
    command.final_hold_s = 2.
    command.debug_vis = False
    # The phase-based timeout handles arbitrary clip length and sampled starts.
    cfg.episode_length_s = 60.
    cfg.terminations["motion_complete"] = TerminationTermCfg(func=motion_complete, time_out=True)
    cfg.actions["joint_pos"] = ReferenceResidualActionCfg(
        entity_name="robot", actuator_names=(".*",), use_default_offset=False,
        scale={
        ".*_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_joint": .25,
        ".*_shoulder_(pitch|roll)_joint": .2,
        ".*_(shoulder_yaw|elbow)_joint": .12,
        },
    )
    cfg.scene.sensors += (ContactSensorCfg(
        name="support_contacts",
        primary=ContactMatch(mode="body", entity="robot", pattern=(
            ".*_ankle_roll_link", ".*_elbow_link", ".*_knee_link")),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce",
    ),)
    cfg.scene.sensors[0].reduce = "maxforce"
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=residual_rate_cost, weight=-.5)
    cfg.rewards["support_contacts"] = RewardTermCfg(func=support_contact_reward, weight=.5)
    cfg.rewards["standing_hold"] = RewardTermCfg(func=stable_standing_reward, weight=2.)
    cfg.rewards["loaded_foot_slip"] = RewardTermCfg(func=loaded_foot_slip, weight=-.1)
    # Let the policy depart modestly from the reference to maintain support.
    cfg.rewards["motion_body_ori"].params["std"] = .6
    cfg.rewards["motion_ee_pos"].weight = 1.
    cfg.rewards["motion_ee_pos"].params["std"] = .25
    noise = {"motion_anchor_pos_b": .02, "motion_anchor_ori_b": .02,
             "base_lin_vel": .05, "base_ang_vel": .05, "joint_pos": .005,
             "joint_vel": .1}
    for key, magnitude in noise.items():
        cfg.observations["actor"].terms[key].noise = UniformNoiseCfg(n_min=-magnitude, n_max=magnitude)
    return cfg
