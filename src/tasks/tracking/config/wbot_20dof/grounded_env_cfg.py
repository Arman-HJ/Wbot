"""Prone tracking with an already-supported first frame and mesh-floor contacts."""
from copy import deepcopy
from pathlib import Path

import mujoco

from src.assets.robots.wbot.wbot_20dof_constants import get_spec
from .env_cfgs import wbot_20dof_flat_tracking_env_cfg


MOTION = (Path(__file__).resolve().parents[4] / "assets/motions/wbot_20dof"
          / "prone_grounded_start/prone_1_grounded_start_tracking.npz")


def grounded_robot_spec():
    """Preserve source XML; use exact visual hull support and primitive self contacts."""
    spec = get_spec()
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.name = "floor_mesh_" + geom.meshname
            geom.contype, geom.conaffinity = 2, 1
            geom.condim = 3
            geom.priority = 1
            geom.friction[:] = [.6,.005,.0001]
        elif geom.contype or geom.conaffinity:
            geom.contype = geom.conaffinity = 4
            geom.condim = 1
    return spec


def grounded_start_tracking_env_cfg(play=False):
    cfg = wbot_20dof_flat_tracking_env_cfg(play=play)
    robot = deepcopy(cfg.scene.entities["robot"])
    robot.spec_fn = grounded_robot_spec
    robot.collisions = ()
    robot.articulation.soft_joint_pos_limit_factor = 1.0
    cfg.scene.entities["robot"] = robot
    cfg.episode_length_s = 9.06 if not play else 1e9
    cfg.sim.mujoco.timestep = .002
    cfg.decimation = 10
    cfg.sim.mujoco.iterations = 20
    cfg.sim.nconmax = 160
    cfg.sim.njmax = 750
    motion = cfg.commands["motion"]
    motion.motion_file = str(MOTION)
    # Avoid lifting/rotating the contact pose during reset. Random reference-state
    # sampling remains available to learn the full motion; play starts at frame 0.
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.,0.)
    # First learn the nominal contact sequence, without random pushes or changed COM.
    for key in ("push_robot","base_com","encoder_bias","foot_friction"):
        cfg.events.pop(key,None)
    cfg.viewer.elevation = -20.
    return cfg
