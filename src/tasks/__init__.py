from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp", "adam_pro", "amp", "tinymdm", "getup2", "amp_v2", "ase", "smp", "hil"]

# DroidUpE1-style AMP locomotion (rsl_rl AmpPPO).
import src.tasks.amp.locomotion.config.wbot_20dof  # noqa: F401
import src.tasks.amp.locomotion.config.wbot_12dof  # noqa: F401
import src.tasks.amp.locomotion.config.e1_21dof  # noqa: F401
import src.tasks.amp.getup.config.wbot_20dof  # noqa: F401
import src.tasks.suspended.config.wbot_12dof  # noqa: F401
import src.tasks.suspended.config.wbot_20dof  # noqa: F401
import src.tasks.tracking.config.wbot_12dof_kick  # noqa: F401
import src.tasks.tracking.config.wbot_20dof_kick  # noqa: F401
import src.tasks.tracking.config.wbot_20dof_roban_20260928  # noqa: F401
import src.tasks.tracking.config.wbot_20dof_e1_mimic  # noqa: F401
import src.tasks.tracking.config.wbot_20dof_vel_style  # noqa: F401
import src.tasks.davis.config.wbot_20dof  # noqa: F401

import src.tasks.standing.config.adam_sp  # noqa: F401
import src.tasks.standing.config.x2  # noqa: F401
import src.tasks.velocity_decoupled.config.x2  # noqa: F401
import src.tasks.velocity_decoupled.config.adam_sp  # noqa: F401

import_packages(__name__, _BLACKLIST_PKGS)
