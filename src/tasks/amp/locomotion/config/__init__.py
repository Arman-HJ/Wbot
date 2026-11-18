"""Robot-specific AMP locomotion task configurations."""

# E1 config modules remain importable for AmpPpoAlgorithmCfg inheritance,
# but are not registered here (no E1 assets in this repo).
from . import wbot_12dof as wbot_12dof
from . import wbot_20dof as wbot_20dof
