"""Installation script for the 'unitree_rl_mjlab' python package."""

from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).resolve().parent
_RSL_RL = _ROOT.parent / "rsl_rl"

INSTALL_REQUIRES = [
  "mjlab",
  "mujoco-warp",
]

if _RSL_RL.is_dir():
  INSTALL_REQUIRES.append(f"rsl_rl @ file://{_RSL_RL}")
else:
  INSTALL_REQUIRES.append("rsl_rl")

setup(
  name="unitree_rl_mjlab",
  packages=["src"],
  version="0.0.1",
  install_requires=INSTALL_REQUIRES,
)
