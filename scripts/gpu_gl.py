"""Force headless MuJoCo rendering onto the NVIDIA GPU (EGL).

The conda env ships a Mesa glvnd vendor; without pinning the NVIDIA vendor,
EGL picks llvmpipe (CPU, ~30x slower). Call before importing ``mujoco``.
"""

from __future__ import annotations

import os
from pathlib import Path

NVIDIA_EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")


def use_gpu_egl() -> None:
  os.environ.setdefault("MUJOCO_GL", "egl")
  if os.environ["MUJOCO_GL"] != "egl":
    return
  os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
  if NVIDIA_EGL_VENDOR.is_file():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(NVIDIA_EGL_VENDOR))
