"""MuJoCo play viewer with stable camera (no TRACKING subtree_com jitter)."""

from __future__ import annotations

import mujoco

from mjlab.viewer.native.viewer import NativeMujocoViewer


class StableCameraNativeViewer(NativeMujocoViewer):
  """Use FREE camera; update ``lookat`` from a body frame origin each frame.

  MuJoCo ``CAMERA_TRACKING`` centers on ``subtree_com[trackbodyid]``, which shifts
  with joint pose, DR (``body_ipos``), and debug meshes. That often looks like the
  view is jumping during AMP play. This viewer keeps azimuth/elevation/distance
  under user control while smoothly following the chosen body origin.
  """

  follow_body_name: str = "torso_link"

  def _setup_camera(self) -> None:
    super()._setup_camera()
    assert self.viewer is not None
    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    self.viewer.cam.trackbodyid = -1
    self.viewer.cam.fixedcamid = -1

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    v = self.viewer
    mjd = self.mjd
    if v is None or mjd is None or not v.is_running():
      return
    try:
      robot = self.env.unwrapped.scene["robot"]
      body_ids, _ = robot.find_bodies(self.follow_body_name)
      if not body_ids:
        return
      bid = robot.indexing.bodies[body_ids[0]].id
      v.cam.lookat[:] = mjd.xpos[bid]
    except (KeyError, IndexError, ValueError):
      return
