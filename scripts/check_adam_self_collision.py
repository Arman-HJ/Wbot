"""List self-contacts at Adam SP locomotion init pose."""
from pathlib import Path

import mujoco

from src.assets.robots.adam_sp.adam_sp_constants import (
  ADAM_SP_XML,
  HOME_KEYFRAME,
  get_spec,
)


def _set_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  assert HOME_KEYFRAME.joint_pos is not None
  for jname, q in HOME_KEYFRAME.joint_pos.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
      data.qpos[model.jnt_qposadr[jid]] = q
  data.qpos[0:3] = HOME_KEYFRAME.pos


def main() -> None:
  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  _set_pose(model, data)
  mujoco.mj_forward(model, data)
  mujoco.mj_collision(model, data)

  print(f"XML: {ADAM_SP_XML.resolve()}")
  print(f"ncon: {data.ncon}")
  for i in range(data.ncon):
    g1 = model.geom(data.contact[i].geom1).name
    g2 = model.geom(data.contact[i].geom2).name
    print(f"  {g1} <-> {g2}")


if __name__ == "__main__":
  main()
