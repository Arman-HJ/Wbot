"""football_jax pitch for DAVIS / mjlab.

football_jax layout (world origin = midfield):
  field  x ∈ [-FIELD_X, +FIELD_X], y ∈ [-FIELD_Y, +FIELD_Y]
  goals  at ±(FIELD_X + 0.3) with backs at ±(FIELD_X + 0.65)
  score  into +x past GOAL_X (= FIELD_X)

DAVIS layout (env origin = scoring goal line, mouth opens +x):
  davis_x = fj_x + FIELD_X
  so the pitch spans x ∈ [0, 2*FIELD_X], midfield at FIELD_X, far goal at 2*FIELD_X.
  Scoring goal uses the football_jax −x goal shifted by +FIELD_X.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from src import SRC_PATH

# Match football_jax/fj/env.py
FIELD_X = 7.0
FIELD_Y = 4.5
GOAL_HALF_W_VIS = 1.35  # post centers (fj scene.xml)
GOAL_POST_DEPTH = 0.3
GOAL_BACK_OFFSET = 0.65
GOAL_BOX_HEIGHT = 1.0  # fj box size z=0.5 → full height 1.0

LINES_STL: Path = SRC_PATH / "assets" / "soccer" / "meshes" / "Lines.stl"
assert LINES_STL.exists(), f"missing pitch lines mesh: {LINES_STL}"


def _add_goal_boxes(
  body: mujoco.MjsBody,
  *,
  goal_x: float,
  opening_sign: float,
  prefix: str,
  collide: bool,
  rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> None:
  """U-shaped goal like football_jax (two side posts + back plate).

  ``opening_sign``: +1 → mouth faces +x (field); −1 → mouth faces −x.
  ``goal_x``: goal-line x (posts sit ``GOAL_POST_DEPTH`` behind the line into the net).
  """
  # Net interior is opposite the opening.
  net_dir = -opening_sign
  post_x = goal_x + net_dir * GOAL_POST_DEPTH
  back_x = goal_x + net_dir * GOAL_BACK_OFFSET
  hz = GOAL_BOX_HEIGHT * 0.5
  white = rgba
  contype = 1 if collide else 0
  conaffinity = 1 if collide else 0
  group = 0 if collide else 1
  common = dict(
    rgba=white,
    contype=contype,
    conaffinity=conaffinity,
    group=group,
  )
  if collide:
    common["mass"] = 50.0

  for side, sy in (("l", GOAL_HALF_W_VIS), ("r", -GOAL_HALF_W_VIS)):
    body.add_geom(
      name=f"{prefix}_post_{side}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(GOAL_POST_DEPTH, 0.05, hz),
      pos=(post_x, sy, hz),
      **common,
    )
  body.add_geom(
    name=f"{prefix}_back",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.05, 1.4, hz),
    pos=(back_x, 0.0, hz),
    **common,
  )


def add_goal_posts(
  body: mujoco.MjsBody,
  *,
  goal_x: float = 0.0,
  half_width: float = 1.20,
  height: float = 1.80,
  prefix: str = "goal",
  post_r: float = 0.05,
  collide: bool = True,
  rgba: tuple[float, float, float, float] = (0.95, 0.95, 0.97, 1.0),
) -> None:
  """Free-standing goal frame: 2 posts + crossbar on the goal line (no backboard)."""
  contype = 1 if collide else 0
  conaffinity = 1 if collide else 0
  group = 0 if collide else 1
  base = dict(rgba=rgba, contype=contype, conaffinity=conaffinity, group=group)

  def _geom(name: str, size, pos, quat=None, mass: float | None = None) -> None:
    kw = dict(base)
    if collide and mass is not None:
      kw["mass"] = mass
    if quat is not None:
      kw["quat"] = quat
    body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=size,
      pos=pos,
      **kw,
    )

  _geom(
    f"{prefix}_post_l",
    (post_r, height * 0.5),
    (goal_x, half_width, height * 0.5),
    mass=30.0,
  )
  _geom(
    f"{prefix}_post_r",
    (post_r, height * 0.5),
    (goal_x, -half_width, height * 0.5),
    mass=30.0,
  )
  _geom(
    f"{prefix}_crossbar",
    (post_r, half_width),
    (goal_x, 0.0, height),
    quat=(0.70710678, 0.70710678, 0.0, 0.0),
    mass=20.0,
  )


def get_goal_spec_posts(
  half_width: float = 1.20,
  height: float = 1.80,
) -> mujoco.MjSpec:
  """Scoring goal posts at env origin (mouth +x / field). Freejoint for multi-env pin."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal", pos=(0.0, 0.0, 0.0))
  body.add_freejoint(name="goal_freejoint")
  add_goal_posts(
    body, goal_x=0.0, half_width=half_width, height=height, prefix="goal", collide=True
  )
  return spec


# Back-compat alias (U-box version no longer used for scoring goal).
def get_goal_spec_fj() -> mujoco.MjSpec:
  return get_goal_spec_posts()


def get_field_spec() -> mujoco.MjSpec:
  """Pitch lines + far-end visual goal posts. Lines mesh centered at midfield."""
  spec = mujoco.MjSpec()
  mesh_name = "Lines"
  mesh = spec.add_mesh(name=mesh_name, file=str(LINES_STL))
  del mesh
  spec.assets = {LINES_STL.name: LINES_STL.read_bytes(), str(LINES_STL): LINES_STL.read_bytes()}

  body = spec.worldbody.add_body(name="field", pos=(0.0, 0.0, 0.0))
  body.add_freejoint(name="field_freejoint")
  # Lines.stl spans x∈[-7,7] about its origin → shift by FIELD_X so goal line = 0.
  body.add_geom(
    name="field_lines",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=mesh_name,
    pos=(FIELD_X, 0.0, 0.0),
    contype=0,
    conaffinity=0,
    group=1,
    rgba=(1.0, 1.0, 1.0, 1.0),
  )
  # Far (+x) decorative posts at davis x=2*FIELD_X.
  add_goal_posts(
    body,
    goal_x=2.0 * FIELD_X,
    half_width=1.20,
    height=1.80,
    prefix="far_goal",
    collide=False,
  )
  return spec
