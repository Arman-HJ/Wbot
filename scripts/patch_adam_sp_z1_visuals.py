#!/usr/bin/env python3
"""Optional: swap adam_sp.xml visual geoms to Z1 OBJ assets (collision unchanged).

Default robot XML uses the original STL meshes under meshes/*.STL.
Run this only when z1_assets are present and you want Z1 visuals:

  python scripts/patch_adam_sp_z1_visuals.py

Requires meshes/z1_assets/*.obj. To go back to STL, restore adam_sp.xml from git.
"""

from __future__ import annotations

from pathlib import Path

XML = Path(__file__).resolve().parents[1] / "src/assets/robots/adam_sp/xml/adam_sp.xml"

Z1_ASSET_BLOCK = """    <mesh name="pelvis_0" file="z1_assets/pelvis_0.obj" />
    <mesh name="pelvis_1" file="z1_assets/pelvis_1.obj" />
    <mesh name="hipPitchLeft_0" file="z1_assets/hipPitchLeft_0.obj" />
    <mesh name="hipPitchLeft_1" file="z1_assets/hipPitchLeft_1.obj" />
    <mesh name="hipPitchLeft_2" file="z1_assets/hipPitchLeft_2.obj" />
    <mesh name="hipRollLeft_0" file="z1_assets/hipRollLeft_0.obj" />
    <mesh name="hipRollLeft_2" file="z1_assets/hipRollLeft_2.obj" />
    <mesh name="thighLeft_1" file="z1_assets/thighLeft_1.obj" />
    <mesh name="thighLeft_2" file="z1_assets/thighLeft_2.obj" />
    <mesh name="shinLeft_0" file="z1_assets/shinLeft_0.obj" />
    <mesh name="shinLeft_1" file="z1_assets/shinLeft_1.obj" />
    <mesh name="shinLeft_2" file="z1_assets/shinLeft_2.obj" />
    <mesh name="anklePitchLeft" file="z1_assets/anklePitchLeft.obj" />
    <mesh name="toeLeft_0" file="z1_assets/toeLeft_0.obj" />
    <mesh name="toeLeft_1" file="z1_assets/toeLeft_1.obj" />
    <mesh name="hipPitchRight_0" file="z1_assets/hipPitchRight_0.obj" />
    <mesh name="hipPitchRight_1" file="z1_assets/hipPitchRight_1.obj" />
    <mesh name="hipPitchRight_2" file="z1_assets/hipPitchRight_2.obj" />
    <mesh name="hipRollRight_0" file="z1_assets/hipRollRight_0.obj" />
    <mesh name="hipRollRight_2" file="z1_assets/hipRollRight_2.obj" />
    <mesh name="thighRight_1" file="z1_assets/thighRight_1.obj" />
    <mesh name="thighRight_2" file="z1_assets/thighRight_2.obj" />
    <mesh name="shinRight_0" file="z1_assets/shinRight_0.obj" />
    <mesh name="shinRight_1" file="z1_assets/shinRight_1.obj" />
    <mesh name="shinRight_2" file="z1_assets/shinRight_2.obj" />
    <mesh name="anklePitchRight" file="z1_assets/anklePitchRight.obj" />
    <mesh name="toeRight_0" file="z1_assets/toeRight_0.obj" />
    <mesh name="toeRight_1" file="z1_assets/toeRight_1.obj" />
    <mesh name="waistRoll_link_0" file="z1_assets/waistRoll_link_0.obj" />
    <mesh name="waistRoll_link_1" file="z1_assets/waistRoll_link_1.obj" />
    <mesh name="waistPitch_link_0" file="z1_assets/waistPitch_link_0.obj" />
    <mesh name="waistPitch_link_1" file="z1_assets/waistPitch_link_1.obj" />
    <mesh name="torso_0" file="z1_assets/torso_0.obj" />
    <mesh name="torso_1" file="z1_assets/torso_1.obj" />
    <mesh name="torso_2" file="z1_assets/torso_2.obj" />
    <mesh name="shoulderPitchLeft" file="z1_assets/shoulderPitchLeft.obj" />
    <mesh name="shoulderRollLeft_0" file="z1_assets/shoulderRollLeft_0.obj" />
    <mesh name="shoulderRollLeft_1" file="z1_assets/shoulderRollLeft_1.obj" />
    <mesh name="shoulderYawLeft_0" file="z1_assets/shoulderYawLeft_0.obj" />
    <mesh name="shoulderYawLeft_1" file="z1_assets/shoulderYawLeft_1.obj" />
    <mesh name="elbowLeft_0" file="z1_assets/elbowLeft_0.obj" />
    <mesh name="elbowLeft_1" file="z1_assets/elbowLeft_1.obj" />
    <mesh name="wristYawLeft_0" file="z1_assets/wristYawLeft_0.obj" />
    <mesh name="wristYawLeft_1" file="z1_assets/wristYawLeft_1.obj" />
    <mesh name="wristPitchLeft" file="z1_assets/wristPitchLeft.obj" />
    <mesh name="wristRollLeft_0" file="z1_assets/wristRollLeft_0.obj" />
    <mesh name="wristRollLeft_1" file="z1_assets/wristRollLeft_1.obj" />
    <mesh name="L_thumb_proximal_base" file="z1_assets/L_thumb_proximal_base.obj" />
    <mesh name="L_thumb_proximal_0" file="z1_assets/L_thumb_proximal_0.obj" />
    <mesh name="L_thumb_proximal_1" file="z1_assets/L_thumb_proximal_1.obj" />
    <mesh name="L_thumb_intermediate" file="z1_assets/L_thumb_intermediate.obj" />
    <mesh name="L_thumb_distal_0" file="z1_assets/L_thumb_distal_0.obj" />
    <mesh name="L_thumb_distal_1" file="z1_assets/L_thumb_distal_1.obj" />
    <mesh name="L_index_proximal_0" file="z1_assets/L_index_proximal_0.obj" />
    <mesh name="L_index_proximal_1" file="z1_assets/L_index_proximal_1.obj" />
    <mesh name="L_index_distal_0" file="z1_assets/L_index_distal_0.obj" />
    <mesh name="L_index_distal_1" file="z1_assets/L_index_distal_1.obj" />
    <mesh name="L_middle_proximal_0" file="z1_assets/L_middle_proximal_0.obj" />
    <mesh name="L_middle_proximal_1" file="z1_assets/L_middle_proximal_1.obj" />
    <mesh name="L_middle_distal_0" file="z1_assets/L_middle_distal_0.obj" />
    <mesh name="L_middle_distal_1" file="z1_assets/L_middle_distal_1.obj" />
    <mesh name="L_ring_proximal_0" file="z1_assets/L_ring_proximal_0.obj" />
    <mesh name="L_ring_proximal_1" file="z1_assets/L_ring_proximal_1.obj" />
    <mesh name="L_ring_distal_0" file="z1_assets/L_ring_distal_0.obj" />
    <mesh name="L_ring_distal_1" file="z1_assets/L_ring_distal_1.obj" />
    <mesh name="L_pinky_proximal_0" file="z1_assets/L_pinky_proximal_0.obj" />
    <mesh name="L_pinky_proximal_1" file="z1_assets/L_pinky_proximal_1.obj" />
    <mesh name="L_pinky_distal_0" file="z1_assets/L_pinky_distal_0.obj" />
    <mesh name="L_pinky_distal_1" file="z1_assets/L_pinky_distal_1.obj" />
    <mesh name="shoulderPitchRight" file="z1_assets/shoulderPitchRight.obj" />
    <mesh name="shoulderRollRight_0" file="z1_assets/shoulderRollRight_0.obj" />
    <mesh name="shoulderRollRight_1" file="z1_assets/shoulderRollRight_1.obj" />
    <mesh name="shoulderYawRight_0" file="z1_assets/shoulderYawRight_0.obj" />
    <mesh name="shoulderYawRight_1" file="z1_assets/shoulderYawRight_1.obj" />
    <mesh name="elbowRight_0" file="z1_assets/elbowRight_0.obj" />
    <mesh name="elbowRight_1" file="z1_assets/elbowRight_1.obj" />
    <mesh name="wristYawRight_0" file="z1_assets/wristYawRight_0.obj" />
    <mesh name="wristYawRight_1" file="z1_assets/wristYawRight_1.obj" />
    <mesh name="wristPitchRight" file="z1_assets/wristPitchRight.obj" />
    <mesh name="wristRollRight_0" file="z1_assets/wristRollRight_0.obj" />
    <mesh name="wristRollRight_1" file="z1_assets/wristRollRight_1.obj" />
    <mesh name="R_thumb_proximal_base" file="z1_assets/R_thumb_proximal_base.obj" />
    <mesh name="R_thumb_proximal_0" file="z1_assets/R_thumb_proximal_0.obj" />
    <mesh name="R_thumb_proximal_1" file="z1_assets/R_thumb_proximal_1.obj" />
    <mesh name="R_thumb_intermediate" file="z1_assets/R_thumb_intermediate.obj" />
    <mesh name="R_thumb_distal_0" file="z1_assets/R_thumb_distal_0.obj" />
    <mesh name="R_thumb_distal_1" file="z1_assets/R_thumb_distal_1.obj" />
    <mesh name="R_index_proximal_0" file="z1_assets/R_index_proximal_0.obj" />
    <mesh name="R_index_proximal_1" file="z1_assets/R_index_proximal_1.obj" />
    <mesh name="R_index_distal_0" file="z1_assets/R_index_distal_0.obj" />
    <mesh name="R_index_distal_1" file="z1_assets/R_index_distal_1.obj" />
    <mesh name="R_middle_proximal_0" file="z1_assets/R_middle_proximal_0.obj" />
    <mesh name="R_middle_proximal_1" file="z1_assets/R_middle_proximal_1.obj" />
    <mesh name="R_middle_distal_0" file="z1_assets/R_middle_distal_0.obj" />
    <mesh name="R_middle_distal_1" file="z1_assets/R_middle_distal_1.obj" />
    <mesh name="R_ring_proximal_0" file="z1_assets/R_ring_proximal_0.obj" />
    <mesh name="R_ring_proximal_1" file="z1_assets/R_ring_proximal_1.obj" />
    <mesh name="R_ring_distal_0" file="z1_assets/R_ring_distal_0.obj" />
    <mesh name="R_ring_distal_1" file="z1_assets/R_ring_distal_1.obj" />
    <mesh name="R_pinky_proximal_0" file="z1_assets/R_pinky_proximal_0.obj" />
    <mesh name="R_pinky_proximal_1" file="z1_assets/R_pinky_proximal_1.obj" />
    <mesh name="R_pinky_distal_0" file="z1_assets/R_pinky_distal_0.obj" />
    <mesh name="R_pinky_distal_1" file="z1_assets/R_pinky_distal_1.obj" />
    <mesh name="neckYaw_link" file="z1_assets/neckYaw_link.obj" />
    <mesh name="neckPitch_link" file="z1_assets/neckPitch_link.obj" />"""

# old -> new visual geom block (collision lines untouched)
REPLACEMENTS: list[tuple[str, str]] = [
  (
    '      <geom class="visual" material="black" mesh="pelvis" />',
    '      <geom class="visual" material="silver" mesh="pelvis_1" pos="0.0069 0 0.040" />',
  ),
  (
    '        <geom class="visual" material="black" mesh="hipPitchLeft" />',
    '        <geom class="visual" material="silver" mesh="hipPitchLeft_2" />',
  ),
  (
    '          <geom class="visual" mesh="hipRollLeft" />',
    '          <geom class="visual" material="silver" mesh="hipRollLeft_0" />\n          <geom class="visual" material="black" mesh="hipRollLeft_2" />',
  ),
  (
    '            <geom class="visual" mesh="thighLeft" />',
    '            <geom class="visual" material="silver" mesh="thighLeft_1" />\n            <geom class="visual" material="silver" mesh="thighLeft_2" />',
  ),
  (
    '              <geom class="visual" mesh="shinLeft" />',
    '              <geom class="visual" material="silver" mesh="shinLeft_0" />\n              <geom class="visual" material="silver" mesh="shinLeft_2" />',
  ),
  (
    '                <geom class="visual" mesh="anklePitchLeft" />',
    '                <geom class="visual" material="silver" mesh="anklePitchLeft" />',
  ),
  (
    '                  <geom class="visual" material="black" mesh="toeLeft" />',
    '                  <geom class="visual" material="silver" mesh="toeLeft_0" />\n                  <geom class="visual" material="black" mesh="toeLeft_1" />',
  ),
  (
    '        <geom class="visual" material="black" mesh="hipPitchRight" />',
    '        <geom class="visual" material="silver" mesh="hipPitchRight_2" />',
  ),
  (
    '          <geom class="visual" mesh="hipRollRight" />',
    '          <geom class="visual" material="silver" mesh="hipRollRight_0" />\n          <geom class="visual" material="black" mesh="hipRollRight_2" />',
  ),
  (
    '            <geom class="visual" mesh="thighRight" />',
    '            <geom class="visual" material="silver" mesh="thighRight_1" />\n            <geom class="visual" material="silver" mesh="thighRight_2" />',
  ),
  (
    '              <geom class="visual" mesh="shinRight" />',
    '              <geom class="visual" material="silver" mesh="shinRight_0" />\n              <geom class="visual" material="silver" mesh="shinRight_1" />\n              <geom class="visual" material="silver" mesh="shinRight_2" />',
  ),
  (
    '                <geom class="visual" mesh="anklePitchRight" />',
    '                <geom class="visual" material="silver" mesh="anklePitchRight" />',
  ),
  (
    '                  <geom class="visual" material="black" mesh="toeRight" />',
    '                  <geom class="visual" material="black" mesh="toeRight_0" />\n                  <geom class="visual" material="silver" mesh="toeRight_1" />',
  ),
  (
    '        <geom class="visual" mesh="waistRoll_link" />',
    '        <geom class="visual" material="black" mesh="waistRoll_link_0" />\n        <geom class="visual" material="silver" mesh="waistRoll_link_1" />',
  ),
  (
    '          <geom class="visual" mesh="waistPitch_link" />',
    '          <geom class="visual" material="black" mesh="waistPitch_link_0" />\n          <geom class="visual" material="silver" mesh="waistPitch_link_1" />',
  ),
  (
    '            <geom class="visual" mesh="torso" />',
    '            <geom class="visual" material="silver" mesh="torso_0" />\n            <geom class="visual" material="silver" mesh="torso_1" />\n            <geom class="visual" material="silver" mesh="torso_2" />',
  ),
  (
    '              <geom class="visual" mesh="shoulderPitchLeft" />',
    '              <geom class="visual" material="silver" mesh="shoulderPitchLeft" />',
  ),
  (
    '                <geom class="visual" mesh="shoulderRollLeft" />',
    '                <geom class="visual" material="silver" mesh="shoulderRollLeft_0" />\n                <geom class="visual" material="black" mesh="shoulderRollLeft_1" />',
  ),
  (
    '                  <geom class="visual" mesh="shoulderYawLeft" />',
    '                  <geom class="visual" material="black" mesh="shoulderYawLeft_0" />\n                  <geom class="visual" material="silver" mesh="shoulderYawLeft_1" />',
  ),
  (
    '                    <geom class="visual" mesh="elbowLeft" />',
    '                    <geom class="visual" material="black" mesh="elbowLeft_0" />\n                    <geom class="visual" material="silver" mesh="elbowLeft_1" />',
  ),
  (
    '                      <geom class="visual" mesh="wristYawLeft" />',
    '                      <geom class="visual" material="black" mesh="wristYawLeft_0" />\n                      <geom class="visual" material="silver" mesh="wristYawLeft_1" />',
  ),
  (
    '                        <geom class="visual" mesh="wristPitchLeft" />',
    '                        <geom class="visual" material="silver" mesh="wristPitchLeft" />',
  ),
  (
    '                          <geom class="visual" mesh="wristRollLeft" />',
    '                          <geom class="visual" material="black" mesh="wristRollLeft_0" />\n                          <geom class="visual" material="silver" mesh="wristRollLeft_1" />',
  ),
  (
    '              <geom class="visual" mesh="shoulderPitchRight" />',
    '              <geom class="visual" material="silver" mesh="shoulderPitchRight" />',
  ),
  (
    '                <geom class="visual" mesh="shoulderRollRight" />',
    '                <geom class="visual" material="black" mesh="shoulderRollRight_0" />\n                <geom class="visual" material="silver" mesh="shoulderRollRight_1" />',
  ),
  (
    '                  <geom class="visual" mesh="shoulderYawRight" />',
    '                  <geom class="visual" material="silver" mesh="shoulderYawRight_0" />\n                  <geom class="visual" material="black" mesh="shoulderYawRight_1" />',
  ),
  (
    '                    <geom class="visual" mesh="elbowRight" />',
    '                    <geom class="visual" material="silver" mesh="elbowRight_0" />\n                    <geom class="visual" material="black" mesh="elbowRight_1" />',
  ),
  (
    '                      <geom class="visual" mesh="wristYawRight" />',
    '                      <geom class="visual" material="silver" mesh="wristYawRight_0" />\n                      <geom class="visual" material="black" mesh="wristYawRight_1" />',
  ),
  (
    '                        <geom class="visual" mesh="wristPitchRight" />',
    '                        <geom class="visual" material="silver" mesh="wristPitchRight" />',
  ),
  (
    '                          <geom class="visual" mesh="wristRollRight" />',
    '                          <geom class="visual" material="silver" mesh="wristRollRight_0" />\n                          <geom class="visual" material="black" mesh="wristRollRight_1" />',
  ),
  (
    '              <geom class="visual" mesh="neckYaw_link" pos="0.015 0 -0.0139" />',
    '              <geom class="visual" material="silver" mesh="neckYaw_link" pos="0.015 0 -0.0139" />',
  ),
  (
    '                <geom class="visual" mesh="neckPitch_link" pos="0 0.0018 0.0217" />',
    '                <geom class="visual" material="silver" mesh="neckPitch_link" pos="0 0.0018 0.0217" />',
  ),
]

FINGER_REPLACEMENTS: list[tuple[str, str]] = [
  ('mesh="L_thumb_proximal_base"', 'class="visual" material="silver" mesh="L_thumb_proximal_base"'),
  ('mesh="L_thumb_proximal"', 'class="visual" material="silver" mesh="L_thumb_proximal_0"'),
  ('mesh="L_thumb_intermediate"', 'class="visual" material="silver" mesh="L_thumb_intermediate"'),
  ('mesh="L_thumb_distal"', 'class="visual" material="black" mesh="L_thumb_distal_0"'),
  ('mesh="L_index_proximal"', 'class="visual" material="silver" mesh="L_index_proximal_0"'),
  ('mesh="L_index_distal"', 'class="visual" material="black" mesh="L_index_distal_0"'),
  ('mesh="L_middle_proximal"', 'class="visual" material="silver" mesh="L_middle_proximal_0"'),
  ('mesh="L_middle_distal"', 'class="visual" material="black" mesh="L_middle_distal_0"'),
  ('mesh="L_ring_proximal"', 'class="visual" material="black" mesh="L_ring_proximal_0"'),
  ('mesh="L_ring_distal"', 'class="visual" material="black" mesh="L_ring_distal_0"'),
  ('mesh="L_pinky_proximal"', 'class="visual" material="silver" mesh="L_pinky_proximal_0"'),
  ('mesh="L_pinky_distal"', 'class="visual" material="silver" mesh="L_pinky_distal_0"'),
  ('mesh="R_thumb_proximal_base"', 'class="visual" material="silver" mesh="R_thumb_proximal_base"'),
  ('mesh="R_thumb_proximal"', 'class="visual" material="black" mesh="R_thumb_proximal_0"'),
  ('mesh="R_thumb_intermediate"', 'class="visual" material="silver" mesh="R_thumb_intermediate"'),
  ('mesh="R_thumb_distal"', 'class="visual" material="silver" mesh="R_thumb_distal_0"'),
  ('mesh="R_index_proximal"', 'class="visual" material="black" mesh="R_index_proximal_0"'),
  ('mesh="R_index_distal"', 'class="visual" material="black" mesh="R_index_distal_0"'),
  ('mesh="R_middle_proximal"', 'class="visual" material="black" mesh="R_middle_proximal_0"'),
  ('mesh="R_middle_distal"', 'class="visual" material="black" mesh="R_middle_distal_0"'),
  ('mesh="R_ring_proximal"', 'class="visual" material="silver" mesh="R_ring_proximal_0"'),
  ('mesh="R_ring_distal"', 'class="visual" material="black" mesh="R_ring_distal_0"'),
  ('mesh="R_pinky_proximal"', 'class="visual" material="black" mesh="R_pinky_proximal_0"'),
  ('mesh="R_pinky_distal"', 'class="visual" material="black" mesh="R_pinky_distal_0"'),
]

OLD_ASSET_START = '    <mesh name="pelvis" file="pelvis.STL" />'
OLD_ASSET_END = '    <mesh name="neckPitch_link" file="neckPitch_link.obj" />'


def main() -> None:
  text = XML.read_text(encoding="utf-8")

  start = text.index(OLD_ASSET_START)
  end = text.index(OLD_ASSET_END) + len(OLD_ASSET_END)
  text = text[:start] + Z1_ASSET_BLOCK + text[end:]

  for old, new in REPLACEMENTS:
    if old not in text:
      raise SystemExit(f"missing visual block: {old[:60]!r}...")
    text = text.replace(old, new, 1)

  for old, new in FINGER_REPLACEMENTS:
    needle = f'<geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.752941 0.752941 0.752941 1" {old} />'
    repl = f"<geom {new} />"
    if needle not in text:
      raise SystemExit(f"missing finger geom: {old}")
    text = text.replace(needle, repl, 1)

  # Add second-part finger meshes after first-part where Z1 has dual meshes.
  dual_fingers = [
    ('mesh="L_thumb_proximal_0"', 'L_thumb_proximal_1', 'black'),
    ('mesh="L_thumb_distal_0"', 'L_thumb_distal_1', 'silver'),
    ('mesh="L_index_proximal_0"', 'L_index_proximal_1', 'black'),
    ('mesh="L_index_distal_0"', 'L_index_distal_1', 'silver'),
    ('mesh="L_middle_proximal_0"', 'L_middle_proximal_1', 'black'),
    ('mesh="L_middle_distal_0"', 'L_middle_distal_1', 'silver'),
    ('mesh="L_ring_proximal_0"', 'L_ring_proximal_1', 'silver'),
    ('mesh="L_ring_distal_0"', 'L_ring_distal_1', 'silver'),
    ('mesh="L_pinky_proximal_0"', 'L_pinky_proximal_1', 'black'),
    ('mesh="L_pinky_distal_0"', 'L_pinky_distal_1', 'black'),
    ('mesh="R_thumb_proximal_0"', 'R_thumb_proximal_1', 'silver'),
    ('mesh="R_thumb_distal_0"', 'R_thumb_distal_1', 'black'),
    ('mesh="R_index_proximal_0"', 'R_index_proximal_1', 'silver'),
    ('mesh="R_index_distal_0"', 'R_index_distal_1', 'silver'),
    ('mesh="R_middle_proximal_0"', 'R_middle_proximal_1', 'silver'),
    ('mesh="R_middle_distal_0"', 'R_middle_distal_1', 'silver'),
    ('mesh="R_ring_proximal_0"', 'R_ring_proximal_1', 'black'),
    ('mesh="R_ring_distal_0"', 'R_ring_distal_1', 'silver'),
    ('mesh="R_pinky_proximal_0"', 'R_pinky_proximal_1', 'silver'),
    ('mesh="R_pinky_distal_0"', 'R_pinky_distal_1', 'silver'),
  ]
  for first_attr, second_mesh, mat in dual_fingers:
    old_line = f'<geom class="visual" material="{mat.split("_")[0] if False else mat}" {first_attr} />'
    # build from actual first line in file
    import re

    pat = re.compile(rf'(<geom class="visual" material="[^"]+" {re.escape(first_attr)} />)')
    m = pat.search(text)
    if not m:
      raise SystemExit(f"missing dual finger first geom: {first_attr}")
    first_line = m.group(1)
    mat_match = re.search(r'material="([^"]+)"', first_line)
    material = mat_match.group(1) if mat_match else mat
    text = text.replace(first_line, first_line + f'\n                            <geom class="visual" material="{mat}" mesh="{second_mesh}" />', 1)

  XML.write_text(text, encoding="utf-8")
  print(f"Patched {XML}")


if __name__ == "__main__":
  main()
