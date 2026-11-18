"""Tune PND Adam Pro manual PD gains in-place.

Usage examples:
  # Preview only (no file change)
  python scripts/tune_pnd_pd.py --scale-kp 0.8 --scale-kd 0.8

  # Set one group exactly, and apply changes
  python scripts/tune_pnd_pd.py --set ADAM_PRO_HIP_PITCH_KNEE:180:7 --apply

  # Per-group scaling + global scaling, then apply
  python scripts/tune_pnd_pd.py \
    --scale-kp 0.9 --scale-kd 0.9 \
    --group-scale ADAM_PRO_ANKLE_ROLL:0.7:0.8 \
    --group-scale ADAM_PRO_WRIST_PR:0.6:0.7 \
    --apply
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "src" / "assets" / "robots" / "adam_sp" / "adam_sp_constants.py"

_MANUAL_PD_RE = re.compile(
  r"^(?P<name>ADAM_PRO_[A-Z0-9_]+)\s*=\s*_manual_pd\("
  r"(?P<expr>\(.*\)),\s*"
  r"(?P<effort>[-+]?\d*\.?\d+),\s*"
  r"(?P<kp>[-+]?\d*\.?\d+),\s*"
  r"(?P<kd>[-+]?\d*\.?\d+)"
  r"\)\s*$"
)


@dataclass
class PdRow:
  name: str
  expr: str
  effort: float
  kp_old: float
  kd_old: float
  kp_new: float
  kd_new: float
  line_idx: int

  @property
  def action_scale_old(self) -> float:
    return min(0.12, 0.25 * self.effort / self.kp_old)

  @property
  def action_scale_new(self) -> float:
    return min(0.12, 0.25 * self.effort / self.kp_new)


def _parse_triplet(s: str, tag: str) -> tuple[str, float, float]:
  parts = s.split(":")
  if len(parts) != 3:
    raise ValueError(f"{tag} format error: '{s}', expected NAME:KP:KD")
  name = parts[0].strip()
  kp = float(parts[1])
  kd = float(parts[2])
  if kp <= 0 or kd < 0:
    raise ValueError(f"{tag} invalid gains for '{name}': kp>0 and kd>=0 required")
  return name, kp, kd


def _parse_overrides(values: list[str], tag: str) -> dict[str, tuple[float, float]]:
  out: dict[str, tuple[float, float]] = {}
  for v in values:
    name, kp, kd = _parse_triplet(v, tag)
    out[name] = (kp, kd)
  return out


def _fmt_gain(v: float) -> str:
  return f"{v:.6f}".rstrip("0").rstrip(".")


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Tune PND Adam Pro manual PD gains.")
  p.add_argument(
    "--target",
    type=Path,
    default=DEFAULT_TARGET,
    help=f"Target constants file (default: {DEFAULT_TARGET})",
  )
  p.add_argument("--scale-kp", type=float, default=1.0, help="Global Kp scale.")
  p.add_argument("--scale-kd", type=float, default=1.0, help="Global Kd scale.")
  p.add_argument(
    "--set",
    action="append",
    default=[],
    metavar="NAME:KP:KD",
    help="Set exact gains for one group. Repeatable.",
  )
  p.add_argument(
    "--group-scale",
    action="append",
    default=[],
    metavar="NAME:KP_SCALE:KD_SCALE",
    help="Extra per-group scale. Repeatable.",
  )
  p.add_argument(
    "--apply",
    action="store_true",
    help="Write changes back to file. Without this flag, preview only.",
  )
  return p


def main() -> None:
  args = _build_argparser().parse_args()
  target: Path = args.target
  if not target.exists():
    raise FileNotFoundError(f"Target file not found: {target}")

  if args.scale_kp <= 0 or args.scale_kd < 0:
    raise ValueError("Global scales must satisfy: scale-kp>0 and scale-kd>=0")

  set_overrides = _parse_overrides(args.set, "--set")
  group_scales = _parse_overrides(args.group_scale, "--group-scale")

  lines = target.read_text(encoding="utf-8").splitlines()
  rows: list[PdRow] = []

  for i, line in enumerate(lines):
    m = _MANUAL_PD_RE.match(line.strip())
    if not m:
      continue
    name = m.group("name")
    expr = m.group("expr")
    effort = float(m.group("effort"))
    kp_old = float(m.group("kp"))
    kd_old = float(m.group("kd"))

    kp_new = kp_old * args.scale_kp
    kd_new = kd_old * args.scale_kd

    if name in group_scales:
      gkp, gkd = group_scales[name]
      kp_new *= gkp
      kd_new *= gkd

    if name in set_overrides:
      kp_new, kd_new = set_overrides[name]

    if kp_new <= 0 or kd_new < 0:
      raise ValueError(f"Invalid result gains for {name}: kp={kp_new}, kd={kd_new}")

    rows.append(
      PdRow(
        name=name,
        expr=expr,
        effort=effort,
        kp_old=kp_old,
        kd_old=kd_old,
        kp_new=kp_new,
        kd_new=kd_new,
        line_idx=i,
      )
    )

  if not rows:
    raise RuntimeError(f"No _manual_pd definitions found in: {target}")

  print("== PND PD Preview ==")
  print(f"target: {target}")
  print(f"apply: {args.apply}")
  print(
    "global scales:",
    f"kp={args.scale_kp:.4f}",
    f"kd={args.scale_kd:.4f}",
  )
  if set_overrides:
    print("set overrides:", ", ".join(sorted(set_overrides)))
  if group_scales:
    print("group scales:", ", ".join(sorted(group_scales)))
  print("")

  header = (
    f"{'GROUP':34s} {'Kp(old->new)':26s} {'Kd(old->new)':26s} {'Scale(old->new)':26s}"
  )
  print(header)
  print("-" * len(header))
  for r in rows:
    kp_txt = f"{r.kp_old:.4f}->{r.kp_new:.4f}"
    kd_txt = f"{r.kd_old:.4f}->{r.kd_new:.4f}"
    sc_txt = f"{r.action_scale_old:.5f}->{r.action_scale_new:.5f}"
    print(f"{r.name:34s} {kp_txt:26s} {kd_txt:26s} {sc_txt:26s}")

  if not args.apply:
    print("\nPreview mode only. Add --apply to write changes.")
    return

  for r in rows:
    lines[r.line_idx] = (
      f"{r.name} = _manual_pd({r.expr}, {_fmt_gain(r.effort)}, "
      f"{_fmt_gain(r.kp_new)}, {_fmt_gain(r.kd_new)})"
    )
  target.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print("\nApplied changes successfully.")


if __name__ == "__main__":
  main()

