"""Runtime shims so ``adam_sp_constants`` works without editing that file."""

from __future__ import annotations

from pathlib import Path

import mjlab.utils.os as mjlab_os

if not hasattr(mjlab_os, "update_assets"):

  def update_assets(
    assets: dict[str, bytes], source_dir: Path | str, meshdir: str
  ) -> None:
    source = Path(source_dir)
    if not source.exists():
      return
    for file in source.rglob("*"):
      if not file.is_file():
        continue
      rel_path = file.relative_to(source).as_posix()
      assets[rel_path] = file.read_bytes()
      if meshdir:
        assets[f"{meshdir.rstrip('/')}/{rel_path}"] = assets[rel_path]

  mjlab_os.update_assets = update_assets  # type: ignore[attr-defined]
