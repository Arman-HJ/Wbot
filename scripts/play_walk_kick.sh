#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${WALK_KICK_PYTHON:-/home/songhat/miniconda3/envs/env_mjlab/bin/python}"
cd "$project_dir"
exec "$python_bin" -u scripts/play_walk_kick.py "$@"
