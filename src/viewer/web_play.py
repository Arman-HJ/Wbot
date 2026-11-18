"""Launch the WASM web play demo from a training run (pack + Vite).

Used by ``scripts/play.py`` / ``play_amp.py`` when ``--viewer wasm``.
Reuses ``--viser-host`` / ``--viser-port`` so cmd.sh only swaps ``viser`` → ``wasm``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB_DEMO = ROOT / "web_demo"
PACK_OUT = WEB_DEMO / "public" / "packs"

# Match play-time SoleHeight / AnkleRollForward clamps.
_SOLE_HEIGHT_MAX_M = 0.10
_ANKLE_ROLL_FORWARD_MAX_M = 0.020
_FOOT_FROMTO = re.compile(
  r'(<geom\b[^>]*\bname="(?:left|right)_foot[1-7]_collision"[^>]*?\bfromto=")([^"]+)(")'
)
# Foot plate visual on ankle_roll (moves aft with collision).
_FOOT_ROLL_VISUAL = re.compile(
  r'<geom\b([^>]*\bclass="visual"[^>]*\bmesh="(?:left|right)_ankle_roll_link"[^>]*)/>'
)


def _patch_scene_foot_offsets(
  scene_xml: Path,
  *,
  sole_height: float = 0.0,
  ankle_roll_forward: float = 0.0,
) -> None:
  """Shift packed foot capsules + foot plate mesh aft together (−X).

  - ``ankle_roll_forward``: collision + ankle_roll mesh aft (−X), clamp [0, 20 mm]
  - ``sole_height``: foot down (−Z), clamp [0, 100 mm]
  """
  dx = float(max(0.0, min(_ANKLE_ROLL_FORWARD_MAX_M, ankle_roll_forward)))
  dz = float(max(0.0, min(_SOLE_HEIGHT_MAX_M, sole_height)))
  if dx < 1e-12 and dz < 1e-12:
    return
  text = scene_xml.read_text()

  def _repl_fromto(m: re.Match[str]) -> str:
    parts = m.group(2).split()
    if len(parts) != 6:
      return m.group(0)
    vals = [float(p) for p in parts]
    vals[0] -= dx
    vals[3] -= dx
    vals[2] -= dz
    vals[5] -= dz
    return m.group(1) + " ".join(f"{v:.8g}" for v in vals) + m.group(3)

  new_text, n_col = _FOOT_FROMTO.subn(_repl_fromto, text)

  def _repl_roll_visual(m: re.Match[str]) -> str:
    attrs = m.group(1)
    attrs = re.sub(r'\spos="[^"]*"', "", attrs)
    pos = f' pos="{-dx:.8g} 0 {-dz:.8g}"'
    return f"<geom{pos}{attrs}/>"

  new_text, n_vis = _FOOT_ROLL_VISUAL.subn(_repl_roll_visual, new_text)
  if n_col == 0:
    print(f"[web] WARN: no foot fromto geoms to patch in {scene_xml.name}")
  if dx > 1e-12 and n_vis == 0:
    print(f"[web] WARN: no ankle_roll visual meshes to patch in {scene_xml.name}")
  scene_xml.write_text(new_text)
  print(
    f"[web] foot offsets: ankle_roll_forward={dx * 1000:.1f} mm, "
    f"sole_height={dz * 1000:.1f} mm  "
    f"(collision_aft={n_col}, roll_mesh_aft={n_vis})",
    flush=True,
  )


def _port_inodes(port: int) -> set[str]:
  """Inodes of LISTEN sockets on ``port`` (IPv4/IPv6)."""
  inodes: set[str] = set()
  for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
      lines = Path(path).read_text().splitlines()[1:]
    except OSError:
      continue
    for line in lines:
      parts = line.split()
      if len(parts) < 10:
        continue
      local = parts[1]
      state = parts[3]
      inode = parts[9]
      try:
        p = int(local.rsplit(":", 1)[1], 16)
      except ValueError:
        continue
      if p == port and state == "0A":  # TCP_LISTEN
        inodes.add(inode)
  return inodes


def _pids_holding_inodes(inodes: set[str]) -> list[int]:
  if not inodes:
    return []
  pids: list[int] = []
  proc = Path("/proc")
  for entry in proc.iterdir():
    if not entry.name.isdigit():
      continue
    fd_dir = entry / "fd"
    try:
      for fd in fd_dir.iterdir():
        try:
          target = os.readlink(fd)
        except OSError:
          continue
        if target.startswith("socket:[") and target[8:-1] in inodes:
          pids.append(int(entry.name))
          break
    except OSError:
      continue
  return pids


def _port_free(port: int) -> bool:
  for host in ("127.0.0.1", "0.0.0.0"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      s.bind((host, port))
    except OSError:
      return False
    finally:
      s.close()
  return True


def _free_or_pick_port(port: int, *, host: str) -> int:
  """Kill our own listeners on ``port``; if still busy, pick the next free port."""
  # Prefer killing leftover vite / npm from this user.
  subprocess.run(
    ["pkill", "-f", f"vite.*--port {port}"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )
  subprocess.run(
    ["pkill", "-f", f"npm run dev.*--port {port}"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )
  time.sleep(0.3)

  inodes = _port_inodes(port)
  for pid in _pids_holding_inodes(inodes):
    try:
      # Only signal processes we own.
      if Path(f"/proc/{pid}").stat().st_uid != os.getuid():
        continue
      os.kill(pid, signal.SIGTERM)
      print(f"[web] freed port {port}: killed pid {pid}")
    except OSError:
      pass
  time.sleep(0.4)

  if _port_free(port):
    return port

  # Another user (or unkillable process) still holds it — scan upward.
  for cand in range(port + 1, port + 50):
    if _port_free(cand):
      print(
        f"[web] port {port} busy (not ours); using {cand} instead. "
        f"Open http://{host if host not in ('0.0.0.0', '::') else '127.0.0.1'}:{cand}/"
      )
      return cand
  raise RuntimeError(f"No free port near {port}")


def _resolve_run_and_preferred(
  path: Path,
) -> tuple[Path, Path | None]:
  """Accept a training run dir or a ``model_*.pt`` under it.

  Returns ``(run_dir, preferred_pt)``. When ``path`` is a run directory,
  ``preferred_pt`` is ``None`` so the packer defaults to the latest step;
  the web UI lists every ``model_*.pt`` for switching.
  """
  p = path.expanduser().resolve()
  if p.is_dir():
    run_dir = p
    preferred: Path | None = None
  elif p.is_file():
    run_dir = p.parent
    preferred = p
  else:
    raise FileNotFoundError(
      f"Checkpoint / run dir not found: {p} "
      "(pass a run directory or a model_*.pt under it)."
    )
  if not (run_dir / "params" / "deployment_alignment.yaml").is_file():
    raise FileNotFoundError(
      f"Missing params/deployment_alignment.yaml under {run_dir} "
      "(web play packs from a full training run dir)."
    )
  return run_dir, preferred


def _pack_run(
  run_dir: Path,
  motion_file: str | Path | None = None,
  checkpoint_file: str | Path | None = None,
) -> dict:
  # Import packer from scripts/ (not a package).
  scripts_dir = str(ROOT / "scripts")
  if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
  import pack_web_demo as packer  # type: ignore

  manifest = packer.pack_run(
    run_dir,
    PACK_OUT,
    motion_file=motion_file,
    checkpoint_file=checkpoint_file,
  )
  # Keep Pack dropdown = this run only.
  ok_ids = {manifest["id"]}
  for pack_dir in list(PACK_OUT.glob("*")):
    if pack_dir.is_dir() and pack_dir.name not in ok_ids:
      shutil.rmtree(pack_dir)
      print(f"[web] prune {pack_dir.name}")
  packer.write_index(PACK_OUT, [manifest])
  return manifest


def launch_web_play(
  *,
  checkpoint_file: str | Path,
  host: str = "127.0.0.1",
  port: int = 5173,
  motion_file: str | Path | None = None,
  sole_height: float = 0.0,
  ankle_roll_forward: float = 0.0,
) -> None:
  """Pack a run (from dir or ``model_*.pt``), start Vite, block until Ctrl+C.

  Prefer passing the run directory: every ``model_*.pt`` is exported and the
  page checkpoint dropdown syncs the step numbers (default = latest).
  """
  run_dir, preferred_pt = _resolve_run_and_preferred(Path(checkpoint_file))
  print(f"[web] packing run: {run_dir}")
  if preferred_pt is None:
    print("[web] no model_*.pt specified → default = latest; page lists all steps")
  manifest = _pack_run(
    run_dir, motion_file=motion_file, checkpoint_file=preferred_pt
  )
  pack_dir = PACK_OUT / str(manifest["id"])
  scene_xml = pack_dir / "scene.xml"
  if scene_xml.is_file():
    _patch_scene_foot_offsets(
      scene_xml,
      sole_height=sole_height,
      ankle_roll_forward=ankle_roll_forward,
    )
    man_path = pack_dir / "manifest.json"
    if man_path.is_file():
      man = json.loads(man_path.read_text())
      man["sole_height"] = float(max(0.0, min(_SOLE_HEIGHT_MAX_M, sole_height)))
      man["ankle_roll_forward"] = float(
        max(0.0, min(_ANKLE_ROLL_FORWARD_MAX_M, ankle_roll_forward))
      )
      man_path.write_text(json.dumps(man, indent=2) + "\n")
  print(
    f"[web] packed id={manifest['id']}  task={manifest.get('task_id')}  "
    f"profile={manifest.get('play_profile')}"
  )

  if not (WEB_DEMO / "node_modules").is_dir():
    print("[web] npm install (first time)…")
    subprocess.check_call(["npm", "install"], cwd=str(WEB_DEMO))

  bind_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
  port = _free_or_pick_port(int(port), host=bind_host)
  url = f"http://{bind_host}:{port}/"
  print(f"[web] starting Vite → {url}")

  # Watch the run dir for new model_*.pt while Vite is up; export ONNX in place
  # so the page can refresh the checkpoint dropdown without restarting play.
  stop_watch = threading.Event()
  pack_id = str(manifest["id"])

  def _watch_checkpoints() -> None:
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
      sys.path.insert(0, scripts_dir)
    import pack_web_demo as packer  # type: ignore

    pack_dir = PACK_OUT / pack_id
    # First poll after a short delay so initial pack + Vite settle.
    while not stop_watch.wait(8.0):
      try:
        n = packer.sync_new_checkpoints(run_dir, pack_dir)
        if n:
          print(f"[web] exported {n} new checkpoint(s) → refresh page dropdown")
      except Exception as exc:
        print(f"[web] checkpoint sync: {exc}")

  watch_thread = threading.Thread(
    target=_watch_checkpoints, name="web-ckpt-watch", daemon=True
  )
  watch_thread.start()
  print(f"[web] watching {run_dir.name} for new model_*.pt (every ~8s)")

  env = os.environ.copy()
  proc = subprocess.Popen(
    ["npm", "run", "dev", "--", "--host", host, "--port", str(port), "--strictPort"],
    cwd=str(WEB_DEMO),
    env=env,
  )
  try:
    proc.wait()
  except KeyboardInterrupt:
    print("\n[web] stopping…")
    proc.terminate()
    try:
      proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
      proc.kill()
  finally:
    stop_watch.set()
