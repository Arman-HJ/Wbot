#!/usr/bin/env python3
"""本机手柄 → UDP → X2 机上 /joy，供 deploy_real（ros_joy）使用。

X2 启动顺序（见 README）：
  1) 机上 bash tools/app_stop.sh
  2) 机上 tools/x2_joy_ros_pub.py（或 bash tools/run_x2_joy_pub.sh）
     本机本脚本发手柄
  3) 机上 ./run.sh

推荐 UDP:
  # 机上（脚本内已激活 venv）
  bash tools/run_x2_joy_pub.sh
  # 本机（HOST 换成 X2 IP）
  python3 tools/x2_joy_bridge.py --via udp --host 10.233.0.199 --port 31337

备选 cobridge:
  python3 tools/x2_joy_bridge.py --via cobridge --host <X2_IP>

机上配置: keyboard_vel.backend=ros_joy, joy_topic=/joy
"""
from __future__ import annotations

import argparse
import array
import fcntl
import json
import os
import socket
import struct
import sys
import time
from typing import List, Optional, Protocol, Tuple

# linux/joystick.h
JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

JOY_SCHEMA = """\
std_msgs/Header header
float32[] axes
int32[] buttons
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


class JoySink(Protocol):
    def publish(self, axes: List[float], buttons: List[int]) -> None: ...

    def close(self) -> None: ...


def _ioctl_u8(fd: int, req: int) -> int:
    buf = array.array("B", [0])
    fcntl.ioctl(fd, req, buf, True)
    return int(buf[0])


def _ioctl_name(fd: int, maxlen: int = 128) -> str:
    req = 0x80006A13 | ((maxlen & 0x1FFF) << 16)
    buf = array.array("B", [0] * maxlen)
    try:
        fcntl.ioctl(fd, req, buf, True)
    except OSError:
        return "joystick"
    raw = bytes(buf).split(b"\x00", 1)[0]
    return raw.decode("utf-8", errors="replace") or "joystick"


def open_joystick(path: str) -> Tuple[int, str, int, int]:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    name = _ioctl_name(fd)
    n_axes = _ioctl_u8(fd, JSIOCGAXES)
    n_btns = _ioctl_u8(fd, JSIOCGBUTTONS)
    return fd, name, n_axes, n_btns


def find_default_js() -> Optional[str]:
    for i in range(8):
        p = f"/dev/input/js{i}"
        if os.path.exists(p):
            return p
    return None


def read_js_events(fd: int, axes: List[float], buttons: List[int]) -> None:
    # struct js_event { uint32 time; int16 value; uint8 type; uint8 number; }
    fmt = "<IhBB"
    size = struct.calcsize(fmt)
    while True:
        try:
            raw = os.read(fd, size)
        except BlockingIOError:
            break
        except OSError:
            break
        if len(raw) < size:
            break
        _t, value, etype, number = struct.unpack(fmt, raw)
        etype &= ~JS_EVENT_INIT
        if etype == JS_EVENT_AXIS and 0 <= number < len(axes):
            axes[number] = float(value) / 32767.0
        elif etype == JS_EVENT_BUTTON and 0 <= number < len(buttons):
            buttons[number] = 1 if value else 0


def cdr_joy(axes: List[float], buttons: List[int], frame_id: str = "") -> bytes:
    """Minimal CDR for sensor_msgs/Joy (for foxglove/cobridge)."""
    # alignment-sensitive; keep simple encapsulation
    # Prefer JSON UDP path for deploy_real; cobridge is best-effort.
    now = time.time()
    sec = int(now)
    nsec = int((now - sec) * 1e9)
    parts = []
    # header stamp
    parts.append(struct.pack("<iI", sec, nsec))
    fb = frame_id.encode("utf-8")
    parts.append(struct.pack("<I", len(fb) + 1))
    parts.append(fb + b"\x00")
    pad = (4 - (len(fb) + 1) % 4) % 4
    parts.append(b"\x00" * pad)
    parts.append(struct.pack("<I", len(axes)))
    for a in axes:
        parts.append(struct.pack("<f", float(a)))
    parts.append(struct.pack("<I", len(buttons)))
    for b in buttons:
        parts.append(struct.pack("<i", int(b)))
    return b"".join(parts)


class UdpJoySink:
    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[udp] → {host}:{port}", flush=True)

    def publish(self, axes: List[float], buttons: List[int]) -> None:
        payload = json.dumps(
            {"axes": axes, "buttons": buttons, "frame_id": "pc_gamepad"},
            separators=(",", ":"),
        ).encode("utf-8")
        self._sock.sendto(payload, self._addr)

    def close(self) -> None:
        self._sock.close()


class CobridgeJoySink:
    def __init__(self, host: str, port: int, topic: str) -> None:
        # Lightweight raw TCP publish of CDR bytes (foxglove-ws/cobridge compatible setups vary).
        # Prefer --via udp for deploy_real.
        self._host = host
        self._port = int(port)
        self._topic = topic
        self._sock = socket.create_connection((host, self._port), timeout=5.0)
        print(f"[cobridge] → {host}:{port} topic={topic}", flush=True)

    def publish(self, axes: List[float], buttons: List[int]) -> None:
        payload = cdr_joy(axes, buttons, frame_id="pc_gamepad")
        # length-prefixed blob; receivers may ignore if protocol differs
        self._sock.sendall(struct.pack(">I", len(payload)) + payload)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def _print_no_js_help() -> None:
    print(
        "未找到 /dev/input/js*。请先连接手柄，或加 --fake 测通路。\n"
        "若只有 event* 没有 js*：可能被 hid_playstation / ds4drv 占用，\n"
        "因此不会出现 js*/pygame 手柄。请先修复驱动绑定：\n"
        "  sudo modprobe -r hid_playstation;  # 再拔插手柄\n",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="PC gamepad → X2 /joy")
    ap.add_argument("--via", choices=("udp", "cobridge"), default="udp")
    ap.add_argument("--host", default="192.168.31.35")
    ap.add_argument(
        "--port",
        type=int,
        default=0,
        help="udp 默认 31337；cobridge 默认 21274",
    )
    ap.add_argument("--topic", default="/joy", help="仅 cobridge 用")
    ap.add_argument("--js", default="", help="e.g. /dev/input/js0")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--min-axes", type=int, default=8)
    ap.add_argument("--min-buttons", type=int, default=16)
    ap.add_argument("--fake", action="store_true", help="无手柄时发零输入")
    ap.add_argument("--seconds", type=float, default=0.0)
    args = ap.parse_args()

    if args.port <= 0:
        args.port = 21274 if args.via == "cobridge" else 31337

    fd: Optional[int] = None
    if args.fake:
        n_axes = max(8, args.min_axes)
        n_btns = max(16, args.min_buttons)
        axes = [0.0] * n_axes
        buttons = [0] * n_btns
        print("[js] --fake", flush=True)
    else:
        js_path = args.js or find_default_js()
        if not js_path:
            _print_no_js_help()
            return 1
        fd, name, n_axes, n_btns = open_joystick(js_path)
        n_axes = max(n_axes, args.min_axes)
        n_btns = max(n_btns, args.min_buttons)
        axes = [0.0] * n_axes
        buttons = [0] * n_btns
        print(f"[js] {js_path} name={name!r} axes={n_axes} buttons={n_btns}", flush=True)

    sink: JoySink
    try:
        if args.via == "udp":
            sink = UdpJoySink(args.host, args.port)
        else:
            sink = CobridgeJoySink(args.host, args.port, args.topic)
    except Exception as e:
        if fd is not None:
            os.close(fd)
        print(f"[sink] 连接失败: {e}", file=sys.stderr)
        return 1

    period = 1.0 / max(1.0, float(args.hz))
    last_print = 0.0
    last_btn_sig = ""
    t_end = time.time() + args.seconds if args.seconds > 0 else None
    try:
        while True:
            if t_end is not None and time.time() >= t_end:
                break
            t0 = time.time()
            if fd is not None:
                read_js_events(fd, axes, buttons)
            sink.publish(axes, buttons)
            now = time.time()
            btn_sig = "".join("1" if b else "0" for b in buttons[:12])
            if btn_sig != last_btn_sig or now - last_print > 1.0:
                last_print = now
                last_btn_sig = btn_sig
                ax_s = ",".join(f"{a:+.2f}" for a in axes[:4])
                print(f"[joy] axes[{ax_s}] buttons[{btn_sig}]", flush=True)
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        sink.close()
        if fd is not None:
            os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
