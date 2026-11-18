/** Twist-command gamepad (port of src/viewer/gamepad.html for local WASM play). */

export type PadEvent = "stop" | "reset" | "pause";
export type PadAxes = [forward: number, lateral: number, yaw: number];

type PadEls = {
  source: HTMLSelectElement;
  enable: HTMLButtonElement;
  stop: HTMLButtonElement;
  left: HTMLElement;
  right: HTMLElement;
  status: HTMLElement;
};

const DEADZONE = 0.12;
const KNOB_MARGIN = 16;
// W3C "standard" gamepad mapping (Xbox / PS layout).
const BTN_A = 0;
const BTN_X = 2;
const BTN_Y = 3;
const BTN_START = 9;

function dead(x: number): number {
  return Math.abs(x) < DEADZONE ? 0 : (Math.sign(x) * (Math.abs(x) - DEADZONE)) / (1 - DEADZONE);
}

/**
 * Virtual (pointer) or physical (Gamepad API) sticks → normalized twist axes.
 * Left stick: forward / lateral, right stick X: yaw. Releasing a stick returns 0.
 * Physical buttons: A enable/disable, X stop, Y reset, Start pause.
 */
export class GamepadInput {
  enabled = false;
  private axes: PadAxes = [0, 0, 0];
  private events: PadEvent[] = [];
  private padIndex: number | null = null;
  private prevPressed: boolean[] = [];
  private lastStatus = "";

  constructor(private readonly els: PadEls) {
    els.enable.addEventListener("click", () => {
      if (this.enabled) this.disable("已关闭手柄控制，速度归零");
      else this.enable();
    });
    els.stop.addEventListener("click", () => this.stop());
    els.source.addEventListener("change", () => {
      this.prevPressed = [];
      this.disable("已切换输入方式，请重新启用");
    });
    this.setupPad(els.left, "left");
    this.setupPad(els.right, "right");
    window.addEventListener("gamepadconnected", (e) => {
      if (this.physical || this.enabled) return;
      els.source.value = "physical";
      this.prevPressed = [];
      this.setStatus(`检测到手柄 ${e.gamepad.id}；按 A 或「启用手柄」开始`);
    });
    window.addEventListener("gamepaddisconnected", (e) => {
      if (e.gamepad.index !== this.padIndex) return;
      this.padIndex = null;
      this.disable("手柄已断开，速度归零");
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.disable("页面切到后台，速度归零");
    });
    this.setStatus("未启用");
  }

  get physical(): boolean {
    return this.els.source.value === "physical";
  }

  /** Call once per frame. ``axes`` is null while disabled. */
  poll(): { axes: PadAxes | null; events: PadEvent[] } {
    if (this.physical) this.pollPhysical();
    const events = this.events;
    this.events = [];
    return { axes: this.enabled ? ([...this.axes] as PadAxes) : null, events };
  }

  enable() {
    this.enabled = true;
    this.resetAxes();
    this.els.enable.textContent = "关闭手柄";
    this.els.enable.classList.add("on");
    this.setStatus(this.physical ? "手柄控制中" : "网页摇杆控制中；松开归零");
  }

  disable(msg = "未启用") {
    this.enabled = false;
    this.resetAxes();
    this.els.enable.textContent = "启用手柄";
    this.els.enable.classList.remove("on");
    this.setStatus(msg);
  }

  private stop() {
    this.events.push("stop");
    this.disable("已停止，速度归零");
  }

  private resetAxes() {
    this.axes = [0, 0, 0];
    this.moveKnob(this.els.left, 0, 0);
    this.moveKnob(this.els.right, 0, 0);
  }

  private setStatus(msg: string) {
    if (msg === this.lastStatus) return;
    this.lastStatus = msg;
    this.els.status.textContent = msg;
  }

  private moveKnob(pad: HTMLElement, x: number, y: number) {
    const knob = pad.querySelector<HTMLElement>(".knob");
    if (!knob) return;
    const r = pad.clientWidth / 2 - KNOB_MARGIN;
    knob.style.transform = x || y ? `translate(${x * r}px, ${y * r}px)` : "";
  }

  private setupPad(pad: HTMLElement, side: "left" | "right") {
    let pointer: number | null = null;
    const move = (e: PointerEvent) => {
      if (pointer !== e.pointerId) return;
      const rect = pad.getBoundingClientRect();
      const radius = rect.width / 2;
      let x = (e.clientX - rect.left - radius) / (radius - KNOB_MARGIN);
      let y = (e.clientY - rect.top - radius) / (radius - KNOB_MARGIN);
      const m = Math.max(1, Math.hypot(x, y));
      x /= m;
      y /= m;
      this.moveKnob(pad, x, y);
      if (side === "left") {
        this.axes[0] = -y;
        this.axes[1] = -x;
      } else {
        this.axes[2] = -x;
      }
    };
    const release = (e: PointerEvent) => {
      if (pointer !== e.pointerId) return;
      pointer = null;
      this.moveKnob(pad, 0, 0);
      if (side === "left") {
        this.axes[0] = 0;
        this.axes[1] = 0;
      } else {
        this.axes[2] = 0;
      }
    };
    pad.addEventListener("pointerdown", (e) => {
      if (!this.enabled || this.physical || pointer !== null) return;
      pointer = e.pointerId;
      pad.setPointerCapture(pointer);
      move(e);
    });
    pad.addEventListener("pointermove", move);
    pad.addEventListener("pointerup", release);
    pad.addEventListener("pointercancel", release);
    pad.addEventListener("lostpointercapture", release);
  }

  private pollPhysical() {
    if (!window.isSecureContext) {
      if (this.enabled) this.disable();
      this.setStatus("实体手柄需要 localhost / HTTPS 打开页面；可切换到网页摇杆");
      return;
    }
    if (!navigator.getGamepads) {
      if (this.enabled) this.disable();
      this.setStatus("此浏览器不支持 Gamepad API（请用 Chrome / Edge / Firefox 打开）");
      return;
    }
    const pads = Array.from(navigator.getGamepads());
    const pad =
      (this.padIndex !== null ? pads[this.padIndex] : null) ??
      pads.find((p) => p?.connected) ??
      null;
    if (!pad?.connected) {
      this.resetAxes();
      this.setStatus("等待手柄：插在运行浏览器的电脑上，页面聚焦时按一下任意按键");
      return;
    }
    // Non-standard pads: assume the common generic layout (may differ per device).
    const standard = pad.mapping === "standard";
    const label = standard ? pad.id : `${pad.id}（非标准映射，按键/右摇杆可能不对）`;
    this.padIndex = pad.index;

    const pressed = pad.buttons.map((b) => b.pressed);
    const edge = (i: number) => !!pressed[i] && !this.prevPressed[i];
    if (edge(BTN_A)) {
      if (this.enabled) this.disable("已关闭手柄控制，速度归零");
      else this.enable();
    }
    if (edge(BTN_X)) this.stop();
    if (edge(BTN_Y)) this.events.push("reset");
    if (edge(BTN_START)) this.events.push("pause");
    this.prevPressed = pressed;

    if (!this.enabled) {
      this.setStatus(`已连接 ${label}；按 A 或「启用手柄」开始`);
      return;
    }
    const lx = dead(pad.axes[0] ?? 0);
    const ly = dead(pad.axes[1] ?? 0);
    const rx = dead(pad.axes[2] ?? 0);
    this.axes = [-ly, -lx, -rx];
    this.moveKnob(this.els.left, lx, ly);
    this.moveKnob(this.els.right, rx, 0);
    this.setStatus(`手柄控制中：${label}`);
  }
}
