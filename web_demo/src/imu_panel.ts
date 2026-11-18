import type { MjData, MjModel } from "mujoco-js";
import { asArray, name2id, type MujocoModule } from "./mujoco_util";

type AxisRow = {
  fill: HTMLElement;
  val: HTMLElement;
};

type VecGroup = {
  root: HTMLElement;
  title: HTMLElement;
  rows: [AxisRow, AxisRow, AxisRow];
  limit: number;
};

/** Live IMU readout: gyro / accel / projected gravity (+ tiny attitude ball). */
export class ImuPanel {
  private el: HTMLElement | null = null;
  private attitude: HTMLCanvasElement | null = null;
  private tipEl: HTMLElement | null = null;
  private pitchFill: HTMLElement | null = null;
  private pitchVal: HTMLElement | null = null;
  private pitchDeg = 0;
  private gyro: VecGroup | null = null;
  private accel: VecGroup | null = null;
  private linVel: VecGroup | null = null;
  private gravity: VecGroup | null = null;
  private gyroAdr = -1;
  private accelAdr = -1;
  private linVelAdr = -1;
  private freeQ = 0;
  private freeV = 0;
  private rootBody = 1;
  private lastMs = 0;
  private readonly intervalMs = 33;
  private readonly R = new Float64Array(9);
  private readonly g = new Float64Array(3);
  private readonly w = new Float64Array(3);
  private readonly a = new Float64Array(3);
  private readonly v = new Float64Array(3);
  enabled = true;

  bindUi(root: HTMLElement) {
    this.el = root;
    root.replaceChildren();
    root.classList.add("imu-panel");

    const head = document.createElement("div");
    head.className = "imu-head";
    this.attitude = document.createElement("canvas");
    this.attitude.width = 88;
    this.attitude.height = 88;
    this.attitude.className = "imu-attitude";
    this.tipEl = document.createElement("div");
    this.tipEl.className = "hint imu-tip";
    this.tipEl.textContent = "等待仿真…";
    head.append(this.attitude, this.tipEl);
    root.append(head);

    const pitch = document.createElement("div");
    pitch.className = "imu-group";
    const pitchTitle = document.createElement("div");
    pitchTitle.className = "imu-group-title";
    pitchTitle.textContent = "俯仰角 pitch（+抬头 / −低头）";
    const pitchRow = document.createElement("div");
    pitchRow.className = "imu-row imu-pitch-row";
    const pitchLab = document.createElement("span");
    pitchLab.className = "imu-axis";
    pitchLab.textContent = "θ";
    const pitchTrack = document.createElement("div");
    pitchTrack.className = "imu-track";
    const pitchZero = document.createElement("div");
    pitchZero.className = "imu-zero";
    this.pitchFill = document.createElement("div");
    this.pitchFill.className = "imu-fill";
    pitchTrack.append(pitchZero, this.pitchFill);
    this.pitchVal = document.createElement("span");
    this.pitchVal.className = "imu-val imu-pitch-val";
    this.pitchVal.textContent = "0.0°";
    pitchRow.append(pitchLab, pitchTrack, this.pitchVal);
    pitch.append(pitchTitle, pitchRow);
    root.append(pitch);

    this.gyro = this.makeGroup(root, "角速度 ω (rad/s)", 4);
    this.accel = this.makeGroup(root, "线加速度 a (m/s²)", 20);
    this.linVel = this.makeGroup(root, "线速度 v (m/s)", 3);
    this.gravity = this.makeGroup(root, "投影重力 g_b", 1.05);
  }

  bindModel(mj: MujocoModule, model: MjModel, opts?: { rootBody?: string; freeJoint?: string }) {
    const rootName = opts?.rootBody ?? "pelvis";
    const freeName = opts?.freeJoint ?? "floating_base_joint";
    this.rootBody = Math.max(0, name2id(mj, model, "body", rootName));
    const free = name2id(mj, model, "joint", freeName);
    this.freeQ = free >= 0 ? model.jnt_qposadr[free] : 0;
    this.freeV = free >= 0 ? model.jnt_dofadr[free] : 0;
    this.gyroAdr = sensorAdr(mj, model, ["imu_ang_vel", "imu_gyro", "gyro"]);
    this.accelAdr = sensorAdr(mj, model, ["imu_lin_acc", "imu_acc", "accelerometer"]);
    this.linVelAdr = sensorAdr(mj, model, ["imu_lin_vel", "velocimeter"]);
    if (this.tipEl) {
      const bits = [
        this.gyroAdr >= 0 ? "gyro✓" : "gyro←qvel",
        this.accelAdr >= 0 ? "accel✓" : "accel—",
        this.linVelAdr >= 0 ? "v✓" : "v—",
      ];
      this.tipEl.textContent = bits.join(" · ");
    }
    if (this.accel) this.accel.root.hidden = this.accelAdr < 0;
    if (this.linVel) this.linVel.root.hidden = this.linVelAdr < 0;
  }

  update(model: MjModel, data: MjData, nowMs: number) {
    if (!this.enabled || !this.el) return;
    if (nowMs - this.lastMs < this.intervalMs) return;
    this.lastMs = nowMs;

    const qvel = asArray(data.qvel, model.nv);
    const sensordata = model.nsensordata > 0 ? asArray(data.sensordata, model.nsensordata) : null;

    // Prefer root-body quaternion (matches mjlab projected_gravity).
    const xquat = asArray(data.xquat, model.nbody * 4);
    const qOff = this.rootBody * 4;
    quatToMat(xquat.subarray(qOff, qOff + 4), this.R);
    this.g[0] = -this.R[6];
    this.g[1] = -this.R[7];
    this.g[2] = -this.R[8];

    if (this.gyroAdr >= 0 && sensordata) {
      this.w[0] = sensordata[this.gyroAdr];
      this.w[1] = sensordata[this.gyroAdr + 1];
      this.w[2] = sensordata[this.gyroAdr + 2];
    } else {
      this.w[0] = qvel[this.freeV + 3];
      this.w[1] = qvel[this.freeV + 4];
      this.w[2] = qvel[this.freeV + 5];
    }

    if (this.accelAdr >= 0 && sensordata) {
      this.a[0] = sensordata[this.accelAdr];
      this.a[1] = sensordata[this.accelAdr + 1];
      this.a[2] = sensordata[this.accelAdr + 2];
    }
    if (this.linVelAdr >= 0 && sensordata) {
      this.v[0] = sensordata[this.linVelAdr];
      this.v[1] = sensordata[this.linVelAdr + 1];
      this.v[2] = sensordata[this.linVelAdr + 2];
    }

    if (this.gyro) paintGroup(this.gyro, this.w);
    if (this.accel && this.accelAdr >= 0) paintGroup(this.accel, this.a);
    if (this.linVel && this.linVelAdr >= 0) paintGroup(this.linVel, this.v);
    if (this.gravity) paintGroup(this.gravity, this.g);

    // g_b = Rᵀ(0,0,-1) ≈ (0,0,-1) upright. pitch>0 = 抬头, roll>0 = 左倾.
    const pitchRad = Math.atan2(this.g[0], Math.hypot(this.g[1], this.g[2]));
    const rollRad = Math.atan2(-this.g[1], -this.g[2]);
    this.pitchDeg = (pitchRad * 180) / Math.PI;
    const rollDeg = (rollRad * 180) / Math.PI;
    const tilt = (Math.acos(clamp(-this.g[2], -1, 1)) * 180) / Math.PI;
    const wMag = Math.hypot(this.w[0], this.w[1], this.w[2]);
    if (this.pitchFill && this.pitchVal) {
      const ratio = clamp(this.pitchDeg / 90, -1, 1);
      const width = Math.abs(ratio) * 50;
      this.pitchFill.style.left = `${ratio < 0 ? 50 - width : 50}%`;
      this.pitchFill.style.width = `${width}%`;
      this.pitchFill.style.background = hexAlpha("#58a6ff", 0.35 + 0.65 * Math.abs(ratio));
      this.pitchVal.textContent = `${this.pitchDeg >= 0 ? "+" : ""}${this.pitchDeg.toFixed(1)}°`;
    }
    if (this.tipEl) {
      const src = this.gyroAdr >= 0 ? "gyro" : "qvel";
      this.tipEl.textContent =
        `pitch ${this.pitchDeg >= 0 ? "+" : ""}${this.pitchDeg.toFixed(1)}°` +
        ` · roll ${rollDeg >= 0 ? "+" : ""}${rollDeg.toFixed(1)}°` +
        ` · 倾角 ${tilt.toFixed(1)}°` +
        ` · |ω| ${wMag.toFixed(2)} · ${src}`;
    }
    this.drawAttitude();
  }

  private makeGroup(parent: HTMLElement, title: string, limit: number): VecGroup {
    const root = document.createElement("div");
    root.className = "imu-group";
    const titleEl = document.createElement("div");
    titleEl.className = "imu-group-title";
    titleEl.textContent = title;
    root.append(titleEl);
    const rows: AxisRow[] = [];
    for (let i = 0; i < 3; i++) {
      const row = document.createElement("div");
      row.className = "imu-row";
      const lab = document.createElement("span");
      lab.className = "imu-axis";
      lab.textContent = "xyz"[i];
      const track = document.createElement("div");
      track.className = "imu-track";
      const zero = document.createElement("div");
      zero.className = "imu-zero";
      const fill = document.createElement("div");
      fill.className = "imu-fill";
      track.append(zero, fill);
      const val = document.createElement("span");
      val.className = "imu-val";
      val.textContent = "0.000";
      row.append(lab, track, val);
      root.append(row);
      rows.push({ fill, val });
    }
    parent.append(root);
    return { root, title: titleEl, rows: rows as [AxisRow, AxisRow, AxisRow], limit };
  }

  private drawAttitude() {
    const c = this.attitude;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const w = c.width;
    const h = c.height;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) * 0.42;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0f1728";
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - r, cy);
    ctx.lineTo(cx + r, cy);
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx, cy + r);
    ctx.stroke();

    // Pitch horizon: shift with pitch (up = nose up).
    const pitchPx = (-this.pitchDeg / 90) * r * 0.85;
    ctx.strokeStyle = "#58a6ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - r * 0.7, cy + pitchPx);
    ctx.lineTo(cx + r * 0.7, cy + pitchPx);
    ctx.stroke();
    ctx.fillStyle = "#58a6ff";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("pitch", 4, h - 6);

    // Projected gravity in body frame → screen (x right, y up).
    const gx = this.g[0];
    const gy = this.g[1];
    const gz = this.g[2];
    const px = cx + gx * r * 0.85;
    const py = cy - gy * r * 0.85;
    ctx.fillStyle = gz >= 0 ? "#5ad4a0" : "#ff8e8e";
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#9db0d0";
    ctx.fillText("g", 4, 12);
  }
}

function sensorAdr(mj: MujocoModule, model: MjModel, names: string[]): number {
  for (const name of names) {
    const sid = name2id(mj, model, "sensor", name);
    if (sid >= 0) return model.sensor_adr[sid];
  }
  return -1;
}

function paintGroup(group: VecGroup, vec: ArrayLike<number>) {
  for (let i = 0; i < 3; i++) {
    const v = Number(vec[i]);
    const ratio = clamp(v / group.limit, -1, 1);
    const width = Math.abs(ratio) * 50;
    const fill = group.rows[i].fill;
    fill.style.left = `${ratio < 0 ? 50 - width : 50}%`;
    fill.style.width = `${width}%`;
    fill.style.background = axisColor(i, Math.abs(ratio));
    group.rows[i].val.textContent = fmt(v);
  }
}

function axisColor(axis: number, intensity: number): string {
  const colors = ["#ff7b72", "#3fb950", "#58a6ff"];
  const a = 0.35 + 0.65 * intensity;
  return hexAlpha(colors[axis], a);
}

function hexAlpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r},${g},${b},${a.toFixed(2)})`;
}

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(3);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** wxyz → row-major 3×3. */
function quatToMat(q: ArrayLike<number>, out: Float64Array) {
  const w = Number(q[0]);
  const x = Number(q[1]);
  const y = Number(q[2]);
  const z = Number(q[3]);
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  out[0] = 1 - 2 * (yy + zz);
  out[1] = 2 * (xy - wz);
  out[2] = 2 * (xz + wy);
  out[3] = 2 * (xy + wz);
  out[4] = 1 - 2 * (xx + zz);
  out[5] = 2 * (yz - wx);
  out[6] = 2 * (xz - wy);
  out[7] = 2 * (yz + wx);
  out[8] = 1 - 2 * (xx + yy);
}
