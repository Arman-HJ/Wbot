import type { MjData, MjModel } from "mujoco-js";
import { asArray } from "./mujoco_util";
import type { PolicyController } from "./policy";

interface JointRow {
  root: HTMLElement;
  qFill: HTMLElement;
  qMark: HTMLElement;
  qVal: HTMLElement;
  tgtFill: HTMLElement;
  tgtMark: HTMLElement;
  tgtVal: HTMLElement;
  tauFill: HTMLElement;
  tauVal: HTMLElement;
  lo: number;
  hi: number;
  tauLimit: number;
  defaultQ: number;
}

/** Per-joint: target / angle / torque (bars + absolute numbers). */
export class JointMeter {
  private el: HTMLElement | null = null;
  private sumEl: HTMLElement | null = null;
  private rows: JointRow[] = [];
  private layoutKey = "";
  private qEma = new Float64Array(0);
  private tgtEma = new Float64Array(0);
  private tauEma = new Float64Array(0);
  private lastMs = 0;
  private lastTextMs = 0;
  private readonly barIntervalMs = 16;
  private readonly textIntervalMs = 50;
  private readonly emaTauS = 0.04;

  bindUi(table: HTMLElement, sumEl: HTMLElement) {
    this.el = table;
    this.sumEl = sumEl;
  }

  /** Skip DOM work when the joints tab is hidden. */
  enabled = true;

  update(policy: PolicyController, model: MjModel, data: MjData, nowMs: number) {
    if (!this.enabled || !this.el || !this.sumEl) return;
    const dtMs = nowMs - this.lastMs;
    if (dtMs < this.barIntervalMs) return;
    this.lastMs = nowMs;

    const names = policy.jointNames;
    const n = names.length;
    const qadr = policy.getJointQposAdr();
    const aids = policy.getActuatorIds();
    const key = `${names.join(",")}|${model.nq}|${model.nu}`;
    let reset = false;
    if (key !== this.layoutKey) {
      this.build(policy, model);
      this.layoutKey = key;
      reset = true;
    }

    const qpos = asArray(data.qpos, model.nq);
    const ctrl = asArray(data.ctrl, model.nu);
    const force = asArray(data.actuator_force, model.nu);
    const alpha = reset ? 1 : 1 - Math.exp(-Math.min(dtMs, 500) / 1000 / this.emaTauS);

    let absSum = 0;
    let absErr = 0;
    for (let i = 0; i < n; i++) {
      const aid = aids[i];
      const q = qpos[qadr[i]];
      const tgt = ctrl[aid];
      const tau = force[aid];
      absSum += Math.abs(tau);
      absErr += Math.abs(tgt - q);
      this.qEma[i] += alpha * (q - this.qEma[i]);
      this.tgtEma[i] += alpha * (tgt - this.tgtEma[i]);
      this.tauEma[i] += alpha * (tau - this.tauEma[i]);

      const r = this.rows[i];
      const zp = pct(clamp(0, r.lo, r.hi), r.lo, r.hi);

      const tp = pct(this.tgtEma[i], r.lo, r.hi);
      r.tgtFill.style.left = `${Math.min(tp, zp)}%`;
      r.tgtFill.style.width = `${Math.abs(tp - zp)}%`;
      r.tgtMark.style.left = `${tp}%`;

      const qp = pct(this.qEma[i], r.lo, r.hi);
      r.qFill.style.left = `${Math.min(qp, zp)}%`;
      r.qFill.style.width = `${Math.abs(qp - zp)}%`;
      r.qMark.style.left = `${qp}%`;

      const ratio = clamp(this.tauEma[i] / r.tauLimit, -1, 1);
      const w = Math.abs(ratio) * 50;
      r.tauFill.style.left = `${ratio < 0 ? 50 - w : 50}%`;
      r.tauFill.style.width = `${w}%`;
      r.tauFill.style.background = loadColor(Math.abs(ratio));
    }

    if (nowMs - this.lastTextMs < this.textIntervalMs && !reset) return;
    this.lastTextMs = nowMs;
    for (let i = 0; i < n; i++) {
      const r = this.rows[i];
      const q = this.qEma[i];
      const tgt = this.tgtEma[i];
      const tau = this.tauEma[i];
      r.tgtVal.textContent = fmtRad(tgt);
      r.qVal.textContent = fmtRad(q);
      r.tauVal.textContent = Math.abs(tau) >= 10 ? tau.toFixed(0) : tau.toFixed(1);
      r.root.title =
        `${names[i]}\n` +
        `tgt=${tgt.toFixed(3)}  q=${q.toFixed(3)}  err=${(q - tgt).toFixed(3)}\n` +
        `range=[${r.lo.toFixed(2)}, ${r.hi.toFixed(2)}] rad\n` +
        `τ=${tau.toFixed(1)} / ${r.tauLimit.toFixed(0)} N·m (${((Math.abs(tau) / r.tauLimit) * 100).toFixed(0)}%)`;
    }
    this.sumEl.textContent =
      `Σ|τ| ≈ ${absSum.toFixed(1)} N·m · Σ|tgt−q| ≈ ${absErr.toFixed(2)} rad`;
  }

  private build(policy: PolicyController, model: MjModel) {
    const names = policy.jointNames;
    const n = names.length;
    const qadr = policy.getJointQposAdr();
    const aids = policy.getActuatorIds();
    const jntQposAdr = asArray(model.jnt_qposadr, model.njnt);
    const jntRange = asArray(model.jnt_range, model.njnt * 2);
    const jntLimited = asArray(model.jnt_limited, model.njnt);
    const forceRange = asArray(model.actuator_forcerange, model.nu * 2);
    const forceLimited = asArray(model.actuator_forcelimited, model.nu);
    const ctrlRange = asArray(model.actuator_ctrlrange, model.nu * 2);

    this.rows = [];
    this.qEma = new Float64Array(n);
    this.tgtEma = new Float64Array(n);
    this.tauEma = new Float64Array(n);

    const frag = document.createDocumentFragment();
    frag.appendChild(
      div(
        "joint-block joint-head",
        div("joint-block-title", "joint"),
        div(
          "joint-metric-list",
          metricHead("tgt"),
          metricHead("q"),
          metricHead("τ"),
        ),
      ),
    );

    for (let i = 0; i < n; i++) {
      let jid = -1;
      for (let j = 0; j < model.njnt; j++) {
        if (jntQposAdr[j] === qadr[i]) {
          jid = j;
          break;
        }
      }
      let lo = -Math.PI;
      let hi = Math.PI;
      if (jid >= 0 && jntLimited[jid] && jntRange[jid * 2 + 1] > jntRange[jid * 2]) {
        lo = jntRange[jid * 2];
        hi = jntRange[jid * 2 + 1];
      } else {
        const aid = aids[i];
        if (ctrlRange[aid * 2 + 1] > ctrlRange[aid * 2]) {
          lo = ctrlRange[aid * 2];
          hi = ctrlRange[aid * 2 + 1];
        }
      }
      const aid = aids[i];
      const fmax = Math.max(Math.abs(forceRange[aid * 2]), Math.abs(forceRange[aid * 2 + 1]));
      const tauLimit = forceLimited[aid] && fmax > 0 ? fmax : 100;
      const defaultQ = policy.defaultPos[i];
      const defPct = `${pct(defaultQ, lo, hi)}%`;

      const tgtFill = div("jbar-fill jbar-tgt");
      const tgtMark = div("jbar-mark jbar-mark-tgt");
      const tgtVal = div("jbar-val");
      const tgtDef = div("jbar-mark jbar-mark-def");
      tgtDef.style.left = defPct;

      const qFill = div("jbar-fill jbar-q");
      const qMark = div("jbar-mark jbar-mark-q");
      const qVal = div("jbar-val");
      const qDef = div("jbar-mark jbar-mark-def");
      qDef.style.left = defPct;

      const tauFill = div("jbar-fill");
      const tauVal = div("jbar-val");

      const root = div(
        "joint-block",
        div("joint-block-title", shortName(names[i])),
        div(
          "joint-metric-list",
          metricRow(
            "tgt",
            div("jbar-track", tgtFill, tgtDef, tgtMark, tgtVal),
          ),
          metricRow("q", div("jbar-track", qFill, qDef, qMark, qVal)),
          metricRow("τ", div("jbar-track", div("jbar-center"), tauFill, tauVal)),
        ),
      );
      frag.appendChild(root);

      this.rows.push({
        root,
        qFill,
        qMark,
        qVal,
        tgtFill,
        tgtMark,
        tgtVal,
        tauFill,
        tauVal,
        lo,
        hi,
        tauLimit,
        defaultQ,
      });
    }
    this.el!.replaceChildren(frag);
  }
}

function metricHead(label: string): HTMLElement {
  return div("joint-metric", div("joint-metric-lab", label), div("joint-head-cell", ""));
}

function metricRow(label: string, track: HTMLElement): HTMLElement {
  return div("joint-metric", div("joint-metric-lab", label), track);
}

function div(cls: string, ...children: (HTMLElement | string)[]): HTMLElement {
  const el = document.createElement("div");
  el.className = cls;
  for (const c of children) el.append(c);
  return el;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function pct(v: number, lo: number, hi: number): number {
  return ((clamp(v, lo, hi) - lo) / (hi - lo)) * 100;
}

function loadColor(r: number): string {
  const hue = r <= 0.5 ? 130 : Math.max(0, 130 - ((r - 0.5) / 0.5) * 130);
  return `hsl(${hue.toFixed(0)}, 70%, 48%)`;
}

function shortName(name: string): string {
  return name.replace(/_joint$/, "").replace(/^(left|right)_/, (_, s) => (s === "left" ? "L_" : "R_"));
}

function fmtRad(v: number): string {
  return (v >= 0 ? " " : "") + v.toFixed(2);
}
