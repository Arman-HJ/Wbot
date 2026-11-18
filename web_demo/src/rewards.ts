import type { MjData, MjModel } from "mujoco-js";
import { quatErrorMagnitude } from "./math";
import { asArray, name2id, type MujocoModule } from "./mujoco_util";
import type { PolicyController } from "./policy";
import { AmpRewardEval } from "./amp_rewards";
import { AmpStyleEval } from "./amp_style";
import type {
  AmpDiscSpec,
  AmpRewardConfig,
  RewardConfig,
  RewardTermSpec,
  VelocityRewardConfig,
} from "./types";
import { VelocityRewardEval } from "./velocity_rewards";

type BodyIds = { name: string; id: number }[];

type RowEls = { root: HTMLElement; name: HTMLElement; fill: HTMLElement; val: HTMLElement };

/** ``info`` rows are shown but not summed into Σ. */
type RowValue = { name: string; value: number; weight: number; note?: string; info?: boolean };

const STYLE_TERM = "style_reward";
const styleLabelTerm = (label: string) => `style·${label}`;

/**
 * Live reward terms (Viser Rewards bar parity) for WASM play.
 * kick: SE(2)-aligned ghost FK vs robot. velocity: ``VelocityRewardEval``.
 * amp_twist: ``AmpRewardEval`` + discriminator style (``AmpStyleEval``).
 * Displays weight * raw (no dt scale).
 */
export class RewardMeter {
  /** Render only while the rewards tab is shown (terms still accumulate). */
  visible = false;
  private cfg: RewardConfig | null = null;
  private vel: VelocityRewardEval | AmpRewardEval | null = null;
  private style: AmpStyleEval | null = null;
  private packUrl = "";
  private policyDt = 0.02;
  private lastBarMs = 0;
  private lastTextMs = 0;
  private dirty = false;
  private barScale = 1e-8;
  private readonly barIntervalMs = 16;
  private readonly textIntervalMs = 50;
  private rows: RowEls[] = [];
  private rowKey = "";
  private lastValues: RowValue[] = [];
  private lastTotal = 0;
  private ghostData: MjData | null = null;
  private mj: MujocoModule | null = null;
  private model: MjModel | null = null;
  private anchorId = -1;
  private bodyCache = new Map<string, number>();
  private prevAction = new Float64Array(0);
  private hasPrevAction = false;
  private histories = new Map<string, number[]>();
  private windowSteps = 50;
  private el: HTMLElement | null = null;
  private totalEl: HTMLElement | null = null;
  private enabled = false;

  bindUi(panel: HTMLElement, totalEl: HTMLElement) {
    this.el = panel;
    this.totalEl = totalEl;
  }

  private setEnabled(on: boolean) {
    this.enabled = on && (!!this.cfg || !!this.vel);
    if (!this.enabled) this.renderUnavailable();
  }

  async load(
    mj: MujocoModule,
    model: MjModel,
    packUrl: string,
    rewardsFile: string | undefined,
    policyDt: number,
    policy: PolicyController,
  ) {
    this.disposeGhost();
    this.cfg = null;
    this.vel = null;
    this.style = null;
    this.packUrl = packUrl;
    this.histories.clear();
    this.hasPrevAction = false;
    this.mj = mj;
    this.model = model;
    this.policyDt = policyDt;
    this.rowKey = "";
    this.windowSteps = Math.max(1, Math.round(1.0 / Math.max(1e-3, policyDt)));

    if (!rewardsFile) {
      this.setEnabled(false);
      return;
    }
    const res = await fetch(`${packUrl}/${rewardsFile}?t=${Date.now()}`);
    if (!res.ok) {
      this.setEnabled(false);
      return;
    }
    const json = await res.json();
    if (json?.profile === "velocity" || json?.profile === "amp") {
      this.vel =
        json.profile === "amp"
          ? new AmpRewardEval(mj, model, json as AmpRewardConfig, policy)
          : new VelocityRewardEval(mj, model, json as VelocityRewardConfig, policy);
      for (const t of this.vel.terms) this.histories.set(t.name, []);
      if (json.profile === "amp") await this.loadStyle(mj, model, policy);
      this.setEnabled(true);
      this.renderEmpty();
      return;
    }
    this.cfg = json as RewardConfig;
    this.ghostData = new mj.MjData(model);
    this.anchorId = name2id(mj, model, "body", this.cfg.anchor_body || "torso_link");
    this.bodyCache.clear();
    for (const t of this.cfg.terms) {
      this.histories.set(t.name, []);
      for (const n of t.body_names || []) {
        if (!this.bodyCache.has(n)) {
          this.bodyCache.set(n, name2id(mj, model, "body", n));
        }
      }
    }
    if (this.anchorId >= 0) {
      this.bodyCache.set(this.cfg.anchor_body || "torso_link", this.anchorId);
    }
    this.setEnabled(true);
    this.renderEmpty();
  }

  private async loadStyle(mj: MujocoModule, model: MjModel, policy: PolicyController) {
    const specFile = policy.manifest.files.amp_disc;
    if (!specFile) return;
    try {
      const res = await fetch(`${this.packUrl}/${specFile}?t=${Date.now()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const spec = (await res.json()) as AmpDiscSpec;
      const root = policy.manifest.root_body || "pelvis";
      this.style = new AmpStyleEval(mj, model, spec, root, policy.manifest.policy_dt);
      this.histories.set(STYLE_TERM, []);
      for (const l of spec.label_names) this.histories.set(styleLabelTerm(l), []);
      await this.syncCheckpoint(policy);
    } catch (e) {
      console.warn("[rewards] AMP style disabled:", e);
      this.style = null;
    }
  }

  /** Load discriminator weights matching the policy's current checkpoint. */
  async syncCheckpoint(policy: PolicyController) {
    const style = this.style;
    if (!style) return;
    const name = policy.getCurrentCheckpoint();
    const entry = (policy.manifest.checkpoints || []).find((e) => e.name === name || e.file === name);
    if (!entry?.disc) {
      console.warn(`[rewards] no discriminator for checkpoint ${name}`);
      return;
    }
    await style.loadWeights(`${this.packUrl}/${entry.disc}`, entry.disc);
    style.reset();
    for (const n of [STYLE_TERM, ...style.spec.label_names.map(styleLabelTerm)]) {
      const h = this.histories.get(n);
      if (h) h.length = 0;
    }
  }

  reset() {
    this.hasPrevAction = false;
    this.prevAction.fill(0);
    this.vel?.reset();
    this.style?.reset();
    for (const h of this.histories.values()) h.length = 0;
    if (this.enabled) this.renderEmpty();
  }

  /** Call once per policy step (after ``policy.step``). */
  update(policy: PolicyController, robotData: MjData) {
    if (!this.enabled) return;
    if (this.vel) {
      this.updateVelocity(policy, robotData);
      return;
    }
    if (!this.cfg || !this.mj || !this.model || !this.ghostData) {
      return;
    }
    if (!policy.hasMotion()) return;
    if (!policy.writeGhostQpos(this.model, robotData, this.ghostData)) return;
    policy.writeGhostVel(this.model, this.ghostData);
    this.mj.mj_forward(this.model, this.ghostData);

    const action = policy.getLastAction();
    const values: { name: string; value: number; weight: number }[] = [];
    let total = 0;
    for (const term of this.cfg.terms) {
      const raw = this.evalTerm(term, robotData, action);
      const weighted = raw * term.weight;
      const v = Number.isFinite(weighted) ? weighted : 0;
      total += v;
      const hist = this.histories.get(term.name)!;
      hist.push(v);
      while (hist.length > this.windowSteps) hist.shift();
      values.push({ name: term.name, value: v, weight: term.weight });
    }
    if (!this.hasPrevAction || this.prevAction.length !== action.length) {
      this.prevAction = new Float64Array(action.length);
      this.hasPrevAction = true;
    }
    this.prevAction.set(action);
    this.render(values, total);
  }

  private updateVelocity(policy: PolicyController, robotData: MjData) {
    const vel = this.vel!;
    const raw = vel.evaluate(policy, robotData);
    const values: RowValue[] = [];
    let total = 0;
    const push = (name: string, rawValue: number, weight: number, note?: string, info = false) => {
      const v = rawValue * weight;
      const hist = this.histories.get(name)!;
      if (Number.isFinite(v)) {
        if (!info) total += v;
        hist.push(v);
        while (hist.length > this.windowSteps) hist.shift();
      }
      values.push({ name, value: Number.isFinite(v) ? v : NaN, weight, note, info });
    };
    const style = this.style;
    if (style) {
      const r = style.evaluate(policy, robotData);
      const note = style.label ? `${style.label} D=${style.prediction.toFixed(2)}` : undefined;
      push(STYLE_TERM, r, style.weight, note);
      style.spec.label_names.forEach((l, k) => {
        const d = style.perLabelD[k];
        const active = l === style.label ? " ◀" : "";
        const lnote = Number.isFinite(d) ? `D=${d.toFixed(2)}${active}` : undefined;
        push(styleLabelTerm(l), style.perLabel[k], style.weight, lnote, true);
      });
    }
    vel.terms.forEach((term, k) => push(term.name, raw[k], term.weight));
    this.render(values, total);
  }

  /** Repaint on tab switch without waiting for the next policy step. */
  refresh() {
    if (!this.enabled) {
      this.renderUnavailable();
      return;
    }
    this.lastBarMs = 0;
    this.tick(performance.now());
  }

  private evalTerm(
    term: RewardTermSpec,
    robotData: MjData,
    action: Float64Array,
  ): number {
    const model = this.model!;
    const ghost = this.ghostData!;
    switch (term.kind) {
      case "anchor_pos":
        return this.expPos(this.anchorId, this.anchorId, robotData, ghost, term.std ?? 0.3);
      case "anchor_ori":
        return this.expOri(this.anchorId, this.anchorId, robotData, ghost, term.std ?? 0.4);
      case "body_pos":
        return this.expBodyPos(term, robotData, ghost);
      case "body_ori":
        return this.expBodyOri(term, robotData, ghost);
      case "body_lin_vel":
        return this.expBodyLinVel(term, robotData, ghost);
      case "body_ang_vel":
        return this.expBodyAngVel(term, robotData, ghost);
      case "foot_pos_z":
        return this.expFootPosZ(term, robotData, ghost);
      case "action_rate": {
        if (!this.hasPrevAction) return 0;
        let s = 0;
        for (let i = 0; i < action.length; i++) {
          const d = action[i] - this.prevAction[i];
          s += d * d;
        }
        return s;
      }
      case "joint_limit":
        return this.jointLimitCost(model, robotData);
      default:
        return 0;
    }
  }

  private resolveBodies(term: RewardTermSpec): BodyIds {
    const names = term.body_names || [];
    const out: BodyIds = [];
    for (const n of names) {
      const id = this.bodyCache.get(n) ?? -1;
      if (id >= 0) out.push({ name: n, id });
    }
    return out;
  }

  private stdFor(term: RewardTermSpec, bodyName: string, fallback: number): number {
    const s = term.std;
    if (typeof s === "number") return s;
    if (s && typeof s === "object" && bodyName in s) return s[bodyName];
    return fallback;
  }

  private expPos(
    robotId: number,
    ghostId: number,
    robot: MjData,
    ghost: MjData,
    std: number | Record<string, number>,
  ): number {
    const model = this.model!;
    const rp = asArray(robot.xpos, model.nbody * 3);
    const gp = asArray(ghost.xpos, model.nbody * 3);
    const o = robotId * 3;
    const g = ghostId * 3;
    const err =
      (rp[o] - gp[g]) ** 2 +
      (rp[o + 1] - gp[g + 1]) ** 2 +
      (rp[o + 2] - gp[g + 2]) ** 2;
    const s = typeof std === "number" ? std : 0.3;
    return Math.exp(-err / (s * s));
  }

  private expOri(
    robotId: number,
    ghostId: number,
    robot: MjData,
    ghost: MjData,
    std: number | Record<string, number>,
  ): number {
    const model = this.model!;
    const rq = asArray(robot.xquat, model.nbody * 4);
    const gq = asArray(ghost.xquat, model.nbody * 4);
    const ang = quatErrorMagnitude(
      gq.subarray(ghostId * 4, ghostId * 4 + 4),
      rq.subarray(robotId * 4, robotId * 4 + 4),
    );
    const s = typeof std === "number" ? std : 0.4;
    return Math.exp(-(ang * ang) / (s * s));
  }

  private expBodyPos(term: RewardTermSpec, robot: MjData, ghost: MjData): number {
    const bodies = this.resolveBodies(term);
    if (!bodies.length) return 0;
    const model = this.model!;
    const rp = asArray(robot.xpos, model.nbody * 3);
    const gp = asArray(ghost.xpos, model.nbody * 3);
    let acc = 0;
    for (const b of bodies) {
      const o = b.id * 3;
      const err =
        (rp[o] - gp[o]) ** 2 +
        (rp[o + 1] - gp[o + 1]) ** 2 +
        (rp[o + 2] - gp[o + 2]) ** 2;
      const s = this.stdFor(term, b.name, 0.3);
      acc += err / (s * s);
    }
    return Math.exp(-acc / bodies.length);
  }

  /** Match training ``motion_foot_pos_z_error_exp``: Z-only, mean over feet. */
  private expFootPosZ(term: RewardTermSpec, robot: MjData, ghost: MjData): number {
    const bodies = this.resolveBodies(term);
    if (!bodies.length) return 0;
    const model = this.model!;
    const rp = asArray(robot.xpos, model.nbody * 3);
    const gp = asArray(ghost.xpos, model.nbody * 3);
    const s =
      typeof term.std === "number"
        ? term.std
        : this.stdFor(term, bodies[0].name, 0.1);
    let acc = 0;
    for (const b of bodies) {
      const dz = rp[b.id * 3 + 2] - gp[b.id * 3 + 2];
      acc += (dz * dz) / (s * s);
    }
    return Math.exp(-acc / bodies.length);
  }

  private expBodyOri(term: RewardTermSpec, robot: MjData, ghost: MjData): number {
    const bodies = this.resolveBodies(term);
    if (!bodies.length) return 0;
    const model = this.model!;
    const rq = asArray(robot.xquat, model.nbody * 4);
    const gq = asArray(ghost.xquat, model.nbody * 4);
    let acc = 0;
    for (const b of bodies) {
      const ang = quatErrorMagnitude(
        gq.subarray(b.id * 4, b.id * 4 + 4),
        rq.subarray(b.id * 4, b.id * 4 + 4),
      );
      const s = this.stdFor(term, b.name, 0.4);
      acc += (ang * ang) / (s * s);
    }
    return Math.exp(-acc / bodies.length);
  }

  private expBodyLinVel(term: RewardTermSpec, robot: MjData, ghost: MjData): number {
    const bodies = this.resolveBodies(term);
    if (!bodies.length) return 0;
    const model = this.model!;
    const rc = asArray(robot.cvel, model.nbody * 6);
    const gc = asArray(ghost.cvel, model.nbody * 6);
    let acc = 0;
    for (const b of bodies) {
      const o = b.id * 6 + 3; // translational part
      const err =
        (rc[o] - gc[o]) ** 2 +
        (rc[o + 1] - gc[o + 1]) ** 2 +
        (rc[o + 2] - gc[o + 2]) ** 2;
      const s = this.stdFor(term, b.name, 1.0);
      acc += err / (s * s);
    }
    return Math.exp(-acc / bodies.length);
  }

  private expBodyAngVel(term: RewardTermSpec, robot: MjData, ghost: MjData): number {
    const bodies = this.resolveBodies(term);
    if (!bodies.length) return 0;
    const model = this.model!;
    const rc = asArray(robot.cvel, model.nbody * 6);
    const gc = asArray(ghost.cvel, model.nbody * 6);
    let acc = 0;
    for (const b of bodies) {
      const o = b.id * 6;
      const err =
        (rc[o] - gc[o]) ** 2 +
        (rc[o + 1] - gc[o + 1]) ** 2 +
        (rc[o + 2] - gc[o + 2]) ** 2;
      const s = this.stdFor(term, b.name, 3.14);
      acc += err / (s * s);
    }
    return Math.exp(-acc / bodies.length);
  }

  private jointLimitCost(model: MjModel, data: MjData): number {
    // Soft hinge limit penalty: sum of squared overflows (matches Isaac-style).
    let cost = 0;
    const n = model.njnt;
    const qpos = asArray(data.qpos, model.nq);
    for (let j = 0; j < n; j++) {
      const jtype = Number(model.jnt_type[j]);
      if (jtype !== 3) continue; // mjJNT_HINGE ≈ 3
      const limited = Number(model.jnt_limited[j]);
      if (!limited) continue;
      const adr = Number(model.jnt_qposadr[j]);
      const lo = Number(model.jnt_range[j * 2]);
      const hi = Number(model.jnt_range[j * 2 + 1]);
      const q = qpos[adr];
      if (q < lo) cost += (lo - q) ** 2;
      else if (q > hi) cost += (q - hi) ** 2;
    }
    return cost;
  }

  private mean(name: string): number {
    const h = this.histories.get(name);
    if (!h?.length) return 0;
    let s = 0;
    for (const v of h) s += v;
    return s / h.length;
  }

  private renderUnavailable() {
    this.rows = [];
    this.rowKey = "";
    this.lastValues = [];
    if (this.el) {
      this.el.innerHTML =
        '<div class="reward-empty">当前 pack 没有奖励配置（重新打包或重启 play）</div>';
    }
    if (this.totalEl) this.totalEl.textContent = "Σ ≈ —";
  }

  private renderEmpty() {
    this.lastBarMs = 0;
    this.rows = [];
    this.rowKey = "";
    this.lastValues = [];
    if (this.el) {
      this.el.innerHTML =
        '<div class="reward-empty">Waiting for data…</div>';
    }
    if (this.totalEl) this.totalEl.textContent = "Σ ≈ —";
  }

  private ensureRows(values: RowValue[]) {
    const key = values.map((v) => v.name).join("|");
    if (key === this.rowKey && this.rows.length === values.length) return;
    if (!this.el) return;
    this.el.innerHTML = "";
    this.rows = values.map(({ name, weight, info }) => {
      const root = document.createElement("div");
      root.className = "reward-row";
      const label = document.createElement("span");
      label.className = "reward-name";
      if (info) {
        label.style.paddingLeft = "1.2em";
        label.style.opacity = "0.8";
      }
      label.textContent = name;
      label.title = `${name}  (w=${weight})`;
      const track = document.createElement("div");
      track.className = "reward-bar-track";
      const fill = document.createElement("div");
      fill.className = "reward-bar-fill";
      const val = document.createElement("span");
      val.className = "reward-val";
      track.append(fill, val);
      root.append(label, track);
      this.el!.appendChild(root);
      return { root, name: label, fill, val };
    });
    this.rowKey = key;
  }

  /** Store the latest step; painting happens in ``tick`` (animation frame). */
  private render(
    values: RowValue[],
    totalInstant: number,
  ) {
    this.lastValues = values;
    this.lastTotal = totalInstant;
    this.dirty = true;
  }

  /** Call every animation frame: bars ~60 Hz, numbers ~20 Hz. */
  tick(nowMs: number) {
    if (!this.el || !this.visible || !this.lastValues.length) return;
    const force = this.lastBarMs === 0;
    if (!this.dirty && !force) return;
    if (!force && nowMs - this.lastBarMs < this.barIntervalMs) return;
    const dtMs = force ? 0 : nowMs - this.lastBarMs;
    this.lastBarMs = nowMs;
    this.dirty = false;
    const values = this.lastValues;
    this.ensureRows(values);

    const means = values.map((v) => ({
      ...v,
      mean: this.histories.get(v.name)?.length ? this.mean(v.name) : NaN,
    }));
    let maxAbs = 1e-8;
    for (const m of means) {
      if (Number.isFinite(m.mean)) maxAbs = Math.max(maxAbs, Math.abs(m.mean));
    }
    // Grow instantly, shrink slowly so bars don't jump when the top term changes.
    if (force || maxAbs >= this.barScale) this.barScale = maxAbs;
    else this.barScale += (maxAbs - this.barScale) * (1 - Math.exp(-dtMs / 1000));

    const text = force || nowMs - this.lastTextMs >= this.textIntervalMs;
    if (text) this.lastTextMs = nowMs;
    means.forEach(({ name, mean, weight, note, info }, i) => {
      const row = this.rows[i];
      if (!row) return;
      if (text) {
        row.name.textContent = note ? `${name} [${note}]` : name;
        row.name.style.fontWeight = info && note?.endsWith("◀") ? "bold" : "";
      }
      if (!Number.isFinite(mean)) {
        if (text) {
          row.fill.style.width = "0%";
          row.val.textContent = "n/a";
          row.root.title = `${name}: 浏览器端未实现`;
        }
        return;
      }
      const pct = Math.min(100, (Math.abs(mean) / this.barScale) * 100);
      row.fill.style.width = `${pct.toFixed(1)}%`;
      row.fill.style.background = info ? "#5c8dd6" : mean >= 0 ? "#4caf50" : "#f44336";
      if (!text) return;
      row.val.textContent =
        Math.abs(mean) < 1e-3 && mean !== 0 ? mean.toExponential(2) : mean.toFixed(4);
      const rawMean = weight !== 0 ? mean / weight : 0;
      row.root.title =
        `${name}\nw=${weight.toPrecision(4)}\nraw≈${rawMean.toPrecision(4)}\nw·raw≈${mean.toPrecision(4)}` +
        (name === STYLE_TERM ? `\nw = (1-λ)/(λ·dt)，与任务项 w·raw 同单位${note ? `\n${note}` : ""}` : "") +
        (info ? "\n假设用该标签时的风格（仅参考，不计入 Σ）；◀ = 当前指令对应的标签" : "");
    });
    if (text && this.totalEl) {
      const meanTotal = means.reduce(
        (s, m) => s + (!m.info && Number.isFinite(m.mean) ? m.mean : 0),
        0,
      );
      this.totalEl.textContent =
        `Σ ≈ ${meanTotal.toFixed(3)}  ×dt ${(meanTotal * this.policyDt).toFixed(4)}` +
        `  (inst ${this.lastTotal.toFixed(3)})`;
    }
  }

  private disposeGhost() {
    this.ghostData?.delete?.();
    this.ghostData = null;
  }

  dispose() {
    this.disposeGhost();
    this.cfg = null;
    this.vel = null;
    this.style = null;
  }
}
