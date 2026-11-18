import type { MjData, MjModel } from "mujoco-js";
import { asArray, name2id, type MujocoModule } from "./mujoco_util";
import type { PolicyController } from "./policy";
import type { AmpDiscSpec } from "./types";

const MJ_JNT_FREE = 0;

type Dense = { w: Float32Array; b: Float32Array; nIn: number; nOut: number };

type Weights = {
  mean: Float32Array;
  std: Float32Array;
  encoders: { indices: number[]; layers: Dense[] }[];
  gate: Dense | null;
  trunk: Dense[];
  head: Dense;
};

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

function dense(x: Float32Array, l: Dense, out: Float32Array, relu: boolean): Float32Array {
  const { w, b, nIn, nOut } = l;
  for (let o = 0; o < nOut; o++) {
    let acc = b[o];
    const r = o * nIn;
    for (let i = 0; i < nIn; i++) acc += w[r + i] * x[i];
    out[o] = relu && acc < 0 ? 0 : acc;
  }
  return out;
}

function mlp(x: Float32Array, layers: Dense[], reluLast: boolean): Float32Array {
  let h = x;
  layers.forEach((l, k) => {
    h = dense(h, l, new Float32Array(l.nOut), reluLast || k < layers.length - 1);
  });
  return h;
}

/**
 * LS-AMP style reward on the browser sim (``AmpPPO.process_env_step`` parity):
 * ``coef * max(0, 1 - 0.25 (D(s, s') - 1)^2)`` with the label taken from the
 * command at the time ``s`` was observed. Weights are swapped per checkpoint.
 */
export class AmpStyleEval {
  readonly spec: AmpDiscSpec;
  /** Same units as task ``w·raw`` rows: training adds ``(1-λ)·style`` vs ``λ·dt·Σ w·raw``. */
  readonly weight: number;
  label = "";
  prediction = NaN;
  /** Style / D for every label on the same (s, s') pair (index = ``spec.label_names``). */
  readonly perLabel: Float64Array;
  readonly perLabelD: Float64Array;

  private readonly model: MjModel;
  private readonly rootId: number;
  private readonly freeDofAdr: number;
  private readonly qposAdr: number[];
  private readonly dofAdr: number[];
  private readonly keyIds: number[];
  private weights: Weights | null = null;
  private loadedFile = "";
  private prevState: Float32Array | null = null;
  private prevLabel = -1;

  constructor(mj: MujocoModule, model: MjModel, spec: AmpDiscSpec, rootBody: string, policyDt: number) {
    this.spec = spec;
    this.model = model;
    this.perLabel = new Float64Array(spec.label_names.length).fill(NaN);
    this.perLabelD = new Float64Array(spec.label_names.length).fill(NaN);
    const lerp = spec.task_reward_lerp;
    this.weight = (1 - lerp) / (lerp * policyDt);
    this.rootId = name2id(mj, model, "body", rootBody);
    let freeD = 0;
    for (let j = 0; j < model.njnt; j++) {
      if (Number(model.jnt_type[j]) === MJ_JNT_FREE && Number(model.jnt_bodyid[j]) === this.rootId) {
        freeD = Number(model.jnt_dofadr[j]);
        break;
      }
    }
    this.freeDofAdr = freeD;
    const jids = spec.joint_names.map((n) => name2id(mj, model, "joint", n));
    if (jids.some((j) => j < 0)) throw new Error("AMP joint missing in scene");
    this.qposAdr = jids.map((j) => Number(model.jnt_qposadr[j]));
    this.dofAdr = jids.map((j) => Number(model.jnt_dofadr[j]));
    this.keyIds = spec.key_body_names.map((n) => name2id(mj, model, "body", n));
    if (this.keyIds.some((b) => b < 0)) throw new Error("AMP key body missing in scene");
  }

  get ready(): boolean {
    return this.weights !== null;
  }

  async loadWeights(url: string, file: string) {
    if (file === this.loadedFile && this.weights) return;
    const res = await fetch(`${url}?t=${Date.now()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status} ${file}`);
    const buf = await res.arrayBuffer();
    const t = new Map<string, { data: Float32Array; shape: number[] }>();
    for (const e of this.spec.layout) {
      const n = e.shape.reduce((a, b) => a * b, 1);
      let data: Float32Array;
      if (e.dtype === "f32") {
        data = new Float32Array(buf.slice(e.offset, e.offset + 4 * n));
      } else {
        const h = new Uint16Array(buf, e.offset, n);
        data = new Float32Array(n);
        for (let i = 0; i < n; i++) data[i] = halfToFloat(h[i]);
      }
      t.set(e.name, { data, shape: e.shape });
    }
    const get = (name: string) => {
      const v = t.get(name);
      if (!v) throw new Error(`disc tensor ${name} missing`);
      return v;
    };
    const layer = (prefix: string): Dense => {
      const w = get(`${prefix}.weight`);
      return { w: w.data, b: get(`${prefix}.bias`).data, nOut: w.shape[0], nIn: w.shape[1] };
    };
    const seq = (prefix: string): Dense[] =>
      [...t.keys()]
        .map((k) => k.match(new RegExp(`^${prefix.replace(/\./g, "\\.")}\\.(\\d+)\\.weight$`)))
        .filter((m): m is RegExpMatchArray => !!m)
        .map((m) => Number(m[1]))
        .sort((a, b) => a - b)
        .map((i) => layer(`${prefix}.${i}`));

    const eps = this.spec.normalizer_epsilon;
    const variance = get("norm.var").data;
    const std = variance.map((v) => Math.sqrt(v + eps));
    const partGate = this.spec.type === "part_gate";
    this.weights = {
      mean: get("norm.mean").data,
      std,
      encoders: partGate
        ? this.spec.parts.map((p) => ({ indices: p.indices, layers: seq(`trunk.encoders.${p.name}`) }))
        : [],
      gate: partGate ? layer("trunk.gate") : null,
      trunk: partGate ? seq("trunk.fuse") : seq("trunk"),
      head: layer("head"),
    };
    this.loadedFile = file;
  }

  reset() {
    this.prevState = null;
    this.prevLabel = -1;
    this.prediction = NaN;
    this.label = "";
    this.perLabel.fill(NaN);
    this.perLabelD.fill(NaN);
  }

  /** Style reward for (previous policy step → now). NaN on the first step / before weights load. */
  evaluate(policy: PolicyController, data: MjData): number {
    const state = this.ampState(data);
    const prev = this.prevState;
    const prevLabel = this.prevLabel;
    this.prevState = state;
    this.prevLabel = this.motionLabel(policy.getCommand());
    if (!prev || prevLabel < 0 || !this.weights) return NaN;
    this.label = this.spec.label_names[prevLabel];
    const preds = this.discriminate(this.normalize(prev), this.normalize(state));
    preds.forEach((d, k) => {
      this.perLabelD[k] = d;
      this.perLabel[k] = this.spec.reward_coefficient * Math.max(0, 1 - 0.25 * (d - 1) ** 2);
    });
    this.prediction = this.perLabelD[prevLabel];
    return this.perLabel[prevLabel];
  }

  /** ``AmpPPO._command_motion_label`` with the Viser joystick command-mode rule. */
  private motionLabel(cmd: Float64Array): number {
    const names = this.spec.label_names;
    const lin = Math.hypot(cmd[0], cmd[1]);
    const yaw = Math.abs(cmd[2]);
    const standing = (lin < 0.05 && yaw < 0.05) || lin + yaw <= 0.05;
    if (standing) return names.indexOf("stand");
    if (lin < 0.1) return names.indexOf("turn");
    if (yaw < 0.1 && Math.abs(cmd[0]) < 0.1) return names.indexOf("side");
    return names.indexOf(cmd[0] > this.spec.velocity_threshold ? "run" : "walk");
  }

  private normalize(s: Float32Array): Float32Array {
    const w = this.weights!;
    const clip = this.spec.normalizer_clip;
    const out = new Float32Array(s.length);
    for (let i = 0; i < s.length; i++) {
      const v = (s[i] - w.mean[i]) / w.std[i];
      out[i] = v > clip ? clip : v < -clip ? -clip : v;
    }
    return out;
  }

  /** D(s, s' | label) for every label; part tokens are label-independent so run once. */
  private discriminate(s: Float32Array, n: Float32Array): number[] {
    const w = this.weights!;
    const nLabel = this.spec.label_names.length;
    const oneHot = (label: number) => Float32Array.from({ length: nLabel }, (_, i) => (i === label ? 1 : 0));
    const head = (h: Float32Array) => dense(h, w.head, new Float32Array(1), false)[0];
    if (w.gate) {
      const gate = w.gate;
      const tokens = w.encoders.map((enc) => {
        const k = enc.indices.length;
        const x = new Float32Array(2 * k);
        for (let i = 0; i < k; i++) {
          x[i] = s[enc.indices[i]];
          x[k + i] = n[enc.indices[i]];
        }
        return mlp(x, enc.layers, true);
      });
      return Array.from({ length: nLabel }, (_, label) => {
        const logits = dense(oneHot(label), gate, new Float32Array(gate.nOut), false);
        const mx = Math.max(...logits);
        const ex = logits.map((v) => Math.exp(v - mx));
        const z = ex.reduce((a, b) => a + b, 0);
        const h = new Float32Array(tokens[0].length);
        tokens.forEach((tok, p) => {
          const a = ex[p] / z;
          for (let i = 0; i < h.length; i++) h[i] += a * tok[i];
        });
        return head(mlp(h, w.trunk, false));
      });
    }
    const d = s.length;
    return Array.from({ length: nLabel }, (_, label) => {
      const x = new Float32Array(2 * (d + nLabel));
      x.set(s, 0);
      x[d + label] = 1;
      x.set(n, d + nLabel);
      x[2 * d + nLabel + label] = 1;
      return head(mlp(x, w.trunk, true));
    });
  }

  /** ``src.tasks.amp.mdp.observations.amp_state`` (root-frame, link frames). */
  private ampState(data: MjData): Float32Array {
    const model = this.model;
    const qpos = asArray(data.qpos, model.nq);
    const qvel = asArray(data.qvel, model.nv);
    const xpos = asArray(data.xpos, model.nbody * 3);
    const xmat = asArray(data.xmat, model.nbody * 9);
    const nj = this.qposAdr.length;
    const nk = this.keyIds.length;
    const out = new Float32Array(6 + 2 * nj + 9 * nk);
    const r = this.rootId * 9;
    const R = (i: number, j: number) => xmat[r + i * 3 + j];
    // Rᵀ v (xmat is row-major world-from-body).
    const toRoot = (v0: number, v1: number, v2: number, o: number) => {
      for (let i = 0; i < 3; i++) out[o + i] = R(0, i) * v0 + R(1, i) * v1 + R(2, i) * v2;
    };
    const d = this.freeDofAdr;
    toRoot(qvel[d], qvel[d + 1], qvel[d + 2], 0);
    out[3] = qvel[d + 3];
    out[4] = qvel[d + 4];
    out[5] = qvel[d + 5];
    for (let j = 0; j < nj; j++) {
      out[6 + j] = qpos[this.qposAdr[j]];
      out[6 + nj + j] = qvel[this.dofAdr[j]];
    }
    const p0 = this.rootId * 3;
    const posOff = 6 + 2 * nj;
    const oriOff = posOff + 3 * nk;
    this.keyIds.forEach((b, k) => {
      toRoot(xpos[b * 3] - xpos[p0], xpos[b * 3 + 1] - xpos[p0 + 1], xpos[b * 3 + 2] - xpos[p0 + 2], posOff + 3 * k);
      const m = b * 9;
      // (Rrootᵀ Rkey)[:, :2], flattened row-major.
      for (let i = 0; i < 3; i++) {
        for (let c = 0; c < 2; c++) {
          out[oriOff + 6 * k + 2 * i + c] =
            R(0, i) * xmat[m + c] + R(1, i) * xmat[m + 3 + c] + R(2, i) * xmat[m + 6 + c];
        }
      }
    });
    return out;
  }
}
