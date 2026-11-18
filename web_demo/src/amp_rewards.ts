import type { MjData, MjModel } from "mujoco-js";
import { ContactGroups, type ContactScan } from "./contacts";
import { asArray, name2id, type MujocoModule } from "./mujoco_util";
import type { PolicyController } from "./policy";
import type { AmpRewardConfig, AmpRewardTerm, AmpTurnProfile } from "./types";

const MJ_JNT_FREE = 0;

/** ``src.tasks.amp.mdp.commands`` COMMAND_MODE_*. */
const enum Mode {
  Mixed,
  Standing,
  Turning,
  Lateral,
}

type Vec3 = [number, number, number];

/** mjlab ContactSensor air-time state, advanced once per policy step. */
type AirTime = {
  air: Float64Array;
  contact: Float64Array;
  lastAir: Float64Array;
};

type Frame = {
  cmd: Float64Array;
  mode: Mode;
  qpos: Float64Array;
  qvel: Float64Array;
  xpos: Float64Array;
  xmat: Float64Array;
  cvel: Float64Array;
  com: Float64Array;
  contacts: Map<string, ContactScan>;
  action: Float64Array;
};

const relu = (x: number) => Math.max(0, x);
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/**
 * Raw (unweighted) AMP walk/run task rewards on the browser sim, once per policy
 * step. Command mode is inferred from the twist like the Viser joystick
 * (``UniformVelocityCommand.compute``). Contact force history / air time use the
 * policy-step state instead of per-substep buffers.
 */
export class AmpRewardEval {
  readonly terms: AmpRewardTerm[];
  private readonly model: MjModel;
  private readonly dt: number;
  private readonly groups: ContactGroups;
  private readonly profiles: AmpRewardConfig["turn_profiles"];

  private readonly rootId: number;
  private readonly freeDofAdr: number;
  private readonly jointNames: string[];
  private readonly defaults: Float64Array;
  private readonly qposAdr = new Map<string, number>();
  private readonly dofAdr = new Map<string, number>();
  private readonly range = new Map<string, [number, number]>();
  private readonly bodyIds = new Map<string, number>();

  private prevAction: Float64Array;
  private prevPrevAction: Float64Array;
  private readonly airTime = new Map<string, AirTime>();
  private turnPhase = 0;
  private readonly out: Float64Array;
  /** Play-side mirror of slow_walk_origin / yaw (set when entering slow-straight cmd). */
  private pathOrigin: [number, number] | null = null;
  private pathYaw = 0;
  private pathActive = false;

  constructor(mj: MujocoModule, model: MjModel, cfg: AmpRewardConfig, policy: PolicyController) {
    this.model = model;
    this.terms = cfg.terms;
    this.dt = policy.manifest.policy_dt;
    this.groups = new ContactGroups(mj, model);
    this.profiles = cfg.turn_profiles;
    this.out = new Float64Array(cfg.terms.length);

    this.rootId = name2id(mj, model, "body", cfg.root_body || "pelvis");
    let freeD = 0;
    for (let j = 0; j < model.njnt; j++) {
      if (Number(model.jnt_type[j]) === MJ_JNT_FREE && Number(model.jnt_bodyid[j]) === this.rootId) {
        freeD = Number(model.jnt_dofadr[j]);
        break;
      }
    }
    this.freeDofAdr = freeD;

    this.jointNames = policy.jointNames;
    this.defaults = Float64Array.from(policy.defaultPos);
    for (const jn of this.jointNames) {
      const jid = name2id(mj, model, "joint", jn);
      this.qposAdr.set(jn, Number(model.jnt_qposadr[jid]));
      this.dofAdr.set(jn, Number(model.jnt_dofadr[jid]));
      this.range.set(jn, [Number(model.jnt_range[jid * 2]), Number(model.jnt_range[jid * 2 + 1])]);
    }
    this.prevAction = new Float64Array(this.jointNames.length);
    this.prevPrevAction = new Float64Array(this.jointNames.length);

    for (const t of this.terms) {
      for (const b of [t.body, ...(t.feet || [])]) {
        if (b && !this.bodyIds.has(b)) this.bodyIds.set(b, name2id(mj, model, "body", b));
      }
      if (t.contact) {
        this.groups.add(t.contact.pattern, t.contact.mode);
        if (!this.airTime.has(t.contact.pattern)) {
          this.airTime.set(t.contact.pattern, { air: new Float64Array(0), contact: new Float64Array(0), lastAir: new Float64Array(0) });
        }
      }
    }
  }

  reset() {
    this.prevAction.fill(0);
    this.prevPrevAction.fill(0);
    this.pathOrigin = null;
    this.pathYaw = 0;
    this.pathActive = false;
    for (const s of this.airTime.values()) {
      s.air.fill(0);
      s.contact.fill(0);
      s.lastAir.fill(0);
    }
    this.turnPhase = 0;
  }

  /** Raw value per term (NaN = not evaluated in browser). Call after ``policy.step``. */
  evaluate(policy: PolicyController, data: MjData): Float64Array {
    const model = this.model;
    const cmd = policy.getCommand();
    const f: Frame = {
      cmd,
      mode: this.commandMode(cmd),
      qpos: asArray(data.qpos, model.nq),
      qvel: asArray(data.qvel, model.nv),
      xpos: asArray(data.xpos, model.nbody * 3),
      xmat: asArray(data.xmat, model.nbody * 9),
      cvel: asArray(data.cvel, model.nbody * 6),
      com: asArray(data.subtree_com, model.nbody * 3),
      contacts: this.groups.scan(data),
      action: policy.getLastAction(),
    };
    this.updateAirTime(f.contacts);

    for (let k = 0; k < this.terms.length; k++) {
      let v: number;
      try {
        v = this.term(this.terms[k], f, data);
      } catch {
        v = NaN;
      }
      this.out[k] = v;
    }

    this.prevPrevAction.set(this.prevAction);
    this.prevAction.set(f.action.subarray(0, this.prevAction.length));
    return this.out;
  }

  private term(t: AmpRewardTerm, f: Frame, data: MjData): number {
    const p = (key: string, def: number) => {
      const v = t.params?.[key];
      return typeof v === "number" ? v : def;
    };
    const flag = (key: string, def: boolean) => {
      const v = t.params?.[key];
      return typeof v === "boolean" ? v : def;
    };
    const { cmd, mode } = f;
    const lin = Math.hypot(cmd[0], cmd[1]);
    const yaw = Math.abs(cmd[2]);
    const magnitude = lin + yaw;
    const moving = magnitude > 0.05 ? 1 : 0;
    const body = this.bodyIds.get(t.body ?? "") ?? this.rootId;
    const feet = (t.feet || []).map((n) => this.bodyIds.get(n) ?? -1);
    const joints = t.joints ?? this.jointNames;
    const contact = t.contact ? f.contacts.get(t.contact.pattern) : undefined;
    const air = t.contact ? this.airTime.get(t.contact.pattern) : undefined;
    const notTurning = (key: string) => (flag(key, false) && mode === Mode.Turning ? 0 : 1);
    const nonForward = mode === Mode.Turning || mode === Mode.Lateral ? 1 : 0;
    const walkMask = () => {
      const walk =
        (mode === Mode.Mixed && cmd[0] <= p("maximum_forward_speed", 0.8)) ||
        (flag("include_turning", true) && mode === Mode.Turning);
      return walk && magnitude > p("minimum_command", 0.05) ? 1 : 0;
    };

    switch (t.kind) {
      case "track_lin_vel_xy": {
        const vb = this.rootLinVelB(f);
        const err = (cmd[0] - vb[0]) ** 2 + (cmd[1] - vb[1]) ** 2;
        const minCmd = p("min_command", 0);
        return Math.exp(-err / p("std", 0.5) ** 2) * (minCmd > 0 && lin < minCmd ? 0 : 1);
      }
      case "track_yaw_vel": {
        const err = (cmd[2] - this.rootAngVelB(f)[2]) ** 2;
        const minCmd = p("min_command", 0);
        return Math.exp(-err / p("std", 0.5) ** 2) * (minCmd > 0 && yaw < minCmd ? 0 : 1);
      }
      case "yaw_rate_error": {
        const minCmd = p("min_command", 0);
        return (cmd[2] - this.rootAngVelB(f)[2]) ** 2 * (minCmd > 0 && yaw < minCmd ? 0 : 1);
      }
      case "lin_vel_xy_error": {
        const vb = this.rootLinVelB(f);
        const err = (cmd[0] - vb[0]) ** 2 + (cmd[1] - vb[1]) ** 2;
        const minCmd = p("min_command", 0);
        return err * (minCmd > 0 && lin < minCmd ? 0 : 1);
      }
      case "straight_walk": {
        const off = this.pathOffset(f, cmd, yaw);
        if (!off) return 0;
        const { lateral, forward } = off;
        if (forward < p("min_forward_progress", 0.05)) return 0;
        const straight = Math.exp(-(lateral * lateral) / p("std", 0.05) ** 2);
        const progress = clamp(forward / Math.max(cmd[0], 1e-3), 0, 1);
        return straight * progress;
      }
      case "straight_walk_penalty": {
        const off = this.pathOffset(f, cmd, yaw);
        if (!off) return 0;
        const { lateral, forward } = off;
        if (forward < p("min_forward_progress", 0.05)) return 0;
        const excess = relu(Math.abs(lateral) - p("deadband", 0.01));
        return excess * excess;
      }
      case "zero_command_joint": {
        const linT = t.params?.lin_threshold;
        const yawT = t.params?.yaw_threshold;
        const gate =
          typeof linT === "number" && typeof yawT === "number"
            ? lin < linT && yaw < yawT
            : magnitude < p("threshold", 0.05);
        return gate ? this.jointDevSq(joints, f) : 0;
      }
      case "alive":
        return 1;
      case "lin_vel_z":
        return this.rootLinVelB(f)[2] ** 2;
      case "root_ang_vel_xy": {
        const w = this.rootAngVelB(f);
        return w[0] ** 2 + w[1] ** 2;
      }
      case "joint_vel":
        return joints.reduce((s, jn) => s + f.qvel[this.dofAdr.get(jn)!] ** 2, 0);
      case "joint_acc": {
        const qacc = asArray(data.qacc, this.model.nv);
        return joints.reduce((s, jn) => s + qacc[this.dofAdr.get(jn)!] ** 2, 0);
      }
      case "action_rate": {
        let s = 0;
        for (let i = 0; i < this.prevAction.length; i++) s += (f.action[i] - this.prevAction[i]) ** 2;
        return s;
      }
      case "action_acc": {
        let s = 0;
        for (let i = 0; i < this.prevAction.length; i++) {
          s += (f.action[i] - 2 * this.prevAction[i] + this.prevPrevAction[i]) ** 2;
        }
        return s;
      }
      case "joint_pos_limits": {
        const fac = t.soft_factor ?? 1.0;
        let s = 0;
        for (const jn of joints) {
          const [lo, hi] = this.range.get(jn)!;
          const mid = 0.5 * (lo + hi);
          const half = 0.5 * (hi - lo) * fac;
          const q = f.qpos[this.qposAdr.get(jn)!];
          s += relu(mid - half - q) + relu(q - (mid + half));
        }
        return s;
      }
      case "energy": {
        const qfrc = asArray(data.qfrc_actuator, this.model.nv);
        let s = 0;
        for (const jn of joints) {
          const d = this.dofAdr.get(jn)!;
          s += (qfrc[d] * f.qvel[d]) ** 2;
        }
        return Math.sqrt(s);
      }
      case "body_orientation": {
        const g = this.gravityB(f, body);
        return g[0] ** 2 + g[1] ** 2;
      }
      case "body_ang_vel_xy": {
        const w = this.bodyAngVelB(f, body);
        return w[0] ** 2 + w[1] ** 2;
      }
      case "body_roll":
        return this.gravityB(f, body)[1] ** 2;
      case "body_roll_ang_vel":
        return this.bodyAngVelB(f, body)[0] ** 2;
      case "stand_body_pitch":
        return this.gravityB(f, body)[0] ** 2 * (1 - moving);
      case "stand_feet_heading": {
        if (magnitude >= p("threshold", 0.05)) return 0;
        return feet.reduce((s, b) => s + this.footHeadingErr(f, b) ** 2, 0);
      }
      case "feet_pelvis_heading": {
        const blend = yaw / (yaw + p("turn_yaw_scale", 0.5));
        const scale = (1 - blend * (1 - p("turn_floor", 0.3))) * (mode === Mode.Turning ? 0 : 1);
        return feet.reduce((s, b) => s + this.footHeadingErr(f, b) ** 2, 0) * scale;
      }
      case "stable_stand_joint_norm":
        return Math.sqrt(this.jointDevSq(joints, f)) * this.stableStandWeight(t, f, body, contact, magnitude);
      case "stable_stand_feet_fore_aft": {
        const db = p("position_deadband", 0.02);
        const err = feet.reduce((s, b) => s + relu(Math.abs(this.footPosB(f, b)[0]) - db), 0);
        return err * this.stableStandWeight(t, f, body, contact, magnitude);
      }
      case "stable_stand_feet_lateral": {
        const center = 0.5 * (this.footPosB(f, feet[0])[1] + this.footPosB(f, feet[1])[1]);
        const err = relu(Math.abs(center) - p("center_deadband", 0.01));
        return err * this.stableStandWeight(t, f, body, contact, magnitude);
      }
      case "soft_landing": {
        if (!contact || !air) return NaN;
        let s = 0;
        for (let i = 0; i < contact.force.length; i++) {
          if (this.firstContact(air, i)) s += contact.force[i];
        }
        return s * (magnitude > p("command_threshold", 0.05) ? 1 : 0);
      }
      case "feet_slide": {
        if (!contact) return NaN;
        const thr = p("threshold", 1.0);
        let s = 0;
        feet.forEach((b, i) => {
          if (contact.force[i] <= thr) return;
          const v = this.bodyLinVelW(f, b);
          s += Math.hypot(v[0], v[1]);
        });
        return s * notTurning("disable_turning");
      }
      case "walk_swing_clearance": {
        if (!air) return NaN;
        const target = p("target_clearance", 0.08);
        const h = p("contact_height", 0.06);
        let s = 0;
        feet.forEach((b, i) => {
          const deficit = Math.min(1, relu(target - (f.xpos[b * 3 + 2] - h)) / target);
          const swing = air.air[i] > p("minimum_air_time", 0.16) && air.air[i] < p("maximum_air_time", 0.36);
          const supported = this.otherContactTime(air, i) > p("minimum_support_time", 0.04);
          if (swing && supported) s += deficit ** 2;
        });
        return (s / Math.max(1, feet.length)) * walkMask();
      }
      case "walk_air_time": {
        if (!air) return NaN;
        const target = p("target_air_time", 0.48);
        const std = p("std", 0.12);
        let s = 0;
        for (let i = 0; i < air.air.length; i++) {
          if (!this.firstContact(air, i)) continue;
          if (this.otherContactTime(air, i) <= p("minimum_support_time", 0.04)) continue;
          s += Math.min(1, Math.abs(air.lastAir[i] - target) / std) ** 2;
        }
        return s * walkMask();
      }
      case "touchdown_foot_vel": {
        if (!air) return NaN;
        const db = p("velocity_deadband", 0.25);
        const vmax = p("maximum_velocity", 2.0);
        let s = 0;
        feet.forEach((b, i) => {
          if (!this.firstContact(air, i)) return;
          const v = this.bodyLinVelW(f, b);
          s += clamp(relu(Math.hypot(v[0], v[1]) - db), 0, vmax - db) ** 2;
        });
        return s * walkMask();
      }
      case "turn_clearance_lift": {
        if (!air) return NaN;
        const turning = mode === Mode.Turning && yaw >= p("min_yaw_command", 0.1);
        const minContact = p("minimum_contact_time", 0.02);
        const inContact = Array.from(air.contact, (c) => c > minContact);
        const single = inContact.filter(Boolean).length === 1;
        const swinging = inContact.some((c, i) => !c && air.air[i] >= p("minimum_air_time", 0.08));
        if (!(turning && single && swinging)) return 0;
        const z0 = f.xpos[feet[0] * 3 + 2];
        const z1 = f.xpos[feet[1] * 3 + 2];
        const short = relu(p("target_clearance", 0.06) - Math.abs(z0 - z1));
        return Math.exp(-(short ** 2) / p("std", 0.03) ** 2);
      }
      case "turn_foot_height":
        return this.turnFootHeight(t, f, feet, yaw, p("std", 0.03), p("min_yaw_command", 0.1));
      case "turn_lateral_air_time": {
        if (!air) return NaN;
        const lo = p("minimum_air_time", 0.12);
        const full = p("full_reward_air_time", 0.28);
        let s = 0;
        for (let i = 0; i < air.air.length; i++) {
          if (!this.firstContact(air, i)) continue;
          if (this.otherContactTime(air, i) <= p("minimum_support_time", 0.04)) continue;
          s += clamp((air.lastAir[i] - lo) / (full - lo), 0, 1);
        }
        return s * nonForward;
      }
      case "turn_lateral_contact": {
        if (!air) return NaN;
        const minContact = p("minimum_contact_time", 0.02);
        const n = Array.from(air.contact).filter((c) => c > minContact).length;
        return (n === 1 ? 1 : 0) * nonForward;
      }
      case "turn_lateral_air_dense": {
        if (!air) return NaN;
        const minContact = p("minimum_contact_time", 0.02);
        const inContact = Array.from(air.contact, (c) => c > minContact);
        if (inContact.filter(Boolean).length !== 1 || !nonForward) return 0;
        const thr =
          mode === Mode.Turning ? p("turn_air_time_threshold", 0.22) : p("lateral_air_time_threshold", 0.22);
        let s = 0;
        inContact.forEach((c, i) => {
          const swingAir = c ? 0 : air.air[i];
          if (swingAir <= thr) s += swingAir / thr;
        });
        return s;
      }
      case "feet_crossing": {
        const d = this.footPosB(f, feet[0])[1] - this.footPosB(f, feet[1])[1];
        return relu(p("minimum_distance", 0.12) - d);
      }
      case "feet_spacing": {
        const d = this.footPosB(f, feet[0])[1] - this.footPosB(f, feet[1])[1];
        const outside = relu(p("minimum_distance", 0.12) - d) + relu(d - p("maximum_distance", 0.15));
        const nonLateral = Math.abs(cmd[1]) < p("lateral_command_threshold", 0.1) ? 1 : 0;
        return outside * nonLateral * notTurning("disable_turning");
      }
      case "feet_support_roll": {
        if (!contact) return NaN;
        const thr = p("contact_force_threshold", 5.0);
        let s = 0;
        feet.forEach((b, i) => {
          if (contact.force[i] > thr) s += this.gravityB(f, b)[1] ** 2;
        });
        return s * moving;
      }
      case "joint_action_rate_loaded": {
        if (!contact) return NaN;
        const thr = p("contact_force_threshold", 20.0);
        let s = 0;
        joints.forEach((jn, i) => {
          const a = this.jointNames.indexOf(jn);
          if (a >= 0 && contact.force[i] > thr) s += (f.action[a] - this.prevAction[a]) ** 2;
        });
        return s * moving;
      }
      case "joint_dev_norm":
        return Math.sqrt(this.jointDevSq(joints, f));
      case "undesired_contacts": {
        if (!contact) return NaN;
        const thr = p("threshold", 1.0);
        return Array.from(contact.force).filter((x) => x > thr).length;
      }
      case "self_collision": {
        if (!contact) return NaN;
        // Training counts substeps over force history; approximate with current state.
        const hit = contact.selfMaxForce > p("force_threshold", 10.0);
        return hit ? Math.max(1, t.contact?.history_length ?? 1) : 0;
      }
      default:
        return NaN;
    }
  }

  /** Same rule as the Viser joystick in ``UniformVelocityCommand.compute``. */
  private commandMode(cmd: Float64Array): Mode {
    const lin = Math.hypot(cmd[0], cmd[1]);
    const yaw = Math.abs(cmd[2]);
    if (lin < 0.1 && yaw < 0.1) return Mode.Standing;
    if (lin < 0.1) return Mode.Turning;
    if (yaw < 0.1 && Math.abs(cmd[0]) < 0.1) return Mode.Lateral;
    return Mode.Mixed;
  }

  private updateAirTime(contacts: Map<string, ContactScan>) {
    for (const [pattern, s] of this.airTime) {
      const c = contacts.get(pattern);
      if (!c) continue;
      const n = c.found.length;
      if (s.air.length !== n) {
        s.air = new Float64Array(n);
        s.contact = new Float64Array(n);
        s.lastAir = new Float64Array(n);
      }
      for (let i = 0; i < n; i++) {
        if (c.found[i]) {
          if (s.contact[i] === 0) s.lastAir[i] = s.air[i];
          s.contact[i] += this.dt;
          s.air[i] = 0;
        } else {
          s.air[i] += this.dt;
          s.contact[i] = 0;
        }
      }
    }
  }

  private firstContact(s: AirTime, i: number): boolean {
    return s.contact[i] > 0 && s.contact[i] < this.dt + 1e-6;
  }

  /** ``torch.flip(current_contact_time)`` for the two-foot sensor. */
  private otherContactTime(s: AirTime, i: number): number {
    return s.contact[s.contact.length - 1 - i] ?? 0;
  }

  private stableStandWeight(
    t: AmpRewardTerm,
    f: Frame,
    torso: number,
    contact: ContactScan | undefined,
    magnitude: number,
  ): number {
    const p = (key: string, def: number) => {
      const v = t.params?.[key];
      return typeof v === "number" ? v : def;
    };
    if (!contact) return NaN;
    if (magnitude >= p("command_threshold", 0.05)) return 0;
    const thr = p("contact_force_threshold", 5.0);
    if (!Array.from(contact.force).every((x) => x > thr)) return 0;
    const vb = this.rootLinVelB(f);
    const w = this.bodyAngVelB(f, torso);
    const planar = Math.hypot(vb[0], vb[1]);
    const angular = Math.hypot(w[0], w[1]);
    return (
      Math.exp(-((planar / p("planar_speed_scale", 0.2)) ** 2)) *
      Math.exp(-((angular / p("torso_ang_vel_scale", 0.5)) ** 2))
    );
  }

  private turnFootHeight(
    _t: AmpRewardTerm,
    f: Frame,
    feet: number[],
    yaw: number,
    std: number,
    minYaw: number,
  ): number {
    if (!this.profiles) return NaN;
    const turning = f.mode === Mode.Turning && yaw >= minYaw;
    const prof: AmpTurnProfile = f.cmd[2] >= 0 ? this.profiles.left_turn : this.profiles.right_turn;
    if (!turning) return 0;
    this.turnPhase = (this.turnPhase + this.dt / prof.period_s) % 1.0;
    const nBins = prof.left.length;
    const idx = Math.min(Math.floor(this.turnPhase * nBins), nBins - 1);
    const zl = f.xpos[feet[0] * 3 + 2];
    const zr = f.xpos[feet[1] * 3 + 2];
    const support = Math.min(zl, zr);
    const err = (zl - support - prof.left[idx]) ** 2 + (zr - support - prof.right[idx]) ** 2;
    return Math.exp(-err / (std * std));
  }

  private jointDevSq(joints: string[], f: Frame): number {
    let s = 0;
    for (const jn of joints) {
      const i = this.jointNames.indexOf(jn);
      if (i < 0) continue;
      s += (f.qpos[this.qposAdr.get(jn)!] - this.defaults[i]) ** 2;
    }
    return s;
  }

  /** v_b = Rᵀ v_w for body ``b`` (xmat is row-major R). */
  private toBody(f: Frame, b: number, v: Vec3): Vec3 {
    const r = b * 9;
    const m = f.xmat;
    return [
      m[r] * v[0] + m[r + 3] * v[1] + m[r + 6] * v[2],
      m[r + 1] * v[0] + m[r + 4] * v[1] + m[r + 7] * v[2],
      m[r + 2] * v[0] + m[r + 5] * v[1] + m[r + 8] * v[2],
    ];
  }

  private gravityB(f: Frame, b: number): Vec3 {
    return this.toBody(f, b, [0, 0, -1]);
  }

  /** Mirror training ``is_slow_walk_env`` + path origin for play display. */
  private pathOffset(
    f: Frame,
    cmd: Float64Array,
    yawCmd: number,
  ): { lateral: number; forward: number } | null {
    const slow =
      Math.abs(cmd[1]) <= 1e-6 && yawCmd <= 1e-6 && cmd[0] >= 0.1 && cmd[0] <= 0.5;
    if (!slow) {
      this.pathActive = false;
      this.pathOrigin = null;
      return null;
    }
    const x = f.xpos[this.rootId * 3];
    const y = f.xpos[this.rootId * 3 + 1];
    // Root yaw from free-joint quat (wxyz at qpos[3:7] typically) — use xmat yaw.
    const r = this.rootId * 9;
    const yaw = Math.atan2(f.xmat[r + 3], f.xmat[r]);
    if (!this.pathActive || !this.pathOrigin) {
      this.pathOrigin = [x, y];
      this.pathYaw = yaw;
      this.pathActive = true;
    }
    const dx = x - this.pathOrigin[0];
    const dy = y - this.pathOrigin[1];
    const c = Math.cos(this.pathYaw);
    const s = Math.sin(this.pathYaw);
    return { forward: c * dx + s * dy, lateral: -s * dx + c * dy };
  }

  private rootLinVelB(f: Frame): Vec3 {
    const d = this.freeDofAdr;
    return this.toBody(f, this.rootId, [f.qvel[d], f.qvel[d + 1], f.qvel[d + 2]]);
  }

  /** Free-joint angular velocity is already expressed in the body frame. */
  private rootAngVelB(f: Frame): Vec3 {
    const d = this.freeDofAdr + 3;
    return [f.qvel[d], f.qvel[d + 1], f.qvel[d + 2]];
  }

  private bodyAngVelB(f: Frame, b: number): Vec3 {
    return this.toBody(f, b, [f.cvel[b * 6], f.cvel[b * 6 + 1], f.cvel[b * 6 + 2]]);
  }

  /** World-frame linear velocity of body origin from com-based cvel. */
  private bodyLinVelW(f: Frame, b: number): Vec3 {
    const root = Number(this.model.body_rootid[b]);
    const c = f.cvel;
    const r: Vec3 = [
      f.xpos[b * 3] - f.com[root * 3],
      f.xpos[b * 3 + 1] - f.com[root * 3 + 1],
      f.xpos[b * 3 + 2] - f.com[root * 3 + 2],
    ];
    const [wx, wy, wz] = [c[b * 6], c[b * 6 + 1], c[b * 6 + 2]];
    return [
      c[b * 6 + 3] + wy * r[2] - wz * r[1],
      c[b * 6 + 4] + wz * r[0] - wx * r[2],
      c[b * 6 + 5] + wx * r[1] - wy * r[0],
    ];
  }

  private footPosB(f: Frame, b: number): Vec3 {
    const r = this.rootId * 3;
    return this.toBody(f, this.rootId, [
      f.xpos[b * 3] - f.xpos[r],
      f.xpos[b * 3 + 1] - f.xpos[r + 1],
      f.xpos[b * 3 + 2] - f.xpos[r + 2],
    ]);
  }

  /** Yaw of the foot's +x axis in the pelvis frame. */
  private footHeadingErr(f: Frame, b: number): number {
    const m = f.xmat;
    const fwd = this.toBody(f, this.rootId, [m[b * 9], m[b * 9 + 3], m[b * 9 + 6]]);
    return Math.atan2(fwd[1], fwd[0]);
  }
}
