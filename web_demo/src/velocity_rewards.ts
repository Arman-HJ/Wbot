import type { MjData, MjModel } from "mujoco-js";
import { ContactGroups, objType } from "./contacts";
import { asArray, name2id, type MujocoModule } from "./mujoco_util";
import type { PolicyController } from "./policy";
import type { VelocityRewardConfig, VelocityRewardTerm } from "./types";

const MJ_JNT_FREE = 0;

/**
 * Raw (unweighted) values of mjlab velocity-task reward terms, evaluated on the
 * browser sim once per policy step. Flat terrain only: foot height = site z,
 * terrain contacts = contacts with world-body geoms.
 */
export class VelocityRewardEval {
  readonly terms: VelocityRewardTerm[];
  private readonly mj: MujocoModule;
  private readonly model: MjModel;
  private readonly policyDt: number;

  private readonly rootId: number;
  private readonly freeQposAdr: number;
  private readonly freeDofAdr: number;
  private readonly qposAdr: Int32Array;
  private readonly dofAdr: Int32Array;
  private readonly defaults: Float64Array;
  private readonly jointNames: string[];
  private readonly jntLo: Float64Array;
  private readonly jntHi: Float64Array;

  private readonly bodyIds = new Map<string, number>();
  private readonly siteIds = new Map<string, number>();
  private readonly groups: ContactGroups;
  private readonly poseStd = new Map<string, Float64Array[]>();

  private prevAction: Float64Array;
  private prevContact = new Map<string, Uint8Array>();
  private readonly out: Float64Array;

  constructor(
    mj: MujocoModule,
    model: MjModel,
    cfg: VelocityRewardConfig,
    policy: PolicyController,
  ) {
    this.mj = mj;
    this.model = model;
    this.terms = cfg.terms;
    this.policyDt = policy.manifest.policy_dt;
    this.groups = new ContactGroups(mj, model);
    this.out = new Float64Array(cfg.terms.length);

    this.rootId = name2id(mj, model, "body", cfg.root_body || "pelvis");
    let freeQ = 0;
    let freeD = 0;
    for (let j = 0; j < model.njnt; j++) {
      if (
        Number(model.jnt_type[j]) === MJ_JNT_FREE &&
        Number(model.jnt_bodyid[j]) === this.rootId
      ) {
        freeQ = Number(model.jnt_qposadr[j]);
        freeD = Number(model.jnt_dofadr[j]);
        break;
      }
    }
    this.freeQposAdr = freeQ;
    this.freeDofAdr = freeD;

    this.jointNames = policy.jointNames;
    const n = this.jointNames.length;
    this.qposAdr = new Int32Array(n);
    this.dofAdr = new Int32Array(n);
    this.jntLo = new Float64Array(n);
    this.jntHi = new Float64Array(n);
    this.defaults = Float64Array.from(policy.defaultPos);
    for (let i = 0; i < n; i++) {
      const jid = name2id(mj, model, "joint", this.jointNames[i]);
      this.qposAdr[i] = Number(model.jnt_qposadr[jid]);
      this.dofAdr[i] = Number(model.jnt_dofadr[jid]);
      this.jntLo[i] = Number(model.jnt_range[jid * 2]);
      this.jntHi[i] = Number(model.jnt_range[jid * 2 + 1]);
    }
    this.prevAction = new Float64Array(n);

    for (const t of this.terms) {
      if (t.body && !this.bodyIds.has(t.body)) {
        this.bodyIds.set(t.body, name2id(mj, model, "body", t.body));
      }
      for (const s of t.site_names || []) {
        if (!this.siteIds.has(s)) this.siteIds.set(s, this.siteId(s));
      }
      if (t.contact) this.groups.add(t.contact.pattern, t.contact.mode);
      if (t.kind === "variable_posture") {
        const resolve = (m?: Record<string, number>) =>
          Float64Array.from(this.jointNames, (jn) => m?.[jn] ?? 1.0);
        this.poseStd.set(t.name, [
          resolve(t.std_standing),
          resolve(t.std_walking),
          resolve(t.std_running),
        ]);
      }
    }
  }

  reset() {
    this.prevAction.fill(0);
    this.prevContact.clear();
  }

  /** Raw value per term (NaN = not evaluated in browser). Call after ``policy.step``. */
  evaluate(policy: PolicyController, data: MjData): Float64Array {
    const model = this.model;
    const cmd = policy.getCommand();
    const speed = Math.hypot(cmd[0], cmd[1]) + Math.abs(cmd[2]);
    const action = policy.getLastAction();
    const qpos = asArray(data.qpos, model.nq);
    const qvel = asArray(data.qvel, model.nv);
    const xmat = asArray(data.xmat, model.nbody * 9);
    const contacts = this.groups.scan(data);
    let subtreeVelDone = false;

    for (let k = 0; k < this.terms.length; k++) {
      const t = this.terms[k];
      const active = speed > (t.command_threshold ?? 0.1) ? 1 : 0;
      let v = NaN;
      switch (t.kind) {
        case "track_lin_vel": {
          const vb = this.rootLinVelB(qvel, xmat);
          const err = (cmd[0] - vb[0]) ** 2 + (cmd[1] - vb[1]) ** 2 + vb[2] ** 2;
          v = Math.exp(-err / (t.std ?? 0.5) ** 2);
          break;
        }
        case "track_ang_vel": {
          const d = this.freeDofAdr + 3;
          const err = (cmd[2] - qvel[d + 2]) ** 2 + qvel[d] ** 2 + qvel[d + 1] ** 2;
          v = Math.exp(-err / (t.std ?? 0.5) ** 2);
          break;
        }
        case "body_orientation": {
          const b = this.bodyIds.get(t.body!) ?? this.rootId;
          v = xmat[b * 9 + 6] ** 2 + xmat[b * 9 + 7] ** 2;
          break;
        }
        case "variable_posture": {
          const [sStand, sWalk, sRun] = this.poseStd.get(t.name)!;
          const std =
            speed < (t.walking_threshold ?? 0.1)
              ? sStand
              : speed >= (t.running_threshold ?? 2.0)
                ? sRun
                : sWalk;
          let acc = 0;
          for (let i = 0; i < this.qposAdr.length; i++) {
            const e = qpos[this.qposAdr[i]] - this.defaults[i];
            acc += (e * e) / (std[i] * std[i]);
          }
          v = Math.exp(-acc / Math.max(1, this.qposAdr.length));
          break;
        }
        case "body_ang_vel_xy": {
          const b = this.bodyIds.get(t.body!) ?? this.rootId;
          const cvel = asArray(data.cvel, model.nbody * 6);
          v = cvel[b * 6] ** 2 + cvel[b * 6 + 1] ** 2;
          break;
        }
        case "angular_momentum": {
          if (!subtreeVelDone) {
            this.mj.mj_subtreeVel(model, data);
            subtreeVelDone = true;
          }
          const b = this.bodyIds.get(t.body!) ?? this.rootId;
          const am = asArray(data.subtree_angmom, model.nbody * 3);
          v = am[b * 3] ** 2 + am[b * 3 + 1] ** 2 + am[b * 3 + 2] ** 2;
          break;
        }
        case "joint_acc": {
          const qacc = asArray(data.qacc, model.nv);
          let s = 0;
          for (let i = 0; i < this.dofAdr.length; i++) s += qacc[this.dofAdr[i]] ** 2;
          v = s;
          break;
        }
        case "joint_pos_limits": {
          const f = t.soft_factor ?? 1.0;
          let s = 0;
          for (let i = 0; i < this.qposAdr.length; i++) {
            const mid = 0.5 * (this.jntLo[i] + this.jntHi[i]);
            const half = 0.5 * (this.jntHi[i] - this.jntLo[i]) * f;
            const q = qpos[this.qposAdr[i]];
            s += Math.max(0, mid - half - q) + Math.max(0, q - (mid + half));
          }
          v = s;
          break;
        }
        case "action_rate": {
          let s = 0;
          for (let i = 0; i < action.length; i++) s += (action[i] - this.prevAction[i]) ** 2;
          v = s;
          break;
        }
        case "feet_gait": {
          const c = contacts.get(t.contact?.pattern ?? "");
          if (!c) break;
          const offs = t.offset ?? [0, 0.5];
          const phase = (policy.getEpisodeSteps() * this.policyDt) / (t.period ?? 0.6);
          let match = 0;
          for (let f = 0; f < c.found.length; f++) {
            const legPhase = (phase + (offs[f] ?? 0)) % 1.0;
            const stance = legPhase < (t.threshold ?? 0.5);
            if (stance === (c.found[f] > 0)) match += 1;
          }
          v = (c.found.length ? match / c.found.length : 0) * active;
          break;
        }
        case "feet_clearance": {
          const target = t.target_height ?? 0.1;
          let s = 0;
          for (const name of t.site_names || []) {
            const sid = this.siteIds.get(name) ?? -1;
            if (sid < 0) continue;
            const z = asArray(data.site_xpos, model.nsite * 3)[sid * 3 + 2];
            const [vx, vy] = this.siteLinVelW(data, sid);
            s += Math.abs(z - target) * Math.hypot(vx, vy);
          }
          v = s * active;
          break;
        }
        case "feet_slip": {
          const c = contacts.get(t.contact?.pattern ?? "");
          if (!c) break;
          let s = 0;
          (t.site_names || []).forEach((name, f) => {
            const sid = this.siteIds.get(name) ?? -1;
            if (sid < 0 || !(c.found[f] > 0)) return;
            const [vx, vy] = this.siteLinVelW(data, sid);
            s += vx * vx + vy * vy;
          });
          v = s * active;
          break;
        }
        case "soft_landing": {
          const pattern = t.contact?.pattern ?? "";
          const c = contacts.get(pattern);
          if (!c) break;
          const prev = this.prevContact.get(pattern);
          let s = 0;
          for (let f = 0; f < c.found.length; f++) {
            if (c.found[f] > 0 && !(prev?.[f] ?? 0)) s += c.force[f];
          }
          v = s * active;
          break;
        }
        case "stand_still": {
          let s = 0;
          for (let i = 0; i < this.qposAdr.length; i++) {
            s += (qpos[this.qposAdr[i]] - this.defaults[i]) ** 2;
          }
          v = speed <= (t.command_threshold ?? 0.1) ? s : 0;
          break;
        }
        case "self_collision": {
          const c = contacts.get(t.contact?.pattern ?? "");
          if (!c) break;
          // Training counts substeps over force history; approximate with current state.
          const hit = c.selfMaxForce > (t.force_threshold ?? 10.0);
          v = hit ? Math.max(1, t.contact?.history_length ?? 1) : 0;
          break;
        }
        default:
          v = NaN;
      }
      this.out[k] = v;
    }

    this.prevAction.set(action.subarray(0, this.prevAction.length));
    this.prevContact.clear();
    for (const [pattern, c] of contacts) this.prevContact.set(pattern, c.found);
    return this.out;
  }

  private rootLinVelB(qvel: Float64Array, xmat: Float64Array): [number, number, number] {
    const d = this.freeDofAdr;
    const vx = qvel[d];
    const vy = qvel[d + 1];
    const vz = qvel[d + 2];
    const r = this.rootId * 9;
    return [
      xmat[r] * vx + xmat[r + 3] * vy + xmat[r + 6] * vz,
      xmat[r + 1] * vx + xmat[r + 4] * vy + xmat[r + 7] * vz,
      xmat[r + 2] * vx + xmat[r + 5] * vy + xmat[r + 8] * vz,
    ];
  }

  /** World-frame linear velocity of a site from its body's com-based cvel. */
  private siteLinVelW(data: MjData, sid: number): [number, number, number] {
    const model = this.model;
    const b = Number(model.site_bodyid[sid]);
    const root = Number(model.body_rootid[b]);
    const cvel = asArray(data.cvel, model.nbody * 6);
    const com = asArray(data.subtree_com, model.nbody * 3);
    const p = asArray(data.site_xpos, model.nsite * 3);
    const wx = cvel[b * 6];
    const wy = cvel[b * 6 + 1];
    const wz = cvel[b * 6 + 2];
    const rx = p[sid * 3] - com[root * 3];
    const ry = p[sid * 3 + 1] - com[root * 3 + 1];
    const rz = p[sid * 3 + 2] - com[root * 3 + 2];
    return [
      cvel[b * 6 + 3] + wy * rz - wz * ry,
      cvel[b * 6 + 4] + wz * rx - wx * rz,
      cvel[b * 6 + 5] + wx * ry - wy * rx,
    ];
  }

  private siteId(name: string): number {
    return this.mj.mj_name2id(this.model, objType(this.mj, "mjOBJ_SITE"), name);
  }
}
