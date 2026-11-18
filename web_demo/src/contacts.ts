import type { MjData, MjModel } from "mujoco-js";
import { asArray, type MujocoModule } from "./mujoco_util";

const MJ_CONE_ELLIPTIC = 1;

export type ContactMode = "subtree" | "body";

export type ContactScan = {
  /** Primary touches a world-body geom (terrain). */
  found: Uint8Array;
  /** Summed normal force against the world, per primary. */
  force: Float64Array;
  /** Largest contact force between two geoms of the same group (self collision). */
  selfMaxForce: number;
};

type ContactGroup = {
  /** geom id → primary index (or -1). */
  geomPrimary: Int32Array;
  count: number;
};

export function objType(mj: MujocoModule, key: "mjOBJ_BODY" | "mjOBJ_SITE"): number {
  const v = (mj.mjtObj as any)[key];
  return typeof v === "number" ? v : v.value;
}

/**
 * Browser stand-in for mjlab ContactSensor groups (keyed by primary pattern).
 * Flat terrain only: terrain contacts = contacts with world-body geoms.
 */
export class ContactGroups {
  private readonly groups = new Map<string, ContactGroup>();
  private readonly cone: number;

  constructor(
    private readonly mj: MujocoModule,
    private readonly model: MjModel,
  ) {
    this.cone = Number(model.opt.cone);
  }

  add(pattern: string, mode: ContactMode) {
    if (!this.groups.has(pattern)) this.groups.set(pattern, this.build(pattern, mode));
  }

  scan(data: MjData): Map<string, ContactScan> {
    const model = this.model;
    const res = new Map<string, ContactScan>();
    for (const [pattern, g] of this.groups) {
      res.set(pattern, {
        found: new Uint8Array(g.count),
        force: new Float64Array(g.count),
        selfMaxForce: 0,
      });
    }
    const ncon = data.ncon;
    if (!ncon || !this.groups.size) return res;
    const efc = asArray(data.efc_force, data.nefc);
    const vec = data.contact;
    try {
      for (let i = 0; i < ncon; i++) {
        const c = vec.get(i);
        if (!c) continue;
        const g1 = c.geom1;
        const g2 = c.geom2;
        const adr = c.efc_address;
        const dim = c.dim;
        c.delete();
        if (adr < 0) continue;
        let fn = efc[adr];
        if (dim > 1 && this.cone !== MJ_CONE_ELLIPTIC) {
          fn = 0;
          for (let k = 0; k < 2 * (dim - 1); k++) fn += efc[adr + k];
        }
        const b1 = Number(model.geom_bodyid[g1]);
        const b2 = Number(model.geom_bodyid[g2]);
        for (const [pattern, g] of this.groups) {
          const r = res.get(pattern)!;
          const p1 = g.geomPrimary[g1];
          const p2 = g.geomPrimary[g2];
          if (p1 >= 0 && p2 >= 0) {
            r.selfMaxForce = Math.max(r.selfMaxForce, fn);
          } else if (p1 >= 0 && b2 === 0) {
            r.found[p1] = 1;
            r.force[p1] += fn;
          } else if (p2 >= 0 && b1 === 0) {
            r.found[p2] = 1;
            r.force[p2] += fn;
          }
        }
      }
    } finally {
      vec.delete();
    }
    return res;
  }

  private build(pattern: string, mode: ContactMode): ContactGroup {
    const model = this.model;
    const re = new RegExp(pattern.startsWith("^") ? pattern : `^(?:${pattern})$`);
    const bodyObj = objType(this.mj, "mjOBJ_BODY");
    const primaries: number[] = [];
    for (let b = 1; b < model.nbody; b++) {
      if (re.test(this.mj.mj_id2name(model, bodyObj, b) || "")) primaries.push(b);
    }
    const bodyPrimary = new Int32Array(model.nbody).fill(-1);
    for (let b = 1; b < model.nbody; b++) {
      let cur = b;
      while (cur > 0) {
        const idx = primaries.indexOf(cur);
        if (idx >= 0) {
          bodyPrimary[b] = idx;
          break;
        }
        if (mode !== "subtree") break;
        cur = Number(model.body_parentid[cur]);
      }
    }
    const geomPrimary = new Int32Array(model.ngeom);
    for (let g = 0; g < model.ngeom; g++) {
      geomPrimary[g] = bodyPrimary[Number(model.geom_bodyid[g])];
    }
    return { geomPrimary, count: primaries.length };
  }
}
