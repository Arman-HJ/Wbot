import * as THREE from "three";
import type { MjData, MjModel } from "mujoco-js";
import { asArray, name2id, type MujocoModule } from "./mujoco_util";
import type { PolicyController } from "./policy";
import type { MotionVisMode } from "./types";

/** Bodies to draw as axes in frames mode (Viser EE + torso). */
const FRAME_BODY_NAMES = [
  "torso_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_elbow_link",
  "right_elbow_link",
] as const;

/**
 * Optional motion reference overlay (Viser "Show motion" / ghost|frames).
 */
export class MotionVisualizer {
  private enabled = false;
  private mode: MotionVisMode = "ghost";
  private ghostRoot = new THREE.Group();
  private ghostBodyGroups = new Map<number, THREE.Group>();
  private ghostData: MjData | null = null;
  private framesRoot = new THREE.Group();
  private desiredAxes: THREE.AxesHelper[] = [];
  private currentAxes: THREE.AxesHelper[] = [];
  private frameBodyIds: number[] = [];
  private model: MjModel | null = null;
  private mj: MujocoModule | null = null;

  constructor(private readonly scene: THREE.Scene) {
    this.ghostRoot.visible = false;
    this.framesRoot.visible = false;
    this.scene.add(this.ghostRoot);
    this.scene.add(this.framesRoot);
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  getMode(): MotionVisMode {
    return this.mode;
  }

  setEnabled(on: boolean) {
    this.enabled = on;
    this.applyVisibility();
  }

  setMode(mode: MotionVisMode) {
    this.mode = mode;
    this.applyVisibility();
  }

  private applyVisibility() {
    const show = this.enabled;
    this.ghostRoot.visible = show && this.mode === "ghost";
    this.framesRoot.visible = show && this.mode === "frames";
  }

  /** Rebuild ghost meshes from the live robot geom layout (translucent clone). */
  rebuild(mj: MujocoModule, model: MjModel, robotBodyGroups: Map<number, THREE.Group>) {
    this.disposeGhostMeshes();
    this.disposeFrames();
    this.mj = mj;
    this.model = model;
    this.ghostData?.delete?.();
    this.ghostData = new mj.MjData(model);

    for (const [bodyId, srcGroup] of robotBodyGroups) {
      const g = new THREE.Group();
      g.matrixAutoUpdate = false;
      for (const child of srcGroup.children) {
        if (!(child instanceof THREE.Mesh)) continue;
        const geom = child.geometry;
        const mat = new THREE.MeshStandardMaterial({
          color: 0x6ec8ff,
          metalness: 0.05,
          roughness: 0.55,
          transparent: true,
          opacity: 0.38,
          depthWrite: false,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.copy(child.position);
        mesh.quaternion.copy(child.quaternion);
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        g.add(mesh);
      }
      this.ghostBodyGroups.set(bodyId, g);
      this.ghostRoot.add(g);
    }

    this.frameBodyIds = [];
    for (const name of FRAME_BODY_NAMES) {
      const id = name2id(mj, model, "body", name);
      if (id < 0) continue;
      this.frameBodyIds.push(id);
      const desired = new THREE.AxesHelper(0.08);
      const current = new THREE.AxesHelper(0.12);
      // Tint current axes slightly via material override isn't built-in; leave default.
      this.desiredAxes.push(desired);
      this.currentAxes.push(current);
      this.framesRoot.add(desired);
      this.framesRoot.add(current);
    }
    this.applyVisibility();
  }

  sync(policy: PolicyController, robotData: MjData) {
    if (!this.enabled || !this.mj || !this.model || !policy.hasMotion()) return;
    if (this.mode === "ghost") {
      this.syncGhost(policy, robotData);
    } else {
      this.syncFrames(policy, robotData);
    }
  }

  private syncGhost(policy: PolicyController, robotData: MjData) {
    if (!this.ghostData || !this.mj || !this.model) return;
    if (!policy.writeGhostQpos(this.model, robotData, this.ghostData)) return;
    this.mj.mj_forward(this.model, this.ghostData);
    this.syncBodyGroups(this.ghostData, this.ghostBodyGroups);
  }

  private syncFrames(policy: PolicyController, robotData: MjData) {
    if (!this.ghostData || !this.mj || !this.model) return;
    if (!policy.writeGhostQpos(this.model, robotData, this.ghostData)) return;
    this.mj.mj_forward(this.model, this.ghostData);

    const nbody = this.model.nbody;
    const dPos = asArray(this.ghostData.xpos, nbody * 3);
    const dMat = asArray(this.ghostData.xmat, nbody * 9);
    const cPos = asArray(robotData.xpos, nbody * 3);
    const cMat = asArray(robotData.xmat, nbody * 9);

    for (let i = 0; i < this.frameBodyIds.length; i++) {
      const bid = this.frameBodyIds[i];
      this.setAxesFromBody(this.desiredAxes[i], dPos, dMat, bid);
      this.setAxesFromBody(this.currentAxes[i], cPos, cMat, bid);
    }
  }

  private setAxesFromBody(
    axes: THREE.AxesHelper,
    xpos: Float64Array,
    xmat: Float64Array,
    bodyId: number,
  ) {
    const p = bodyId * 3;
    const o = bodyId * 9;
    axes.position.set(xpos[p], xpos[p + 1], xpos[p + 2]);
    const te = axes.matrix.elements;
    // xmat is row-major 3x3; Three matrix is column-major.
    te[0] = xmat[o + 0];
    te[1] = xmat[o + 3];
    te[2] = xmat[o + 6];
    te[3] = 0;
    te[4] = xmat[o + 1];
    te[5] = xmat[o + 4];
    te[6] = xmat[o + 7];
    te[7] = 0;
    te[8] = xmat[o + 2];
    te[9] = xmat[o + 5];
    te[10] = xmat[o + 8];
    te[11] = 0;
    te[12] = xpos[p];
    te[13] = xpos[p + 1];
    te[14] = xpos[p + 2];
    te[15] = 1;
    axes.matrixAutoUpdate = false;
    axes.matrixWorldNeedsUpdate = true;
  }

  private syncBodyGroups(data: MjData, groups: Map<number, THREE.Group>) {
    if (!this.model) return;
    const nbody = this.model.nbody;
    const xpos = asArray(data.xpos, nbody * 3);
    const xmat = asArray(data.xmat, nbody * 9);
    for (const [bodyId, group] of groups) {
      const o = bodyId * 9;
      const p = bodyId * 3;
      const te = group.matrix.elements;
      te[0] = xmat[o + 0];
      te[1] = xmat[o + 3];
      te[2] = xmat[o + 6];
      te[3] = 0;
      te[4] = xmat[o + 1];
      te[5] = xmat[o + 4];
      te[6] = xmat[o + 7];
      te[7] = 0;
      te[8] = xmat[o + 2];
      te[9] = xmat[o + 5];
      te[10] = xmat[o + 8];
      te[11] = 0;
      te[12] = xpos[p + 0];
      te[13] = xpos[p + 1];
      te[14] = xpos[p + 2];
      te[15] = 1;
      group.matrixWorldNeedsUpdate = true;
    }
    this.ghostRoot.updateMatrixWorld(true);
  }

  private disposeGhostMeshes() {
    for (const g of this.ghostBodyGroups.values()) {
      for (const child of [...g.children]) {
        if (child instanceof THREE.Mesh) {
          (child.material as THREE.Material).dispose();
        }
        g.remove(child);
      }
      this.ghostRoot.remove(g);
    }
    this.ghostBodyGroups.clear();
  }

  private disposeFrames() {
    for (const a of [...this.desiredAxes, ...this.currentAxes]) {
      this.framesRoot.remove(a);
      a.geometry.dispose();
      const mat = a.material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else mat.dispose();
    }
    this.desiredAxes = [];
    this.currentAxes = [];
    this.frameBodyIds = [];
  }

  dispose() {
    this.disposeGhostMeshes();
    this.disposeFrames();
    this.ghostData?.delete?.();
    this.ghostData = null;
  }
}
