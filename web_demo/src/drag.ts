import * as THREE from "three";
import type { MjData, MjModel } from "mujoco-js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { asArray } from "./mujoco_util";
import { cross } from "./math";

type DragState = {
  bodyId: number;
  grabLocal: THREE.Vector3;
  target: THREE.Vector3;
  mass: number;
};

type MatBackup = {
  mat: THREE.MeshStandardMaterial;
  emissive: THREE.Color;
  emissiveIntensity: number;
};

const HIGHLIGHT = new THREE.Color(0xd4a574);

/**
 * Ctrl + left-drag body → spring wrench into xfrc_applied (MuJoCo-like).
 * Ctrl+hover / active drag highlights the picked body meshes.
 */
export class BodyDragController {
  enabled = true;
  gain = 1.0;
  private state: DragState | null = null;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  private hitPoint = new THREE.Vector3();
  private forceLine: THREE.Line;
  private model!: MjModel;
  private bodyMeshes: THREE.Object3D[] = [];
  private camera: THREE.Camera;
  private controls: OrbitControls;
  private canvas: HTMLCanvasElement;
  private lastBodyId = -1;
  private hoverBodyId = -1;
  private highlightBackups: MatBackup[] = [];

  constructor(
    canvas: HTMLCanvasElement,
    camera: THREE.Camera,
    controls: OrbitControls,
    scene: THREE.Scene,
  ) {
    this.canvas = canvas;
    this.camera = camera;
    this.controls = controls;

    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(),
    ]);
    this.forceLine = new THREE.Line(
      geo,
      new THREE.LineBasicMaterial({ color: 0xff7a18 }),
    );
    this.forceLine.visible = false;
    scene.add(this.forceLine);

    // Capture phase so we win over OrbitControls when Ctrl+pick hits a body.
    canvas.addEventListener("pointerdown", (e) => this.onDown(e), true);
    window.addEventListener("pointermove", (e) => this.onMove(e));
    window.addEventListener("pointerup", (e) => this.onUp(e));
    canvas.addEventListener("pointerleave", () => {
      if (!this.state) this.setHighlightBody(-1);
    });
  }

  bind(model: MjModel, bodyMeshes: THREE.Object3D[]) {
    this.clearHighlight();
    this.model = model;
    this.bodyMeshes = bodyMeshes;
    this.clearWrench();
    this.state = null;
    this.hoverBodyId = -1;
    this.forceLine.visible = false;
    this.controls.enabled = true;
  }

  private setPointer(e: PointerEvent) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  private pickBodyId(): number {
    for (const obj of this.bodyMeshes) {
      obj.updateWorldMatrix(true, false);
    }
    const hits = this.raycaster.intersectObjects(this.bodyMeshes, false);
    if (!hits.length) return -1;
    const bodyId = Number(hits[0].object.userData.bodyId);
    return bodyId > 0 ? bodyId : -1;
  }

  private clearHighlight() {
    for (const b of this.highlightBackups) {
      b.mat.emissive.copy(b.emissive);
      b.mat.emissiveIntensity = b.emissiveIntensity;
    }
    this.highlightBackups = [];
    this.hoverBodyId = -1;
  }

  private setHighlightBody(bodyId: number) {
    if (bodyId === this.hoverBodyId) return;
    this.clearHighlight();
    if (!(bodyId > 0)) return;
    this.hoverBodyId = bodyId;
    for (const obj of this.bodyMeshes) {
      if (Number(obj.userData.bodyId) !== bodyId) continue;
      const mesh = obj as THREE.Mesh;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const m of mats) {
        if (!(m instanceof THREE.MeshStandardMaterial)) continue;
        this.highlightBackups.push({
          mat: m,
          emissive: m.emissive.clone(),
          emissiveIntensity: m.emissiveIntensity,
        });
        m.emissive.copy(HIGHLIGHT);
        m.emissiveIntensity = 0.28;
      }
    }
  }

  private onDown(e: PointerEvent) {
    if (!this.enabled || !this.model || !(e.ctrlKey || e.metaKey) || e.button !== 0) {
      return;
    }
    this.setPointer(e);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const bodyId = this.pickBodyId();
    if (!(bodyId > 0)) return;

    const hits = this.raycaster.intersectObjects(this.bodyMeshes, false);
    const hit = hits[0];
    e.preventDefault();
    e.stopPropagation();
    this.controls.enabled = false;

    // Grab point in body frame. Mesh may be parented under a body Group
    // (geom_pos/quat local); never use mesh.position as body origin.
    const grabWorld = hit.point.clone();
    const bodyWorldPos = new THREE.Vector3();
    const bodyWorldQuat = new THREE.Quaternion();
    const bodyObj = hit.object.parent ?? hit.object;
    bodyObj.updateWorldMatrix(true, false);
    bodyObj.matrixWorld.decompose(
      bodyWorldPos,
      bodyWorldQuat,
      new THREE.Vector3(),
    );
    const grabLocal = grabWorld
      .clone()
      .sub(bodyWorldPos)
      .applyQuaternion(bodyWorldQuat.clone().invert());

    const mass = Math.max(0.1, Number(this.model.body_mass[bodyId] ?? 1.0));
    this.state = {
      bodyId,
      grabLocal,
      target: grabWorld.clone(),
      mass,
    };
    this.lastBodyId = bodyId;
    this.setHighlightBody(bodyId);
    this.plane.setFromNormalAndCoplanarPoint(
      this.camera.getWorldDirection(new THREE.Vector3()).normalize(),
      grabWorld,
    );
    this.forceLine.visible = true;
  }

  private onMove(e: PointerEvent) {
    this.setPointer(e);
    this.raycaster.setFromCamera(this.pointer, this.camera);

    if (this.state) {
      if (this.raycaster.ray.intersectPlane(this.plane, this.hitPoint)) {
        this.state.target.copy(this.hitPoint);
      }
      return;
    }

    // Ctrl/Meta hover preview highlight.
    if (e.ctrlKey || e.metaKey) {
      this.setHighlightBody(this.pickBodyId());
    } else if (this.hoverBodyId >= 0) {
      this.setHighlightBody(-1);
    }
  }

  private onUp(_e: PointerEvent) {
    if (!this.state) return;
    this.clearWrench();
    this.state = null;
    this.controls.enabled = true;
    this.forceLine.visible = false;
    this.setHighlightBody(-1);
  }

  private clearWrench() {
    if (!this.model) return;
    // Cleared each apply() call when inactive; keep lastBodyId for one-shot clear.
  }

  /** Write spring wrench into xfrc_applied each physics tick. */
  apply(data: MjData) {
    if (!this.model) return;
    const nbody = this.model.nbody;

    if (!this.state) {
      if (this.lastBodyId >= 0) {
        for (let i = 0; i < 6; i++) data.xfrc_applied[this.lastBodyId * 6 + i] = 0;
        this.lastBodyId = -1;
      }
      return;
    }

    const bid = this.state.bodyId;
    const xpos = asArray(data.xpos, nbody * 3);
    const xquat = asArray(data.xquat, nbody * 4);
    const bodyPos = new THREE.Vector3(xpos[bid * 3], xpos[bid * 3 + 1], xpos[bid * 3 + 2]);
    const bodyQuat = new THREE.Quaternion(
      xquat[bid * 4 + 1],
      xquat[bid * 4 + 2],
      xquat[bid * 4 + 3],
      xquat[bid * 4 + 0],
    );
    const grabWorld = this.state.grabLocal.clone().applyQuaternion(bodyQuat).add(bodyPos);
    const forceVec = this.state.target.clone().sub(grabWorld);
    const k = 20.0 * this.state.mass * this.gain;
    forceVec.multiplyScalar(k);
    const lever = grabWorld.clone().sub(bodyPos);
    const torque = cross(
      [lever.x, lever.y, lever.z],
      [forceVec.x, forceVec.y, forceVec.z],
    );

    for (let b = 0; b < nbody; b++) {
      for (let i = 0; i < 6; i++) data.xfrc_applied[b * 6 + i] = 0;
    }
    data.xfrc_applied[bid * 6 + 0] = forceVec.x;
    data.xfrc_applied[bid * 6 + 1] = forceVec.y;
    data.xfrc_applied[bid * 6 + 2] = forceVec.z;
    data.xfrc_applied[bid * 6 + 3] = torque[0];
    data.xfrc_applied[bid * 6 + 4] = torque[1];
    data.xfrc_applied[bid * 6 + 5] = torque[2];

    const posAttr = this.forceLine.geometry.attributes.position as THREE.BufferAttribute;
    posAttr.setXYZ(0, grabWorld.x, grabWorld.y, grabWorld.z);
    posAttr.setXYZ(1, this.state.target.x, this.state.target.y, this.state.target.z);
    posAttr.needsUpdate = true;
    this.lastBodyId = bid;
  }
}
