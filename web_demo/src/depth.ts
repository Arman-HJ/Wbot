import * as THREE from "three";
import type { MjData, MjModel } from "mujoco-js";
import { asArray, name2id, type MujocoModule } from "./mujoco_util";
import type { DepthMeta } from "./types";

/** Training defaults: ``robot_wbot_neck.HEAD_CAM_*`` / CameraSensorCfg zed2i_depth. */
const DEFAULT_CAM_POS: [number, number, number] = [0.06, 0.0, 0.04];
const DEFAULT_CAM_QUAT: [number, number, number, number] = [0.5, 0.5, -0.5, -0.5];

/**
 * Head-mounted ZED2i depth for DAVIS, aligned with training:
 * - parent ``head_camera_link`` (follows neck yaw/pitch)
 * - local ``cam_pos`` / ``cam_quat_wxyz`` (not body origin)
 * - fovy 60°, render 168×94 → crop 168×80, normalize like ``align_depth``
 *
 * Rays are cast analytically at floor + ball + goal (browser has no Warp depth).
 */
export class DavisDepthCamera {
  private readonly scene: THREE.Scene;
  private readonly cam: THREE.PerspectiveCamera;
  private readonly depthBuf: Float32Array;
  private readonly out: Float32Array;
  private readonly meta: DepthMeta;
  private headBodyId = -1;
  private camId = -1;
  private ballBodyId = -1;
  private goalBodyId = -1;
  private _headName = "head_camera_link";
  private readonly camName = "zed2i_depth";
  private readonly renderH: number;
  private readonly renderW: number;
  private readonly cropH: number;
  private readonly cropW: number;
  private readonly depthMin: number;
  private readonly depthMax: number;
  private readonly camPos: [number, number, number];
  private readonly camQuat: [number, number, number, number];
  private frustum: THREE.LineSegments | null = null;
  private frustumVisible = true;
  private readonly ballRadius = 0.11;
  private readonly postRadius = 0.05;
  private readonly postHalfH = 0.9;
  private readonly postY = 1.2;
  private readonly crossbarZ = 1.8;
  private readonly crossbarHalf = 1.2;

  // Scratch for head→world and cam axes.
  private readonly Rhead = new Float64Array(9);
  private readonly Rcam = new Float64Array(9); // world columns: right, up, -look? see capture
  private readonly origin = new Float64Array(3);
  private readonly look = new Float64Array(3);
  private readonly up = new Float64Array(3);
  private readonly right = new Float64Array(3);

  constructor(
    scene: THREE.Scene,
    _gl: THREE.WebGLRenderer,
    meta: DepthMeta | undefined,
  ) {
    this.scene = scene;
    this.meta = meta || {};
    this.renderH = Math.max(1, Number(this.meta.render_height ?? 94));
    this.renderW = Math.max(1, Number(this.meta.render_width ?? 168));
    this.cropH = Math.max(1, Number(this.meta.height ?? 80));
    this.cropW = Math.max(1, Number(this.meta.width ?? 168));
    this.depthMin = Number(this.meta.depth_min_m ?? 0.3);
    this.depthMax = Number(this.meta.depth_max_m ?? 5.0);
    const fovy = Number(this.meta.fovy_deg ?? 60);
    const cp = this.meta.cam_pos;
    this.camPos = [
      Number(cp?.[0] ?? DEFAULT_CAM_POS[0]),
      Number(cp?.[1] ?? DEFAULT_CAM_POS[1]),
      Number(cp?.[2] ?? DEFAULT_CAM_POS[2]),
    ];
    const cq = this.meta.cam_quat_wxyz;
    this.camQuat = [
      Number(cq?.[0] ?? DEFAULT_CAM_QUAT[0]),
      Number(cq?.[1] ?? DEFAULT_CAM_QUAT[1]),
      Number(cq?.[2] ?? DEFAULT_CAM_QUAT[2]),
      Number(cq?.[3] ?? DEFAULT_CAM_QUAT[3]),
    ];

    this.cam = new THREE.PerspectiveCamera(
      fovy,
      this.renderW / this.renderH,
      Math.max(0.05, this.depthMin * 0.5),
      this.depthMax * 1.2,
    );
    this.cam.up.set(0, 0, 1);
    this.depthBuf = new Float32Array(this.renderW * this.renderH);
    this.out = new Float32Array(this.cropH * this.cropW);
  }

  bindHead(mj: MujocoModule, model: MjModel, bodyName = "head_camera_link") {
    this._headName = this.meta.head_body || bodyName;
    this.headBodyId = name2id(mj, model, "body", this._headName);
    this.camId = name2id(mj, model, "camera", this.camName);
  }

  setHeadBodyId(id: number) {
    this.headBodyId = id;
  }

  bindSceneBodies(mj: MujocoModule, model: MjModel) {
    if (this.headBodyId < 0) {
      this.headBodyId = name2id(mj, model, "body", this.getHeadName());
    }
    if (this.camId < 0) {
      this.camId = name2id(mj, model, "camera", this.camName);
    }
    this.ballBodyId = name2id(mj, model, "body", "ball");
    this.goalBodyId = name2id(mj, model, "body", "goal");
  }

  getHeadName(): string {
    return this._headName || this.meta.head_body || "head_camera_link";
  }

  getOutput(): Float32Array {
    return this.out;
  }

  getShape(): [number, number] {
    return [this.cropH, this.cropW];
  }

  getExtrinsics(): { pos: number[]; quat: number[]; fovy: number; parent: string } {
    return {
      pos: [...this.camPos],
      quat: [...this.camQuat],
      fovy: this.cam.fov,
      parent: this.getHeadName(),
    };
  }

  /** Last world optical pose (after capture / updateOpticalPose). */
  getWorldPose(): {
    origin: number[];
    look: number[];
    up: number[];
    right: number[];
    headBodyId: number;
    camId: number;
  } {
    return {
      origin: [this.origin[0], this.origin[1], this.origin[2]],
      look: [this.look[0], this.look[1], this.look[2]],
      up: [this.up[0], this.up[1], this.up[2]],
      right: [this.right[0], this.right[1], this.right[2]],
      headBodyId: this.headBodyId,
      camId: this.camId,
    };
  }

  setDepthPassHooks(_hooks: { begin?: () => void; end?: () => void }) {
    /* analytic depth — no mesh hide */
  }

  ensureFrustum(parentScene: THREE.Scene) {
    if (this.frustum) return;
    const near = this.cam.near;
    const far = Math.min(this.cam.far, 2.5);
    const fov = (this.cam.fov * Math.PI) / 180;
    const aspect = this.cam.aspect;
    const nh = Math.tan(fov / 2) * near;
    const nw = nh * aspect;
    const fh = Math.tan(fov / 2) * far;
    const fw = fh * aspect;
    // Three.js cam looks −Z; frustum in that local frame.
    const corners = [
      [-nw, nh, -near],
      [nw, nh, -near],
      [nw, -nh, -near],
      [-nw, -nh, -near],
      [-fw, fh, -far],
      [fw, fh, -far],
      [fw, -fh, -far],
      [-fw, -fh, -far],
    ] as const;
    const edges: number[] = [];
    const push = (a: readonly number[], b: readonly number[]) => {
      edges.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    };
    const origin = [0, 0, 0] as const;
    for (let i = 0; i < 4; i++) {
      push(origin, corners[i]);
      push(corners[i], corners[(i + 1) % 4]);
      push(corners[i + 4], corners[((i + 1) % 4) + 4]);
      push(corners[i], corners[i + 4]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(new Float32Array(edges), 3),
    );
    this.frustum = new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({
        color: 0x3ec7ff,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
      }),
    );
    this.frustum.frustumCulled = false;
    this.frustum.matrixAutoUpdate = false;
    parentScene.add(this.frustum);
  }

  setFrustumVisible(on: boolean) {
    this.frustumVisible = on;
    if (this.frustum) this.frustum.visible = on;
  }

  paintPreview(canvas: HTMLCanvasElement): void {
    const w = this.cropW;
    const h = this.cropH;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(w, h);
    const d = img.data;
    const src = this.out;
    for (let i = 0; i < src.length; i++) {
      const n = src[i];
      const g =
        Math.abs(n) < 1e-6
          ? 0
          : Math.max(0, Math.min(255, Math.round(((n + 1) * 0.5) * 255)));
      const o = i * 4;
      d[o] = Math.round(g * 0.55);
      d[o + 1] = Math.round(g * 0.9);
      d[o + 2] = Math.min(255, Math.round(g * 1.15));
      d[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /** Update optical pose from MuJoCo ``zed2i_depth`` (same as Viser/Warp); fill depth. */
  capture(model: MjModel, data: MjData): Float32Array {
    if (this.camId < 0 && this.headBodyId < 0) {
      this.out.fill(0);
      return this.out;
    }
    const xpos = asArray(data.xpos, model.nbody * 3);
    const xmat = asArray(data.xmat, model.nbody * 9);
    if (this.camId >= 0 && model.ncam > 0) {
      const camXpos = asArray(data.cam_xpos, model.ncam * 3);
      const camXmat = asArray(data.cam_xmat, model.ncam * 9);
      this.updateOpticalPoseFromCamera(camXpos, camXmat);
    } else {
      this.updateOpticalPoseFromHead(xpos, xmat);
    }

    let bx = 1.5;
    let by = 0;
    let bz = 0.11;
    if (this.ballBodyId >= 0) {
      const bp = this.ballBodyId * 3;
      bx = xpos[bp];
      by = xpos[bp + 1];
      bz = xpos[bp + 2];
    }
    let gx = 0;
    let gy = 0;
    let gz = 0;
    if (this.goalBodyId >= 0) {
      const gp = this.goalBodyId * 3;
      gx = xpos[gp];
      gy = xpos[gp + 1];
      gz = xpos[gp + 2];
    }

    const ox = this.origin[0];
    const oy = this.origin[1];
    const oz = this.origin[2];
    const lx = this.look[0];
    const ly = this.look[1];
    const lz = this.look[2];
    const ux = this.up[0];
    const uy = this.up[1];
    const uz = this.up[2];
    const rx = this.right[0];
    const ry = this.right[1];
    const rz = this.right[2];

    const fovY = (this.cam.fov * Math.PI) / 180;
    const tanHalfY = Math.tan(fovY / 2);
    const tanHalfX = tanHalfY * this.cam.aspect;

    for (let v = 0; v < this.renderH; v++) {
      // Image row 0 = top → +Y_cam (MuJoCo / OpenGL style).
      const ndcY = 1 - ((v + 0.5) / this.renderH) * 2;
      const py = ndcY * tanHalfY;
      for (let u = 0; u < this.renderW; u++) {
        const ndcX = ((u + 0.5) / this.renderW) * 2 - 1;
        const px = ndcX * tanHalfX;
        // Ray = look + right*px + up*py  (MuJoCo cam: look=−Z, right=+X, up=+Y)
        let dx = lx + rx * px + ux * py;
        let dy = ly + ry * px + uy * py;
        let dz = lz + rz * px + uz * py;
        const inv = 1 / (Math.hypot(dx, dy, dz) || 1);
        dx *= inv;
        dy *= inv;
        dz *= inv;

        let best = Infinity;
        if (dz < -1e-6) {
          const t = -oz / dz;
          if (t > this.cam.near && t < best) best = t;
        }
        {
          const t = raySphere(ox, oy, oz, dx, dy, dz, bx, by, bz, this.ballRadius);
          if (t > this.cam.near && t < best) best = t;
        }
        {
          const tL = rayZCyl(
            ox, oy, oz, dx, dy, dz,
            gx, gy + this.postY, gz + this.postHalfH,
            this.postRadius, this.postHalfH,
          );
          if (tL > this.cam.near && tL < best) best = tL;
          const tR = rayZCyl(
            ox, oy, oz, dx, dy, dz,
            gx, gy - this.postY, gz + this.postHalfH,
            this.postRadius, this.postHalfH,
          );
          if (tR > this.cam.near && tR < best) best = tR;
          const tC = rayYCyl(
            ox, oy, oz, dx, dy, dz,
            gx, gy, gz + this.crossbarZ,
            this.postRadius, this.crossbarHalf,
          );
          if (tC > this.cam.near && tC < best) best = tC;
        }
        this.depthBuf[v * this.renderW + u] =
          Number.isFinite(best) && best < this.cam.far ? best : 0;
      }
    }

    this.cropAndNormalize();
    return this.out;
  }

  /** Prefer MuJoCo fixed-camera pose (Viser / CameraSensor / ``zed2i_depth``). */
  private updateOpticalPoseFromCamera(
    camXpos: Float32Array | Float64Array,
    camXmat: Float32Array | Float64Array,
  ) {
    const p = this.camId * 3;
    const o = this.camId * 9;
    this.origin[0] = camXpos[p];
    this.origin[1] = camXpos[p + 1];
    this.origin[2] = camXpos[p + 2];
    for (let i = 0; i < 9; i++) this.Rcam[i] = camXmat[o + i];
    this.applyCamAxesToFrustum();
  }

  /**
   * Fallback: world optical pose = head_camera_link × (cam_pos, cam_quat).
   * Same math as MuJoCo when ``zed2i_depth`` is parented to the head.
   */
  private updateOpticalPoseFromHead(
    xpos: Float32Array | Float64Array,
    xmat: Float32Array | Float64Array,
  ) {
    const p = this.headBodyId * 3;
    const o = this.headBodyId * 9;
    for (let i = 0; i < 9; i++) this.Rhead[i] = xmat[o + i];

    const lx = this.camPos[0];
    const ly = this.camPos[1];
    const lz = this.camPos[2];
    this.origin[0] =
      xpos[p] + this.Rhead[0] * lx + this.Rhead[1] * ly + this.Rhead[2] * lz;
    this.origin[1] =
      xpos[p + 1] + this.Rhead[3] * lx + this.Rhead[4] * ly + this.Rhead[5] * lz;
    this.origin[2] =
      xpos[p + 2] + this.Rhead[6] * lx + this.Rhead[7] * ly + this.Rhead[8] * lz;

    const Rlc = quatToMat9(this.camQuat);
    mulMat9(this.Rhead, Rlc, this.Rcam);
    this.applyCamAxesToFrustum();
  }

  private applyCamAxesToFrustum() {
    // MuJoCo camera axes: right=col0, up=col1, look=−col2
    this.right[0] = this.Rcam[0];
    this.right[1] = this.Rcam[3];
    this.right[2] = this.Rcam[6];
    this.up[0] = this.Rcam[1];
    this.up[1] = this.Rcam[4];
    this.up[2] = this.Rcam[7];
    this.look[0] = -this.Rcam[2];
    this.look[1] = -this.Rcam[5];
    this.look[2] = -this.Rcam[8];

    this.cam.position.set(this.origin[0], this.origin[1], this.origin[2]);
    this.cam.up.set(this.up[0], this.up[1], this.up[2]);
    this.cam.lookAt(
      this.origin[0] + this.look[0],
      this.origin[1] + this.look[1],
      this.origin[2] + this.look[2],
    );
    this.cam.updateMatrixWorld(true);
    if (this.frustum) {
      this.frustum.matrix.copy(this.cam.matrixWorld);
      this.frustum.visible = this.frustumVisible;
    }
  }

  /** Match ``align_depth`` / ``normalize_depth`` (play=no DR). */
  private cropAndNormalize() {
    const top = Math.max(0, Math.floor((this.renderH - this.cropH) / 2));
    const left = Math.max(0, Math.floor((this.renderW - this.cropW) / 2));
    const maxM = this.depthMax;
    const minM = this.depthMin;
    for (let y = 0; y < this.cropH; y++) {
      for (let x = 0; x < this.cropW; x++) {
        const d = this.depthBuf[(top + y) * this.renderW + (left + x)];
        const valid = Number.isFinite(d) && d > 0;
        let v = 0;
        // Paper: only depths in [min, max] → 2D/max−1; else 0.
        if (valid && d >= minM && d <= maxM) {
          v = (2.0 * d) / maxM - 1.0;
        }
        this.out[y * this.cropW + x] = v;
      }
    }
  }

  dispose() {
    if (this.frustum) {
      this.frustum.parent?.remove(this.frustum);
      this.frustum.geometry.dispose();
      (this.frustum.material as THREE.Material).dispose();
      this.frustum = null;
    }
  }
}

/** Row-major 3×3 from MuJoCo wxyz quat. */
function quatToMat9(q: [number, number, number, number]): Float64Array {
  const [w, x, y, z] = q;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  const m = new Float64Array(9);
  m[0] = 1 - 2 * (yy + zz);
  m[1] = 2 * (xy - wz);
  m[2] = 2 * (xz + wy);
  m[3] = 2 * (xy + wz);
  m[4] = 1 - 2 * (xx + zz);
  m[5] = 2 * (yz - wx);
  m[6] = 2 * (xz - wy);
  m[7] = 2 * (yz + wx);
  m[8] = 1 - 2 * (xx + yy);
  return m;
}

/** C = A @ B, row-major 3×3. */
function mulMat9(A: Float64Array, B: Float64Array, C: Float64Array) {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      C[r * 3 + c] =
        A[r * 3 + 0] * B[0 * 3 + c] +
        A[r * 3 + 1] * B[1 * 3 + c] +
        A[r * 3 + 2] * B[2 * 3 + c];
    }
  }
}

function raySphere(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  cx: number, cy: number, cz: number,
  r: number,
): number {
  const fx = ox - cx;
  const fy = oy - cy;
  const fz = oz - cz;
  const b = fx * dx + fy * dy + fz * dz;
  const c = fx * fx + fy * fy + fz * fz - r * r;
  const disc = b * b - c;
  if (disc < 0) return Infinity;
  const s = Math.sqrt(disc);
  const t0 = -b - s;
  const t1 = -b + s;
  if (t0 > 1e-4) return t0;
  if (t1 > 1e-4) return t1;
  return Infinity;
}

function rayZCyl(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  cx: number, cy: number, cz: number,
  r: number, halfH: number,
): number {
  const fx = ox - cx;
  const fy = oy - cy;
  const a = dx * dx + dy * dy;
  if (a < 1e-12) return Infinity;
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - r * r;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return Infinity;
  const s = Math.sqrt(disc);
  let best = Infinity;
  for (const t of [(-b - s) / (2 * a), (-b + s) / (2 * a)]) {
    if (t <= 1e-4) continue;
    const z = oz + t * dz;
    if (Math.abs(z - cz) <= halfH && t < best) best = t;
  }
  return best;
}

function rayYCyl(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  cx: number, cy: number, cz: number,
  r: number, halfL: number,
): number {
  const fx = ox - cx;
  const fz = oz - cz;
  const a = dx * dx + dz * dz;
  if (a < 1e-12) return Infinity;
  const b = 2 * (fx * dx + fz * dz);
  const c = fx * fx + fz * fz - r * r;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return Infinity;
  const s = Math.sqrt(disc);
  let best = Infinity;
  for (const t of [(-b - s) / (2 * a), (-b + s) / (2 * a)]) {
    if (t <= 1e-4) continue;
    const y = oy + t * dy;
    if (Math.abs(y - cy) <= halfL && t < best) best = t;
  }
  return best;
}
