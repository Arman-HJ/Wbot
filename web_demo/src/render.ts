import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { MjData, MjModel } from "mujoco-js";
import { asArray } from "./mujoco_util";

/** MuJoCo mjtGeom values. */
const MJGEOM_PLANE = 0;
const MJGEOM_SPHERE = 2;
const MJGEOM_CAPSULE = 3;
const MJGEOM_ELLIPSOID = 4;
const MJGEOM_CYLINDER = 5;
const MJGEOM_BOX = 6;
const MJGEOM_MESH = 7;

type GeomMesh = THREE.Mesh & { userData: { bodyId: number; geomId: number } };

/**
 * MuJoCo "groundplane": builtin checker (2x2 squares per tile), mark="edge"
 * outlines the tile, texuniform + texrepeat="5 5" → 5 tiles per meter.
 */
function makeGroundplaneTexture(floorSize: number): THREE.CanvasTexture {
  const n = 512;
  const half = n / 2;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = n;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(51, 77, 102)"; // rgb1 0.2 0.3 0.4
  ctx.fillRect(0, 0, n, n);
  ctx.fillStyle = "rgb(26, 51, 77)"; // rgb2 0.1 0.2 0.3
  ctx.fillRect(half, 0, half, half);
  ctx.fillRect(0, half, half, half);
  ctx.strokeStyle = "rgb(204, 204, 204)"; // markrgb 0.8 0.8 0.8
  ctx.lineWidth = 4;
  ctx.strokeRect(0, 0, n, n);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(floorSize * 5, floorSize * 5);
  tex.anisotropy = 8;
  return tex;
}

/**
 * Three.js renderer matching mjviser/Viser: mesh vertices stay in the geom
 * frame, geoms are parented under body groups, and each frame body groups
 * take xpos/xmat (geom_pos/quat stay as local child transforms).
 */
export class SceneRenderer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private geomMeshes: GeomMesh[] = [];
  private bodyGroups = new Map<number, THREE.Group>();
  private bodyPickMeshes: THREE.Object3D[] = [];
  private rootGroup = new THREE.Group();
  private model!: MjModel;
  private followCam = true;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b1220);
    this.scene.add(this.rootGroup);

    this.camera = new THREE.PerspectiveCamera(
      50,
      window.innerWidth / window.innerHeight,
      0.05,
      50,
    );
    this.camera.position.set(2.2, -2.4, 1.4);
    this.camera.up.set(0, 0, 1);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0, 0.7);
    this.controls.enableDamping = true;
    this.controls.update();

    const hemi = new THREE.HemisphereLight(0xc9d8ff, 0x223344, 1.1);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(3, -2, 5);
    dir.castShadow = true;
    this.scene.add(dir);

    const floorSize = 20;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(floorSize, floorSize),
      new THREE.MeshStandardMaterial({
        map: makeGroundplaneTexture(floorSize),
        roughness: 0.9,
      }),
    );
    floor.receiveShadow = true;
    this.scene.add(floor);

    window.addEventListener("resize", () => this.onResize());
  }

  /** Bodies/geoms eligible for Ctrl+drag picking. */
  getBodyMeshes(): THREE.Object3D[] {
    return this.bodyPickMeshes;
  }

  /** Body groups for ghost mesh cloning (Show motion). */
  getBodyGroups(): Map<number, THREE.Group> {
    return this.bodyGroups;
  }

  private depthHidden: THREE.Object3D[] = [];

  /**
   * Hide robot bodies during head-depth capture so the policy sees ball/goal/floor
   * (browser depth is approximate; self-occlusion near the head destroys standing).
   */
  beginDepthPass(keepBodyIds: Iterable<number>) {
    const keep = new Set(keepBodyIds);
    this.depthHidden = [];
    for (const [id, group] of this.bodyGroups) {
      if (keep.has(id)) continue;
      if (group.visible) {
        group.visible = false;
        this.depthHidden.push(group);
      }
    }
  }

  endDepthPass() {
    for (const g of this.depthHidden) g.visible = true;
    this.depthHidden = [];
  }

  onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  buildFromModel(model: MjModel) {
    this.model = model;
    for (const m of this.geomMeshes) {
      m.parent?.remove(m);
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    }
    for (const g of this.bodyGroups.values()) {
      this.rootGroup.remove(g);
    }
    this.geomMeshes = [];
    this.bodyGroups.clear();
    this.bodyPickMeshes = [];

    const ngeom = model.ngeom;
    const geomPos = asArray(model.geom_pos as any, ngeom * 3);
    const geomQuat = asArray(model.geom_quat as any, ngeom * 4);
    // Per-geom clone — do not share BufferGeometry across geoms that apply
    // different local frames.
    let nMesh = 0;
    let nPrim = 0;
    let meshVerts = 0;

    for (let g = 0; g < ngeom; g++) {
      const group = Number(model.geom_group[g]);
      const gtype = Number(model.geom_type[g]);
      if (gtype === MJGEOM_PLANE) continue;
      if (gtype !== MJGEOM_MESH && group >= 3) continue;

      const geometry = this.makeGeomGeometry(model, g, gtype);
      if (!geometry) continue;
      if (gtype === MJGEOM_MESH) {
        nMesh += 1;
        const pos = geometry.getAttribute("position");
        meshVerts += pos ? pos.count : 0;
      } else {
        nPrim += 1;
      }

      const rgba = [
        Number(model.geom_rgba[g * 4]),
        Number(model.geom_rgba[g * 4 + 1]),
        Number(model.geom_rgba[g * 4 + 2]),
        Number(model.geom_rgba[g * 4 + 3]),
      ];
      const matId = Number(model.geom_matid[g]);
      let color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
      let opacity = rgba[3];
      if (matId >= 0 && Number.isFinite(matId)) {
        color = new THREE.Color(
          Number(model.mat_rgba[matId * 4]),
          Number(model.mat_rgba[matId * 4 + 1]),
          Number(model.mat_rgba[matId * 4 + 2]),
        );
        opacity = Number(model.mat_rgba[matId * 4 + 3]);
      }
      if (color.r + color.g + color.b < 0.05) {
        color.set(0xc8cdd6);
      }

      const material = new THREE.MeshStandardMaterial({
        color,
        metalness: 0.2,
        roughness: 0.5,
        transparent: opacity < 0.99,
        opacity,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material) as unknown as GeomMesh;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const bodyId = Number(model.geom_bodyid[g]);
      mesh.userData = { bodyId, geomId: g };

      // Local geom pose in body frame (same as mjviser.merge_geoms).
      mesh.position.set(geomPos[g * 3], geomPos[g * 3 + 1], geomPos[g * 3 + 2]);
      // MuJoCo quat is wxyz → Three.js xyzw.
      mesh.quaternion.set(
        geomQuat[g * 4 + 1],
        geomQuat[g * 4 + 2],
        geomQuat[g * 4 + 3],
        geomQuat[g * 4 + 0],
      );
      mesh.matrixAutoUpdate = true;

      let bodyGroup = this.bodyGroups.get(bodyId);
      if (!bodyGroup) {
        bodyGroup = new THREE.Group();
        bodyGroup.matrixAutoUpdate = false;
        this.bodyGroups.set(bodyId, bodyGroup);
        this.rootGroup.add(bodyGroup);
      }
      bodyGroup.add(mesh);
      this.geomMeshes.push(mesh);
      if (bodyId > 0) this.bodyPickMeshes.push(mesh);
    }
    console.info(
      `[render] geoms: mesh=${nMesh} (verts≈${meshVerts}) prim=${nPrim} shown=${this.geomMeshes.length} bodies=${this.bodyGroups.size}`,
    );
  }

  private makeGeomGeometry(
    model: MjModel,
    g: number,
    gtype: number,
  ): THREE.BufferGeometry | null {
    const size0 = Number(model.geom_size[g * 3]);
    const size1 = Number(model.geom_size[g * 3 + 1]);
    const size2 = Number(model.geom_size[g * 3 + 2]);

    if (gtype === MJGEOM_SPHERE) {
      return new THREE.SphereGeometry(size0, 16, 12);
    }
    if (gtype === MJGEOM_BOX) {
      return new THREE.BoxGeometry(size0 * 2, size1 * 2, size2 * 2);
    }
    if (gtype === MJGEOM_CAPSULE) {
      const geom = new THREE.CapsuleGeometry(size0, size1 * 2, 6, 10);
      geom.rotateX(Math.PI / 2);
      return geom;
    }
    if (gtype === MJGEOM_CYLINDER) {
      const geom = new THREE.CylinderGeometry(size0, size0, size1 * 2, 16);
      geom.rotateX(Math.PI / 2);
      return geom;
    }
    if (gtype === MJGEOM_ELLIPSOID) {
      const geom = new THREE.SphereGeometry(1, 16, 12);
      geom.scale(size0, size1, size2);
      return geom;
    }
    if (gtype === MJGEOM_MESH) {
      const meshId = Number(model.geom_dataid[g]);
      if (meshId < 0) return null;
      return buildMeshGeometry(model, meshId);
    }
    return null;
  }

  sync(data: MjData) {
    const nbody = this.model.nbody;
    const xpos = asArray(data.xpos, nbody * 3);
    const xmat = asArray(data.xmat, nbody * 9);

    for (const [bodyId, group] of this.bodyGroups) {
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
    // Keep matrixWorld fresh for Ctrl+drag raycasts (groups skip auto-update).
    this.rootGroup.updateMatrixWorld(true);

    if (this.followCam && nbody > 1) {
      // Carry the camera along in xy so fast gaits stay framed (orbit offset kept).
      const target = this.controls.target;
      this.camera.position.x += xpos[3] - target.x;
      this.camera.position.y += xpos[4] - target.y;
      target.set(xpos[3], xpos[4], xpos[5] + 0.2);
    }
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function buildMeshGeometry(model: MjModel, meshId: number): THREE.BufferGeometry {
  const vertAdr = Number(model.mesh_vertadr[meshId]);
  const vertNum = Number(model.mesh_vertnum[meshId]);
  const faceAdr = Number(model.mesh_faceadr[meshId]);
  const faceNum = Number(model.mesh_facenum[meshId]);

  const verts = asArray(model.mesh_vert, (vertAdr + vertNum) * 3);
  const faces = new Uint32Array(faceNum * 3);
  for (let i = 0; i < faceNum * 3; i++) {
    faces[i] =
      typeof model.mesh_face.get === "function"
        ? Number(model.mesh_face.get(faceAdr * 3 + i))
        : Number(model.mesh_face[faceAdr * 3 + i]);
  }

  const positions = new Float32Array(vertNum * 3);
  for (let i = 0; i < vertNum * 3; i++) {
    positions[i] = verts[vertAdr * 3 + i];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(faces, 1));
  geometry.computeVertexNormals();
  return geometry;
}
