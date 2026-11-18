import * as ort from "onnxruntime-web";
import type { MjData, MjModel } from "mujoco-js";
import type {
  Alignment,
  FieldSpec,
  Manifest,
  MotionData,
  ObsLayout,
  PlayProfile,
} from "./types";
import {
  matrixFromQuatFirst2Cols,
  quatApply,
  quatApplyInverse,
  quatInv,
  quatMul,
  yawQuat,
} from "./math";
import { asArray, name2id, writeNum, type MujocoModule } from "./mujoco_util";
import type { DavisDepthCamera } from "./depth";

ort.env.wasm.wasmPaths = "/ort/";

function normalizeFields(
  fields: Alignment["observations"]["actor"]["fields"],
): FieldSpec[] {
  if (Array.isArray(fields)) {
    return fields.filter((f) => f && f.name);
  }
  if (fields && typeof fields === "object") {
    return Object.entries(fields).map(([name, v]) => ({ name, ...v }));
  }
  return [];
}

/**
 * Policy controller driven by deployment_alignment fields (not a hardcoded
 * cross-task layout). Play profile comes from Viser routing (AMP vs Kick).
 */
export class PolicyController {
  readonly manifest: Manifest;
  readonly alignment: Alignment;
  readonly playProfile: PlayProfile;
  readonly jointNames: string[];
  readonly defaultPos: Float64Array;
  readonly actionScale: Float64Array;
  readonly frameDim: number;
  readonly historyLength: number;
  readonly historyFrameDim: number;
  readonly historyTerms: string[];
  readonly obsLayout: ObsLayout;
  readonly obsDim: number;
  readonly fields: FieldSpec[];
  readonly hasVelocityCommand: boolean;

  private session!: ort.InferenceSession;
  private packUrl = "";
  private currentCheckpoint = "";
  private history: Float64Array[] = [];
  private lastAction: Float64Array;
  private command = new Float64Array(3);
  private jointQposAdr!: Int32Array;
  private jointDofAdr!: Int32Array;
  private actuatorIds!: Int32Array;
  private torsoBodyId = -1;
  private rootBodyId = -1;
  private gyroAdr = -1;
  private episodeSteps = 0;
  private mj!: MujocoModule;

  // Kick motion tracking (Viser KickMotionCommand play-once).
  private motion: MotionData | null = null;
  private motionJointMap: Int32Array | null = null; // policy joint i → motion joint index
  private motionNumJoints = 0;
  private timeStep = 0;
  private holdFrame = 0;
  private clipPlaying = false;
  private alignOn = false;
  private alignYaw = new Float32Array([1, 0, 0, 0]);
  private alignT = new Float32Array(3);
  private alignF0 = new Float32Array(3);
  private freeQposAdr = -1;
  private freeDofAdr = -1;
  private readonly tmpQuat = new Float32Array(4);
  private readonly tmpQuat2 = new Float32Array(4);
  private readonly tmpVec = new Float32Array(3);
  private readonly tmpOri6 = new Float32Array(6);
  private kickCommandBuf: Float32Array | null = null;
  // Policy paused: ctrl blends from blendFrom to blendTo, then holds blendTo.
  private blendFrom: Float64Array | null = null;
  private blendTo: Float64Array | null = null;
  private blendT = 0;
  private blendDur = 1;
  private depthCam: DavisDepthCamera | null = null;
  private readonly dualInput: boolean;
  /** Soft-start ONNX actions after reset (avoids tip-over from cold depth). */

  constructor(manifest: Manifest, alignment: Alignment) {
    this.manifest = manifest;
    this.alignment = alignment;
    this.playProfile =
      manifest.play_profile ||
      alignment.meta?.play_profile ||
      (manifest.task_id?.includes("Davis")
        ? "davis"
        : manifest.task_id?.includes("Kick") ||
            manifest.task_id?.includes("Tracking")
          ? "kick"
          : "amp_twist");
    this.dualInput =
      this.playProfile === "davis" ||
      (manifest.obs_inputs || []).includes("depth");
    this.jointNames = manifest.joint_names;
    this.defaultPos = Float64Array.from(manifest.default_joint_pos);
    this.actionScale = Float64Array.from(manifest.action_scale);
    this.fields = normalizeFields(alignment.observations.actor.fields);
    this.hasVelocityCommand =
      manifest.has_velocity_command ??
      ((this.playProfile === "amp_twist" || this.playProfile === "velocity") &&
        this.fields.some((f) => f.name === "command"));
    this.frameDim =
      manifest.frame_dim ||
      alignment.observations.actor.frame_dim ||
      this.fields.reduce((s, f) => s + (f.dim || 0), 0);
    const ah = alignment.observations.actor_history;
    this.obsLayout =
      manifest.obs_layout ||
      alignment.observations.obs_layout ||
      (ah ? "actor_plus_history" : "stacked");
    this.historyTerms = [
      ...(manifest.history_terms || ah?.terms || []),
    ];
    this.historyFrameDim = Math.max(
      1,
      manifest.history_frame_dim ||
        ah?.frame_dim ||
        (this.obsLayout === "actor_plus_history"
          ? this.historyTerms.reduce((s, name) => {
              const f = this.fields.find((x) => x.name === name);
              return s + (f?.dim || 0);
            }, 0)
          : this.frameDim),
    );
    this.historyLength = Math.max(
      1,
      manifest.history_length ||
        ah?.history_length ||
        alignment.observations.actor.history_length ||
        1,
    );
    this.obsDim =
      manifest.obs_dim ||
      (this.obsLayout === "actor_plus_history"
        ? this.frameDim + this.historyLength * this.historyFrameDim
        : this.frameDim * this.historyLength);
    this.lastAction = new Float64Array(manifest.num_actions);

    if (!this.fields.length) {
      throw new Error(
        `Pack ${manifest.id} has no actor fields — cannot build observations`,
      );
    }
    if (
      this.obsLayout === "actor_plus_history" &&
      this.historyTerms.some((n) => !this.fields.some((f) => f.name === n))
    ) {
      throw new Error(
        `Pack ${manifest.id} history_terms missing from actor fields: ` +
          `${this.historyTerms.join(",")}`,
      );
    }
  }

  getCurrentCheckpoint(): string {
    return this.currentCheckpoint;
  }

  /** Attach head depth renderer (DAVIS dual-input packs). */
  setDepthCamera(cam: DavisDepthCamera | null) {
    this.depthCam = cam;
  }

  /** Swap ONNX weights without rebuilding the scene (Viser checkpoint dropdown). */
  async loadCheckpoint(nameOrFile: string) {
    if (!this.packUrl) throw new Error("Policy not initialized");
    const entries = this.manifest.checkpoints || [];
    const entry =
      entries.find((c) => c.name === nameOrFile || c.file === nameOrFile) ||
      null;
    const rel =
      entry?.file ||
      (nameOrFile.includes("/") || nameOrFile.endsWith(".onnx")
        ? nameOrFile
        : this.manifest.files.policy);
    const url = `${this.packUrl}/${rel}?t=${Date.now()}`;
    this.session = await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
    });
    this.currentCheckpoint = entry?.name || nameOrFile;
    this.resetHistory();
  }

  async init(mj: MujocoModule, model: MjModel, packUrl: string) {
    this.mj = mj;
    this.packUrl = packUrl;
    const defaultName =
      this.manifest.default_checkpoint ||
      this.manifest.checkpoints?.[this.manifest.checkpoints.length - 1]?.name ||
      this.manifest.files.policy;
    await this.loadCheckpoint(defaultName);
    const n = this.jointNames.length;
    this.jointQposAdr = new Int32Array(n);
    this.jointDofAdr = new Int32Array(n);
    this.actuatorIds = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const jid = name2id(mj, model, "joint", this.jointNames[i]);
      if (jid < 0) throw new Error(`Joint missing: ${this.jointNames[i]}`);
      this.jointQposAdr[i] = model.jnt_qposadr[jid];
      this.jointDofAdr[i] = model.jnt_dofadr[jid];
      const aid = name2id(mj, model, "actuator", this.jointNames[i]);
      if (aid < 0) throw new Error(`Actuator missing: ${this.jointNames[i]}`);
      this.actuatorIds[i] = aid;
    }
    const torso =
      this.manifest.torso_body || this.alignment.bodies?.torso || "torso_link";
    this.torsoBodyId = name2id(mj, model, "body", torso);
    if (this.torsoBodyId < 0) throw new Error(`Body missing: ${torso}`);
    const root = this.manifest.root_body || this.alignment.bodies?.root || "pelvis";
    this.rootBodyId = name2id(mj, model, "body", root);
    if (this.rootBodyId < 0) throw new Error(`Body missing: ${root}`);

    // Training uses builtin_sensor("robot/imu_ang_vel") for kick/tracking too
    // (pelvis gyro). Using torso cvel here made getup fail mid-clip.
    {
      const angField = this.fields.find((f) => f.name === "base_ang_vel");
      const sensorName = String(
        angField?.params?.sensor_name ?? "imu_ang_vel",
      ).replace(/^robot\//, "");
      const sid = name2id(mj, model, "sensor", sensorName);
      if (sid < 0) throw new Error(`Sensor missing: ${sensorName}`);
      this.gyroAdr = model.sensor_adr[sid];
    }
    const freeJid = name2id(mj, model, "joint", "floating_base_joint");
    this.freeQposAdr = freeJid >= 0 ? model.jnt_qposadr[freeJid] : 0;
    this.freeDofAdr = freeJid >= 0 ? model.jnt_dofadr[freeJid] : 0;

    if (this.depthCam) {
      const headName = this.depthCam.getHeadName();
      const hid = name2id(mj, model, "body", headName);
      if (hid < 0) throw new Error(`DAVIS head body missing: ${headName}`);
      this.depthCam.setHeadBodyId(hid);
    }

    if (this.playProfile === "kick") {
      const motionFile = this.manifest.files.motion || "motion.json";
      const res = await fetch(`${packUrl}/${motionFile}`);
      if (!res.ok) {
        throw new Error(
          `Kick pack missing ${motionFile} (HTTP ${res.status}) — re-pack the run`,
        );
      }
      this.motion = (await res.json()) as MotionData;
      this.motionNumJoints = this.motion.joint_names.length;
      this.motionJointMap = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const idx = this.motion.joint_names.indexOf(this.jointNames[i]);
        if (idx < 0) {
          throw new Error(
            `Motion missing joint ${this.jointNames[i]} (motion has ${this.motionNumJoints} joints)`,
          );
        }
        this.motionJointMap[i] = idx;
      }
      this.kickCommandBuf = new Float32Array(2 * n);
      this.holdAtSelectedFrame();
    }

    this.resetHistory();
  }

  setCommand(vx: number, vy: number, wz: number) {
    this.command[0] = vx;
    this.command[1] = vy;
    this.command[2] = wz;
  }

  getCommand(): Float64Array {
    return this.command;
  }

  private clampHoldFrame(frame: number): number {
    if (!this.motion) return 0;
    return Math.max(0, Math.min(Math.floor(frame), this.motion.T - 1));
  }

  /** Kick: hold reference at selected Hold Frame (aligned to robot). */
  holdAtSelectedFrame() {
    this.clipPlaying = false;
    this.holdFrame = this.clampHoldFrame(this.holdFrame);
    this.timeStep = this.holdFrame;
    this.alignOn = false;
  }

  /** @deprecated use holdAtSelectedFrame */
  holdFrameZero() {
    this.holdFrame = 0;
    this.holdAtSelectedFrame();
  }

  setHoldFrame(frame: number) {
    this.holdFrame = this.clampHoldFrame(frame);
    if (!this.clipPlaying) {
      this.timeStep = this.holdFrame;
    }
  }

  getHoldFrame(): number {
    return this.holdFrame;
  }

  getMotionLength(): number {
    return this.motion?.T ?? 0;
  }

  /** Kick: freeze SE(2) align at frame 0 and advance motion once (Viser Play Clip). */
  startPlayClip(model: MjModel, data: MjData): boolean {
    if (this.playProfile !== "kick" || !this.motion || this.clipPlaying) {
      return false;
    }
    this.stopBlend();
    this.timeStep = 0;
    this.captureAlign(model, data, 0);
    this.clipPlaying = true;
    return true;
  }

  /** Joint targets of the episode start pose (getup: Hold Frame of the clip). */
  private initJointTargets(): Float64Array {
    const n = this.jointNames.length;
    if (!this.manifest.reset_from_motion || !this.motion) {
      return Float64Array.from(this.defaultPos);
    }
    const t = this.clampHoldFrame(this.holdFrame);
    const map = this.motionJointMap!;
    const nj = this.motionNumJoints;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.motion.joint_pos[t * nj + map[i]];
    return out;
  }

  /** Pause the policy and PD-blend joint targets from current joints to the start pose. */
  startBlendToInit(model: MjModel, data: MjData, durationS: number) {
    const qpos = asArray(data.qpos, model.nq);
    const n = this.jointNames.length;
    const from = new Float64Array(n);
    for (let i = 0; i < n; i++) from[i] = qpos[this.jointQposAdr[i]];
    this.clipPlaying = false;
    this.holdAtSelectedFrame();
    this.lastAction.fill(0);
    this.blendFrom = from;
    this.blendTo = this.initJointTargets();
    this.blendT = 0;
    this.blendDur = Math.max(0.05, durationS);
  }

  stopBlend() {
    this.blendFrom = null;
    this.blendTo = null;
  }

  isBlending(): boolean {
    return this.blendTo !== null;
  }

  private stepBlend(data: MjData) {
    const from = this.blendFrom!;
    const to = this.blendTo!;
    this.blendT = Math.min(this.blendDur, this.blendT + this.manifest.policy_dt);
    const s = this.blendT / this.blendDur;
    const w = s * s * (3 - 2 * s);
    for (let i = 0; i < this.jointNames.length; i++) {
      writeNum(data.ctrl, this.actuatorIds[i], from[i] + (to[i] - from[i]) * w);
    }
  }

  isClipPlaying(): boolean {
    return this.clipPlaying;
  }

  motionStatus(): string {
    if (this.blendTo) {
      if (this.blendT < this.blendDur) {
        return `插值到初始位… ${this.blendT.toFixed(1)}/${this.blendDur.toFixed(1)}s（策略暂停）`;
      }
      return "已到初始位（策略暂停），点 Play Clip 开始";
    }
    if (!this.motion) return "";
    if (this.clipPlaying) {
      return `Playing clip… frame ${this.timeStep}/${this.motion.T - 1}`;
    }
    return `Holding frame ${this.holdFrame} (aligned to robot)`;
  }

  hasMotion(): boolean {
    return this.playProfile === "kick" && !!this.motion;
  }

  getMotionTimeStep(): number {
    return this.timeStep;
  }

  getTorsoBodyId(): number {
    return this.torsoBodyId;
  }

  getEpisodeSteps(): number {
    return this.episodeSteps;
  }

  getLastAction(): Float64Array {
    return this.lastAction;
  }

  /**
   * Write current motion reference into ``ghostData`` qpos (SE(2)-aligned),
   * matching Viser ghost: free-joint root + controlled joints.
   * While holding frame 0, re-captures align so the ghost tracks the robot.
   */
  writeGhostQpos(
    model: MjModel,
    robotData: MjData,
    ghostData: MjData,
  ): boolean {
    if (!this.motion || this.freeQposAdr < 0) return false;
    if (!this.clipPlaying) {
      this.timeStep = this.clampHoldFrame(this.holdFrame);
      this.captureAlign(model, robotData, this.timeStep);
    }
    const m = this.motion;
    const t = this.timeStep;
    const rootPos = this.rootPosAt(t);
    const rootQuat = this.rootQuatAt(t);
    let px: number, py: number, pz: number;
    let qw: number, qx: number, qy: number, qz: number;
    if (this.alignOn) {
      this.tmpVec[0] = rootPos[0] - this.alignF0[0];
      this.tmpVec[1] = rootPos[1] - this.alignF0[1];
      this.tmpVec[2] = rootPos[2] - this.alignF0[2];
      quatApply(this.alignYaw, this.tmpVec, this.tmpVec);
      px = this.alignT[0] + this.tmpVec[0];
      py = this.alignT[1] + this.tmpVec[1];
      pz = this.alignT[2] + this.tmpVec[2];
      quatMul(this.alignYaw, rootQuat, this.tmpQuat);
      qw = this.tmpQuat[0];
      qx = this.tmpQuat[1];
      qy = this.tmpQuat[2];
      qz = this.tmpQuat[3];
    } else {
      px = rootPos[0];
      py = rootPos[1];
      pz = rootPos[2];
      qw = rootQuat[0];
      qx = rootQuat[1];
      qy = rootQuat[2];
      qz = rootQuat[3];
    }
    const adr = this.freeQposAdr;
    writeNum(ghostData.qpos, adr + 0, px);
    writeNum(ghostData.qpos, adr + 1, py);
    writeNum(ghostData.qpos, adr + 2, pz);
    writeNum(ghostData.qpos, adr + 3, qw);
    writeNum(ghostData.qpos, adr + 4, qx);
    writeNum(ghostData.qpos, adr + 5, qy);
    writeNum(ghostData.qpos, adr + 6, qz);

    const map = this.motionJointMap!;
    const nj = this.motionNumJoints;
    const jp = m.joint_pos;
    for (let i = 0; i < this.jointNames.length; i++) {
      writeNum(ghostData.qpos, this.jointQposAdr[i], jp[t * nj + map[i]]);
    }
    return true;
  }

  /** Align a motion root pose the same way as ``writeGhostQpos``. */
  private alignedRootAt(
    t: number,
    outPos: Float32Array,
    outQuat: Float32Array,
  ) {
    const rootPos = this.rootPosAt(t);
    const rootQuat = this.rootQuatAt(t);
    if (this.alignOn) {
      this.tmpVec[0] = rootPos[0] - this.alignF0[0];
      this.tmpVec[1] = rootPos[1] - this.alignF0[1];
      this.tmpVec[2] = rootPos[2] - this.alignF0[2];
      quatApply(this.alignYaw, this.tmpVec, this.tmpVec);
      outPos[0] = this.alignT[0] + this.tmpVec[0];
      outPos[1] = this.alignT[1] + this.tmpVec[1];
      outPos[2] = this.alignT[2] + this.tmpVec[2];
      quatMul(this.alignYaw, rootQuat, outQuat);
    } else {
      outPos[0] = rootPos[0];
      outPos[1] = rootPos[1];
      outPos[2] = rootPos[2];
      outQuat[0] = rootQuat[0];
      outQuat[1] = rootQuat[1];
      outQuat[2] = rootQuat[2];
      outQuat[3] = rootQuat[3];
    }
  }

  /**
   * Fill ghost qvel from motion joint_vel + finite-diff free-joint root
   * (needed for lin/ang vel reward terms after ``mj_forward``).
   */
  writeGhostVel(_model: MjModel, ghostData: MjData): void {
    if (!this.motion || this.freeDofAdr < 0) return;
    const m = this.motion;
    const t = this.timeStep;
    const map = this.motionJointMap!;
    const nj = this.motionNumJoints;
    const jv = m.joint_vel;
    for (let i = 0; i < this.jointNames.length; i++) {
      writeNum(ghostData.qvel, this.jointDofAdr[i], jv[t * nj + map[i]]);
    }
    const fps = Math.max(1, m.fps || 50);
    const dt = 1 / fps;
    const t1 = Math.min(t + 1, m.T - 1);
    const p0 = new Float32Array(3);
    const p1 = new Float32Array(3);
    const q0 = new Float32Array(4);
    const q1 = new Float32Array(4);
    this.alignedRootAt(t, p0, q0);
    this.alignedRootAt(t1, p1, q1);
    const adr = this.freeDofAdr;
    writeNum(ghostData.qvel, adr + 0, (p1[0] - p0[0]) / dt);
    writeNum(ghostData.qvel, adr + 1, (p1[1] - p0[1]) / dt);
    writeNum(ghostData.qvel, adr + 2, (p1[2] - p0[2]) / dt);
    // Angular ≈ 2 * xyz(inv(q0)*q1) / dt
    quatMul(quatInv(q0, this.tmpQuat), q1, this.tmpQuat2);
    let w = this.tmpQuat2[0];
    let x = this.tmpQuat2[1];
    let y = this.tmpQuat2[2];
    let z = this.tmpQuat2[3];
    if (w < 0) {
      w = -w;
      x = -x;
      y = -y;
      z = -z;
    }
    writeNum(ghostData.qvel, adr + 3, (2 * x) / dt);
    writeNum(ghostData.qvel, adr + 4, (2 * y) / dt);
    writeNum(ghostData.qvel, adr + 5, (2 * z) / dt);
  }

  private captureAlign(model: MjModel, data: MjData, frame = 0) {
    if (!this.motion) return;
    const t = this.clampHoldFrame(frame);
    const xpos = asArray(data.xpos, model.nbody * 3);
    const xquat = asArray(data.xquat, model.nbody * 4);
    const bid = this.torsoBodyId;
    const robotPos = xpos.subarray(bid * 3, bid * 3 + 3);
    const robotQ = xquat.subarray(bid * 4, bid * 4 + 4);
    const f0Pos = this.anchorPosAt(t);
    const f0Quat = this.anchorQuatAt(t);
    // yaw(robot * inv(f0)) — matches KickMotionCommand._capture_align_to_robot
    quatMul(robotQ, quatInv(f0Quat, this.tmpQuat2), this.tmpQuat);
    yawQuat(this.tmpQuat, this.alignYaw);
    this.alignF0[0] = f0Pos[0];
    this.alignF0[1] = f0Pos[1];
    this.alignF0[2] = f0Pos[2];
    this.alignT[0] = robotPos[0];
    this.alignT[1] = robotPos[1];
    this.alignT[2] = f0Pos[2]; // keep motion height
    this.alignOn = true;
  }

  private anchorPosAt(t: number): Float32Array {
    const m = this.motion!;
    if (m.anchor_pos_w && m.anchor_pos_w.length >= (t + 1) * 3) {
      const o = t * 3;
      return new Float32Array(m.anchor_pos_w.slice(o, o + 3));
    }
    // Fallback: use root pos if older packs lack anchor_pos_w.
    return this.rootPosAt(t);
  }

  private anchorQuatAt(t: number): Float32Array {
    const m = this.motion!;
    const o = t * 4;
    return new Float32Array(m.anchor_quat_w.slice(o, o + 4));
  }

  private rootPosAt(t: number): Float32Array {
    const m = this.motion!;
    if (m.root_pos_w && m.root_pos_w.length >= (t + 1) * 3) {
      const o = t * 3;
      return new Float32Array(m.root_pos_w.slice(o, o + 3));
    }
    return new Float32Array([0, 0, 0.8]);
  }

  private rootQuatAt(t: number): Float32Array {
    const m = this.motion!;
    if (m.root_quat_w && m.root_quat_w.length >= (t + 1) * 4) {
      const o = t * 4;
      return new Float32Array(m.root_quat_w.slice(o, o + 4));
    }
    return this.anchorQuatAt(t);
  }

  private alignedAnchorQuat(t: number, out: Float32Array): Float32Array {
    const raw = this.anchorQuatAt(t);
    if (!this.alignOn) {
      out[0] = raw[0];
      out[1] = raw[1];
      out[2] = raw[2];
      out[3] = raw[3];
      return out;
    }
    return quatMul(this.alignYaw, raw, out);
  }

  private fillKickCommand(out: Float32Array) {
    const m = this.motion!;
    const map = this.motionJointMap!;
    const n = this.jointNames.length;
    const t = this.timeStep;
    const jp = m.joint_pos;
    const jv = m.joint_vel;
    const nj = this.motionNumJoints;
    for (let i = 0; i < n; i++) {
      const mi = map[i];
      out[i] = jp[t * nj + mi];
      out[n + i] = jv[t * nj + mi];
    }
  }

  /** After each policy step: advance / hold motion reference. */
  private advanceMotion(model: MjModel, data: MjData) {
    if (!this.motion) return;
    if (!this.clipPlaying) {
      this.timeStep = this.clampHoldFrame(this.holdFrame);
      this.captureAlign(model, data, this.timeStep);
      return;
    }
    this.timeStep += 1;
    if (this.timeStep >= this.motion.T) {
      this.holdAtSelectedFrame();
      this.captureAlign(model, data, this.timeStep);
    }
  }

  resetHistory() {
    this.stopBlend();
    this.lastAction.fill(0);
    this.episodeSteps = 0;
    const slot =
      this.obsLayout === "actor_plus_history"
        ? this.historyFrameDim
        : this.frameDim;
    this.history = Array.from(
      { length: this.historyLength },
      () => new Float64Array(slot),
    );
    if (this.playProfile === "kick") {
      this.holdAtSelectedFrame();
    }
  }

  /** Extract the proprio subset that forms one ``actor_history`` frame. */
  private extractHistoryFrame(actorFrame: Float64Array): Float64Array {
    const out = new Float64Array(this.historyFrameDim);
    let o = 0;
    for (const name of this.historyTerms) {
      const f = this.fields.find((x) => x.name === name);
      if (!f) continue;
      const [lo] = f.slice;
      const dim = f.dim || 0;
      for (let i = 0; i < dim; i++) out[o++] = actorFrame[lo + i];
    }
    return out;
  }

  applyDefaultPose(data: MjData) {
    // DAVIS shoot: explicit free-joint spawn (behind ball, face −x / goal).
    // mj_resetData alone is not enough if qpos0 was compiled from a stale XML.
    if (this.playProfile === "davis" && this.freeQposAdr >= 0) {
      const qa = this.freeQposAdr;
      writeNum(data.qpos, qa + 0, 3.2);
      writeNum(data.qpos, qa + 1, 0);
      writeNum(data.qpos, qa + 2, 0.59);
      // yaw=π → face −x (MuJoCo wxyz)
      writeNum(data.qpos, qa + 3, 0);
      writeNum(data.qpos, qa + 4, 0);
      writeNum(data.qpos, qa + 5, 0);
      writeNum(data.qpos, qa + 6, 1);
      for (let k = 0; k < 6; k++) writeNum(data.qvel, this.freeDofAdr + k, 0);
    }
    for (let i = 0; i < this.jointNames.length; i++) {
      writeNum(data.qpos, this.jointQposAdr[i], this.defaultPos[i]);
      writeNum(data.ctrl, this.actuatorIds[i], this.defaultPos[i]);
    }
    for (let i = 0; i < this.jointNames.length; i++) {
      writeNum(data.qvel, this.jointDofAdr[i], 0);
    }
  }

  /**
   * Episode start pose. ``reset_from_motion`` packs (getup) start from the
   * Hold Frame of the clip (root pose + joints), like mjlab play
   * ``sampling_mode="start"``; others use the standing default pose.
   */
  applyInitPose(data: MjData) {
    if (!this.manifest.reset_from_motion || !this.motion || this.freeQposAdr < 0) {
      this.applyDefaultPose(data);
      return;
    }
    const t = this.clampHoldFrame(this.holdFrame);
    const pos = this.rootPosAt(t);
    const quat = this.rootQuatAt(t);
    const qa = this.freeQposAdr;
    for (let k = 0; k < 3; k++) writeNum(data.qpos, qa + k, pos[k]);
    for (let k = 0; k < 4; k++) writeNum(data.qpos, qa + 3 + k, quat[k]);
    for (let k = 0; k < 6; k++) writeNum(data.qvel, this.freeDofAdr + k, 0);
    const map = this.motionJointMap!;
    const nj = this.motionNumJoints;
    const jp = this.motion.joint_pos;
    const jv = this.motion.joint_vel;
    for (let i = 0; i < this.jointNames.length; i++) {
      const q = jp[t * nj + map[i]];
      writeNum(data.qpos, this.jointQposAdr[i], q);
      writeNum(data.ctrl, this.actuatorIds[i], q);
      writeNum(data.qvel, this.jointDofAdr[i], jv[t * nj + map[i]]);
    }
  }

  /** Assemble one frame by walking alignment fields in order (Viser/deploy). */
  private buildFrame(model: MjModel, data: MjData): Float64Array {
    const scales = this.alignment.observations.actor.obs_scales || {};
    const bid = this.torsoBodyId;
    const xquat = asArray(data.xquat, model.nbody * 4);
    const qpos = asArray(data.qpos, model.nq);
    const qvel = asArray(data.qvel, model.nv);

    // Anchor / motion_anchor_ori_b still uses torso (manifest.torso_body).
    // base_ang_vel + projected_gravity match mjlab: pelvis IMU + root link.
    const quat = xquat.subarray(bid * 4, bid * 4 + 4);
    const sensordata = asArray(data.sensordata, model.nsensordata);
    const angBody = sensordata.subarray(this.gyroAdr, this.gyroAdr + 3);
    const rid = this.rootBodyId;
    const gravityB = quatApplyInverse(
      xquat.subarray(rid * 4, rid * 4 + 4),
      [0, 0, -1],
    );

    // Kick: refresh align while holding so Hold Frame tracks robot yaw/xy.
    if (this.playProfile === "kick" && this.motion && !this.clipPlaying) {
      this.timeStep = this.clampHoldFrame(this.holdFrame);
      this.captureAlign(model, data, this.timeStep);
    }

    const frame = new Float64Array(this.frameDim);
    for (const field of this.fields) {
      const [lo, hi] = field.slice;
      const scale = field.scale ?? scales[field.name] ?? 1.0;
      switch (field.name) {
        case "base_ang_vel": {
          const s = scales.base_ang_vel ?? scale;
          frame[lo] = s * angBody[0];
          frame[lo + 1] = s * angBody[1];
          frame[lo + 2] = s * angBody[2];
          break;
        }
        case "projected_gravity": {
          const s = scales.projected_gravity ?? scale;
          frame[lo] = s * gravityB[0];
          frame[lo + 1] = s * gravityB[1];
          frame[lo + 2] = s * gravityB[2];
          break;
        }
        case "command": {
          const s = scales.command ?? scale;
          if (this.playProfile === "kick" && this.kickCommandBuf) {
            this.fillKickCommand(this.kickCommandBuf);
            const dim = Math.min(field.dim || this.kickCommandBuf.length, hi - lo);
            for (let i = 0; i < dim; i++) {
              frame[lo + i] = s * this.kickCommandBuf[i];
            }
          } else {
            frame[lo] = s * this.command[0];
            frame[lo + 1] = s * this.command[1];
            frame[lo + 2] = s * this.command[2];
          }
          break;
        }
        case "motion_anchor_ori_b": {
          if (!this.motion) {
            throw new Error("motion_anchor_ori_b requires packed motion.json");
          }
          const s = scales.motion_anchor_ori_b ?? scale;
          // q_rel = inv(robot_anchor) * aligned_motion_anchor
          this.alignedAnchorQuat(this.timeStep, this.tmpQuat2);
          quatMul(quatInv(quat, this.tmpQuat), this.tmpQuat2, this.tmpQuat);
          matrixFromQuatFirst2Cols(this.tmpQuat, this.tmpOri6);
          for (let i = 0; i < 6; i++) {
            frame[lo + i] = s * this.tmpOri6[i];
          }
          break;
        }
        case "phase": {
          // velocity mdp.phase: sin/cos of episode time over period, zero when standing.
          const period = Number(field.params?.period ?? 0.6);
          const c = this.command;
          if (Math.hypot(c[0], c[1], c[2]) < 0.1) {
            frame[lo] = 0;
            frame[lo + 1] = 0;
          } else {
            const t = this.episodeSteps * this.manifest.policy_dt;
            const ph = ((t % period) / period) * 2 * Math.PI;
            frame[lo] = scale * Math.sin(ph);
            frame[lo + 1] = scale * Math.cos(ph);
          }
          break;
        }
        case "joint_pos": {
          const s = scales.joint_pos ?? scale;
          for (let i = 0; i < this.jointNames.length; i++) {
            frame[lo + i] = s * (qpos[this.jointQposAdr[i]] - this.defaultPos[i]);
          }
          break;
        }
        case "joint_vel": {
          const s = scales.joint_vel ?? scale;
          for (let i = 0; i < this.jointNames.length; i++) {
            frame[lo + i] = s * qvel[this.jointDofAdr[i]];
          }
          break;
        }
        case "actions": {
          const s = scales.actions ?? scale;
          for (let i = 0; i < this.jointNames.length; i++) {
            frame[lo + i] = s * this.lastAction[i];
          }
          break;
        }
        default:
          throw new Error(
            `Unsupported actor field "${field.name}" for profile=${this.playProfile} ` +
              `(task=${this.manifest.task_id}). Extend the field builder or pack a ` +
              `compatible layout.`,
          );
      }
      void hi;
    }
    return frame;
  }

  getJointQposAdr(): Int32Array {
    return this.jointQposAdr;
  }

  getActuatorIds(): Int32Array {
    return this.actuatorIds;
  }

  async step(model: MjModel, data: MjData): Promise<void> {
    if (this.blendTo) {
      this.stepBlend(data);
      return;
    }
    const frame = this.buildFrame(model, data);
    const histFrame =
      this.obsLayout === "actor_plus_history"
        ? this.extractHistoryFrame(frame)
        : frame;
    this.history.push(histFrame);
    while (this.history.length > this.historyLength) this.history.shift();
    while (this.history.length < this.historyLength) {
      this.history.unshift(new Float64Array(histFrame));
    }

    const obs = new Float32Array(this.obsDim);
    let offset = 0;
    if (this.obsLayout === "actor_plus_history") {
      // obs = [actor (A) | actor_history (H×F)], time-major oldest→newest.
      for (let i = 0; i < this.frameDim; i++) obs[offset++] = frame[i];
      for (const h of this.history) {
        for (let i = 0; i < this.historyFrameDim; i++) obs[offset++] = h[i];
      }
    } else {
      for (const h of this.history) {
        for (let i = 0; i < this.frameDim; i++) obs[offset++] = h[i];
      }
    }

    let result: ort.InferenceSession.OnnxValueMapType;
    if (this.dualInput) {
      if (!this.depthCam) {
        throw new Error("DAVIS pack requires depth camera — call setDepthCamera()");
      }
      const depth = this.depthCam.capture(model, data);
      const [dh, dw] = this.depthCam.getShape();
      const names = this.session.inputNames;
      const proprioName =
        names.find((n) => /proprio|obs/i.test(n)) || names[0];
      const depthName =
        names.find((n) => /depth/i.test(n)) || names[1] || names[0];
      result = await this.session.run({
        [proprioName]: new ort.Tensor("float32", obs, [1, this.obsDim]),
        [depthName]: new ort.Tensor("float32", depth, [1, 1, dh, dw]),
      });
    } else {
      const inputName = this.session.inputNames[0];
      const tensor = new ort.Tensor("float32", obs, [1, this.obsDim]);
      result = await this.session.run({ [inputName]: tensor });
    }
    const out = result[this.session.outputNames[0]].data as Float32Array;

    // Match Viser play: full action gain from step 0 (no soft-start).
    for (let i = 0; i < this.jointNames.length; i++) {
      this.lastAction[i] = out[i];
      writeNum(
        data.ctrl,
        this.actuatorIds[i],
        this.defaultPos[i] + this.lastAction[i] * this.actionScale[i],
      );
    }

    this.episodeSteps += 1;
    if (this.playProfile === "kick") {
      this.advanceMotion(model, data);
    }
  }
}

/** @deprecated use PolicyController */
export { PolicyController as AmpPolicyController };
