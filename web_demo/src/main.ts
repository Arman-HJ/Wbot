import loadMujoco, { type MjData, type MjModel } from "mujoco-js";
import { BodyDragController } from "./drag";
import { GamepadInput, type PadAxes } from "./gamepad";
import { loadPackScene, type MujocoModule } from "./mujoco_util";
import { MotionVisualizer } from "./motion_vis";
import { PolicyController } from "./policy";
import { ImuPanel } from "./imu_panel";
import { JointMeter } from "./joint_meter";
import { RewardMeter } from "./rewards";
import { SceneRenderer } from "./render";
import { DavisDepthCamera } from "./depth";
import type {
  Alignment,
  CommandLimits,
  Manifest,
  MotionVisMode,
  PackIndex,
  PlayProfile,
} from "./types";
import { name2id } from "./mujoco_util";

const statusEl = document.getElementById("status")!;
const packSelect = document.getElementById("packSelect") as HTMLSelectElement;
const ckptSelect = document.getElementById("ckptSelect") as HTMLSelectElement;
const profileBadge = document.getElementById("profileBadge")!;
const panelTwist = document.getElementById("panelTwist")!;
const panelKick = document.getElementById("panelKick")!;
const btnPlayPause = document.getElementById("btnPlayPause") as HTMLButtonElement;
const btnReset = document.getElementById("btnReset") as HTMLButtonElement;
const btnZero = document.getElementById("btnZero") as HTMLButtonElement;
const speedSelect = document.getElementById("speedSelect") as HTMLSelectElement;
const cmdEnable = document.getElementById("cmdEnable") as HTMLInputElement;
const cmdVx = document.getElementById("cmdVx") as HTMLInputElement;
const cmdVy = document.getElementById("cmdVy") as HTMLInputElement;
const cmdWz = document.getElementById("cmdWz") as HTMLInputElement;
const cmdVxRange = document.getElementById("cmdVxRange")!;
const cmdReadout = document.getElementById("cmdReadout")!;
const padGain = document.getElementById("padGain") as HTMLInputElement;
const dragGain = document.getElementById("dragGain") as HTMLInputElement;
const pushDir = document.getElementById("pushDir") as HTMLSelectElement;
const pushForce = document.getElementById("pushForce") as HTMLInputElement;
const pushDur = document.getElementById("pushDur") as HTMLInputElement;
const btnPush = document.getElementById("btnPush") as HTMLButtonElement;
const btnCancelPush = document.getElementById("btnCancelPush") as HTMLButtonElement;
const pushStatus = document.getElementById("pushStatus")!;
const btnPlayClip = document.getElementById("btnPlayClip") as HTMLButtonElement;
const btnBlendInit = document.getElementById("btnBlendInit") as HTMLButtonElement;
const blendDur = document.getElementById("blendDur") as HTMLInputElement;
const holdFrameInput = document.getElementById("holdFrame") as HTMLInputElement;
const holdFrameVal = document.getElementById("holdFrameVal")!;
const clipStatus = document.getElementById("clipStatus")!;
const showMotion = document.getElementById("showMotion") as HTMLInputElement;
const motionVisMode = document.getElementById(
  "motionVisMode",
) as HTMLSelectElement;
const rewardBars = document.getElementById("rewardBars")!;
const rewardTotal = document.getElementById("rewardTotal")!;
const jointTable = document.getElementById("jointTable")!;
const jointSum = document.getElementById("jointSum")!;
const tabBtnControl = document.getElementById("tabBtnControl") as HTMLButtonElement;
const tabBtnJoints = document.getElementById("tabBtnJoints") as HTMLButtonElement;
const tabBtnImu = document.getElementById("tabBtnImu") as HTMLButtonElement;
const tabBtnRewards = document.getElementById("tabBtnRewards") as HTMLButtonElement;
const tabControl = document.getElementById("tabControl")!;
const tabJoints = document.getElementById("tabJoints")!;
const tabImu = document.getElementById("tabImu")!;
const tabRewards = document.getElementById("tabRewards")!;
const imuPanelEl = document.getElementById("imuPanel")!;
const canvas = document.getElementById("viewport") as HTMLCanvasElement;
const depthHud = document.getElementById("depthHud") as HTMLDivElement;
const depthPreview = document.getElementById("depthPreview") as HTMLCanvasElement;
const depthMetaEl = document.getElementById("depthMeta")!;

function setStatus(msg: string, error = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", error);
}

/** Local ForcePulse equivalent for KickViserPlayViewer. */
type PushState = {
  active: boolean;
  bodyId: number;
  forceW: [number, number, number];
  endTime: number;
};

type AppState = {
  mj: MujocoModule;
  model: MjModel | null;
  data: MjData | null;
  policy: PolicyController | null;
  packId: string | null;
  playProfile: PlayProfile | null;
  playing: boolean;
  speed: number;
  simSteps: number;
  accumulator: number;
  lastFrameMs: number;
  keys: Set<string>;
  push: PushState | null;
};

const renderer = new SceneRenderer(canvas);
const motionVis = new MotionVisualizer(renderer.scene);
let depthCam: DavisDepthCamera | null = null;

function setDepthHudVisible(on: boolean) {
  depthHud.classList.toggle("visible", on);
}

function paintDepthHud() {
  if (!depthCam) return;
  depthCam.paintPreview(depthPreview);
}
const rewardMeter = new RewardMeter();
rewardMeter.bindUi(rewardBars, rewardTotal);
const jointMeter = new JointMeter();
jointMeter.bindUi(jointTable, jointSum);
jointMeter.enabled = false;
const imuPanel = new ImuPanel();
imuPanel.bindUi(imuPanelEl);
imuPanel.enabled = false;
const gamepad = new GamepadInput({
  source: document.getElementById("padSource") as HTMLSelectElement,
  enable: document.getElementById("padEnable") as HTMLButtonElement,
  stop: document.getElementById("padStop") as HTMLButtonElement,
  left: document.getElementById("padLeft")!,
  right: document.getElementById("padRight")!,
  status: document.getElementById("padStatus")!,
});
let padWasActive = false;

type HudTab = "control" | "joints" | "imu" | "rewards";

function setHudTab(which: HudTab) {
  tabControl.hidden = which !== "control";
  tabJoints.hidden = which !== "joints";
  tabImu.hidden = which !== "imu";
  tabRewards.hidden = which !== "rewards";
  tabBtnControl.classList.toggle("active", which === "control");
  tabBtnJoints.classList.toggle("active", which === "joints");
  tabBtnImu.classList.toggle("active", which === "imu");
  tabBtnRewards.classList.toggle("active", which === "rewards");
  jointMeter.enabled = which === "joints";
  imuPanel.enabled = which === "imu";
  rewardMeter.visible = which === "rewards";
  rewardMeter.refresh();
}

tabBtnControl.addEventListener("click", () => setHudTab("control"));
tabBtnJoints.addEventListener("click", () => setHudTab("joints"));
tabBtnImu.addEventListener("click", () => setHudTab("imu"));
tabBtnRewards.addEventListener("click", () => setHudTab("rewards"));
const drag = new BodyDragController(
  canvas,
  renderer.camera,
  renderer.controls,
  renderer.scene,
);

const state: AppState = {
  mj: null as unknown as MujocoModule,
  model: null,
  data: null,
  policy: null,
  packId: null,
  playProfile: null,
  playing: true,
  speed: 1,
  simSteps: 0,
  accumulator: 0,
  lastFrameMs: performance.now(),
  keys: new Set(),
  push: null,
};

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function readSliders(): { vx: number; vy: number; wz: number } {
  return {
    vx: Number(cmdVx.value),
    vy: Number(cmdVy.value),
    wz: Number(cmdWz.value),
  };
}

function writeSliders(vx: number, vy: number, wz: number) {
  cmdVx.value = String(clamp(vx, Number(cmdVx.min), Number(cmdVx.max)));
  cmdVy.value = String(clamp(vy, Number(cmdVy.min), Number(cmdVy.max)));
  cmdWz.value = String(clamp(wz, Number(cmdWz.min), Number(cmdWz.max)));
}

const DEFAULT_COMMAND_LIMITS: CommandLimits = {
  lin_vel_x: [-1.5, 1.5],
  lin_vel_y: [-0.8, 0.8],
  ang_vel_z: [-1.5, 1.5],
};

function configureTwistSliders(limits: CommandLimits | undefined) {
  const l = limits ?? DEFAULT_COMMAND_LIMITS;
  const set = (el: HTMLInputElement, [lo, hi]: [number, number]) => {
    el.min = String(lo);
    el.max = String(hi);
  };
  set(cmdVx, l.lin_vel_x);
  set(cmdVy, l.lin_vel_y);
  set(cmdWz, l.ang_vel_z);
  cmdVxRange.textContent = `[${l.lin_vel_x[0]}, ${l.lin_vel_x[1]}] m/s`;
}

function applyProfileUi(profile: PlayProfile, hasVelCmd: boolean, manifest: Manifest) {
  state.playProfile = profile;
  profileBadge.textContent =
    `profile=${profile}` +
    (hasVelCmd ? " · twist cmd" : "") +
    (profile === "davis" ? " · depth" : "");
  // Viser: Commands folder only when velocity command term exists.
  panelTwist.hidden = !(
    (profile === "amp_twist" || profile === "velocity") && hasVelCmd
  );
  configureTwistSliders(manifest.command_limits);
  panelKick.hidden = profile !== "kick";
  // Viser UniformVelocityCommand: Enable starts false.
  cmdEnable.checked = false;
  writeSliders(0, 0, 0);
  state.policy?.setCommand(0, 0, 0);
  gamepad.disable("切换 pack，手柄已关闭");
  cancelPush("切换 pack，推力已清零");
  updateClipUi();
}

function syncHoldFrameUi() {
  if (!state.policy || !state.policy.hasMotion()) {
    holdFrameInput.disabled = true;
    holdFrameInput.max = "0";
    holdFrameInput.value = "0";
    holdFrameVal.textContent = "/ 0";
    return;
  }
  const maxF = Math.max(0, state.policy.getMotionLength() - 1);
  holdFrameInput.max = String(maxF);
  const f = state.policy.getHoldFrame();
  if (document.activeElement !== holdFrameInput) {
    holdFrameInput.value = String(f);
  }
  holdFrameVal.textContent = `/ ${maxF}`;
  holdFrameInput.disabled = state.policy.isClipPlaying();
}

function updateClipUi() {
  btnBlendInit.disabled = !state.policy;
  if (!state.policy || state.playProfile !== "kick") {
    clipStatus.textContent = "";
    btnPlayClip.disabled = true;
    holdFrameInput.disabled = true;
    showMotion.disabled = true;
    motionVisMode.disabled = true;
    motionVis.setEnabled(false);
    return;
  }
  clipStatus.textContent = state.policy.motionStatus();
  btnPlayClip.disabled = state.policy.isClipPlaying();
  syncHoldFrameUi();
  const has = state.policy.hasMotion();
  showMotion.disabled = !has;
  motionVisMode.disabled = !has || !showMotion.checked;
  motionVis.setEnabled(has && showMotion.checked);
  motionVis.setMode(motionVisMode.value as MotionVisMode);
}

function togglePlaying() {
  state.playing = !state.playing;
  btnPlayPause.textContent = state.playing ? "Pause" : "Play";
}

/** Handle pad buttons; returns the twist (m/s, rad/s) while the pad drives the command. */
function pollGamepad(): PadAxes | null {
  const { axes, events } = gamepad.poll();
  for (const ev of events) {
    if (ev === "stop") {
      writeSliders(0, 0, 0);
      state.policy?.setCommand(0, 0, 0);
    } else if (ev === "reset") {
      resetSim();
    } else if (ev === "pause") {
      togglePlaying();
    }
  }
  if (!axes || panelTwist.hidden) {
    if (padWasActive) writeSliders(0, 0, 0);
    padWasActive = false;
    return null;
  }
  padWasActive = true;
  cmdEnable.checked = true;
  const gain = Number(padGain.value);
  // Full stick deflection = slider limit on that side.
  const scale = (a: number, el: HTMLInputElement) =>
    gain * (a >= 0 ? a * Number(el.max) : -a * Number(el.min));
  return [scale(axes[0], cmdVx), scale(axes[1], cmdVy), scale(axes[2], cmdWz)];
}

function updateCommandFromInput(padTwist: PadAxes | null) {
  if (!state.policy) return;
  const twistProfile =
    state.playProfile === "amp_twist" || state.playProfile === "velocity";
  if (!twistProfile || !state.policy.hasVelocityCommand) {
    state.policy.setCommand(0, 0, 0);
    return;
  }
  // Match Viser: only apply joystick when Enable is on.
  if (!cmdEnable.checked) {
    state.policy.setCommand(0, 0, 0);
    return;
  }
  if (padTwist) {
    const [vx, vy, wz] = padTwist;
    writeSliders(vx, vy, wz);
    state.policy.setCommand(vx, vy, wz);
    return;
  }
  let { vx, vy, wz } = readSliders();
  const stepLin = 0.05;
  const stepAng = 0.05;
  const keyed = (el: HTMLInputElement, v: number) =>
    clamp(v, Number(el.min), Number(el.max));
  if (state.keys.has("KeyW") || state.keys.has("ArrowUp")) vx = keyed(cmdVx, vx + stepLin);
  if (state.keys.has("KeyS") || state.keys.has("ArrowDown")) vx = keyed(cmdVx, vx - stepLin);
  if (state.keys.has("KeyA") || state.keys.has("ArrowLeft")) vy = keyed(cmdVy, vy + stepLin);
  if (state.keys.has("KeyD") || state.keys.has("ArrowRight")) vy = keyed(cmdVy, vy - stepLin);
  if (state.keys.has("KeyQ")) wz = keyed(cmdWz, wz + stepAng);
  if (state.keys.has("KeyE")) wz = keyed(cmdWz, wz - stepAng);
  if (
    state.keys.has("KeyW") ||
    state.keys.has("KeyS") ||
    state.keys.has("KeyA") ||
    state.keys.has("KeyD") ||
    state.keys.has("KeyQ") ||
    state.keys.has("KeyE") ||
    state.keys.has("ArrowUp") ||
    state.keys.has("ArrowDown") ||
    state.keys.has("ArrowLeft") ||
    state.keys.has("ArrowRight")
  ) {
    writeSliders(vx, vy, wz);
  }
  state.policy.setCommand(vx, vy, wz);
}

function yawFromQuatWxyz(w: number, x: number, y: number, z: number): number {
  // yaw about +Z
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

/** Commanded vs measured heading-frame speed (free joint qvel is world-frame). */
function updateCmdReadout() {
  if (panelTwist.hidden || !state.data || !state.policy) return;
  const c = state.policy.getCommand();
  const q = state.data.qpos;
  const v = state.data.qvel;
  const yaw = yawFromQuatWxyz(Number(q[3]), Number(q[4]), Number(q[5]), Number(q[6]));
  const vxW = Number(v[0]);
  const vyW = Number(v[1]);
  const fwd = Math.cos(yaw) * vxW + Math.sin(yaw) * vyW;
  const lat = -Math.sin(yaw) * vxW + Math.cos(yaw) * vyW;
  cmdReadout.textContent =
    `cmd  vx=${c[0].toFixed(2)} vy=${c[1].toFixed(2)} wz=${c[2].toFixed(2)}\n` +
    `实际 vx=${fwd.toFixed(2)} vy=${lat.toFixed(2)} wz=${Number(v[5]).toFixed(2)}`;
}

function startPush() {
  if (!state.mj || !state.model || !state.data || !state.policy) return;
  const torso =
    state.policy.manifest.torso_body || state.policy.alignment.bodies?.torso || "torso_link";
  const bodyId = name2id(state.mj, state.model, "body", torso);
  if (bodyId < 0) {
    pushStatus.textContent = `找不到 body ${torso}`;
    return;
  }
  const force = Number(pushForce.value);
  const dur = Number(pushDur.value);
  const nbody = state.model.nbody;
  const xquat = state.data.xquat;
  const w = Number(xquat[bodyId * 4]);
  const x = Number(xquat[bodyId * 4 + 1]);
  const y = Number(xquat[bodyId * 4 + 2]);
  const z = Number(xquat[bodyId * 4 + 3]);
  const yaw = yawFromQuatWxyz(w, x, y, z);
  const fwd: [number, number] = [Math.cos(yaw), Math.sin(yaw)];
  const left: [number, number] = [-fwd[1], fwd[0]];
  let fx = 0;
  let fy = 0;
  let fz = 0;
  switch (pushDir.value) {
    case "向前":
      fx = fwd[0] * force;
      fy = fwd[1] * force;
      break;
    case "向后":
      fx = -fwd[0] * force;
      fy = -fwd[1] * force;
      break;
    case "向左":
      fx = left[0] * force;
      fy = left[1] * force;
      break;
    case "向右":
      fx = -left[0] * force;
      fy = -left[1] * force;
      break;
    case "向下":
      fz = -force;
      break;
  }
  void nbody;
  state.push = {
    active: true,
    bodyId,
    forceW: [fx, fy, fz],
    endTime: state.data.time + dur,
  };
  pushStatus.textContent = `施力中 ${pushDir.value} ${force}N × ${dur}s → ${torso}`;
}

function cancelPush(msg = "推力已清零") {
  if (state.push && state.model && state.data) {
    const bid = state.push.bodyId;
    for (let i = 0; i < 6; i++) state.data.xfrc_applied[bid * 6 + i] = 0;
  }
  state.push = null;
  pushStatus.textContent = msg;
}

function applyPush(data: MjData) {
  if (!state.push?.active || !state.model) return;
  if (data.time >= state.push.endTime) {
    cancelPush("推力结束，已清零");
    return;
  }
  const bid = state.push.bodyId;
  const [fx, fy, fz] = state.push.forceW;
  // Clear then set (drag may also write; push takes priority this tick after drag)
  for (let i = 0; i < 6; i++) data.xfrc_applied[bid * 6 + i] = 0;
  data.xfrc_applied[bid * 6 + 0] = fx;
  data.xfrc_applied[bid * 6 + 1] = fy;
  data.xfrc_applied[bid * 6 + 2] = fz;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  const ct = res.headers.get("content-type") || "";
  if (!res.ok || !ct.includes("json")) {
    const peek = (await res.text()).slice(0, 80);
    throw new Error(
      `Failed to load ${url} (HTTP ${res.status}, ${ct || "no content-type"}): ${peek}`,
    );
  }
  return res.json() as Promise<T>;
}

function fillCheckpointSelect(manifest: Manifest, keepSelection?: string | null) {
  const prev = keepSelection ?? ckptSelect.value;
  ckptSelect.innerHTML = "";
  const entries = manifest.checkpoints?.length
    ? manifest.checkpoints
    : [{ name: manifest.files.policy, step: null, file: manifest.files.policy }];
  for (const c of entries) {
    const opt = document.createElement("option");
    opt.value = c.name;
    opt.textContent =
      c.step != null ? `${c.name}  (iter ${c.step})` : c.name;
    ckptSelect.appendChild(opt);
  }
  const names = new Set(entries.map((c) => c.name));
  if (prev && names.has(prev)) {
    ckptSelect.value = prev;
  } else {
    const def =
      manifest.default_checkpoint ||
      entries[entries.length - 1]?.name ||
      entries[0]?.name;
    if (def) ckptSelect.value = def;
  }
}

/** Poll pack manifest for newly exported model_*.pt → ONNX (server-side watch). */
async function refreshCheckpointList() {
  if (!state.packId || !state.policy) return;
  try {
    const manifest = await fetchJson<Manifest>(
      `/packs/${state.packId}/manifest.json?t=${Date.now()}`,
    );
    const entries = manifest.checkpoints || [];
    const nextKey = entries.map((c) => c.name).join("|");
    const prevKey = [...ckptSelect.options].map((o) => o.value).join("|");
    if (nextKey === prevKey) return;
    const before = ckptSelect.value;
    state.policy.manifest.checkpoints = entries;
    state.policy.manifest.default_checkpoint = manifest.default_checkpoint;
    fillCheckpointSelect(manifest, before);
    const n = entries.length;
    const latest = entries[entries.length - 1]?.name || "?";
    console.info(`[web] checkpoints updated: ${n} available (latest ${latest})`);
    setStatus(
      `Ready: ${state.packId}\n` +
        `task=${manifest.task_id || "?"} profile=${state.policy.playProfile} ` +
        `robot=${manifest.robot_xml || "?"}\n` +
        `ckpt=${state.policy.getCurrentCheckpoint()} (${n} available)\n` +
        `obs=${manifest.obs_dim} hist=${manifest.history_length} ` +
        `act=${manifest.num_actions} dt=${manifest.policy_dt}s`,
    );
  } catch {
    /* pack may be mid-write */
  }
}

async function loadCheckpointOnly(name: string) {
  if (!state.policy) return;
  setStatus(`Loading checkpoint ${name}…`);
  await state.policy.loadCheckpoint(name);
  await rewardMeter.syncCheckpoint(state.policy).catch((e) =>
    console.warn("[web] discriminator load failed:", e),
  );
  state.policy.resetHistory();
  if (state.model && state.data) {
    state.policy.applyInitPose(state.data);
    state.mj.mj_forward(state.model, state.data);
    renderer.sync(state.data);
  }
  state.simSteps = 0;
  state.accumulator = 0;
  updateClipUi();
  setStatus(
    `Checkpoint ${name}\n` +
      `pack=${state.packId} profile=${state.policy.playProfile}`,
  );
}

async function loadPack(packId: string) {
  setStatus(`Loading pack ${packId}…`);
  const packUrl = `/packs/${packId}`;
  const [manifest, alignment] = await Promise.all([
    fetchJson<Manifest>(`${packUrl}/manifest.json`),
    fetchJson<Alignment>(`${packUrl}/deployment_alignment.json`),
  ]);

  state.data?.delete();
  state.model?.delete();
  state.data = null;
  state.model = null;
  state.policy = null;
  depthCam?.dispose();
  depthCam = null;
  setDepthHudVisible(false);

  const { model, data } = await loadPackScene(state.mj, packUrl);
  const policy = new PolicyController(manifest, alignment);
  const depthMeta =
    manifest.depth || alignment.observations?.actor?.depth || undefined;
  if (
    policy.playProfile === "davis" ||
    (manifest.obs_inputs || []).includes("depth")
  ) {
    depthCam = new DavisDepthCamera(
      renderer.scene,
      renderer.renderer,
      depthMeta,
    );
    depthCam.bindHead(
      state.mj,
      model,
      depthMeta?.head_body || "head_camera_link",
    );
    depthCam.bindSceneBodies(state.mj, model);
    depthCam.ensureFrustum(renderer.scene);
    policy.setDepthCamera(depthCam);
    const [dh, dw] = depthCam.getShape();
    depthMetaEl.textContent = `${dh}×${dw}`;
    setDepthHudVisible(true);
  }
  await policy.init(state.mj, model, packUrl);

  state.model = model;
  state.data = data;
  state.policy = policy;
  state.packId = packId;
  state.simSteps = 0;
  state.accumulator = 0;

  fillCheckpointSelect(manifest);
  applyProfileUi(policy.playProfile, policy.hasVelocityCommand, manifest);

  // DAVIS: start paused so standing pose is visible; user hits Play.
  if (policy.playProfile === "davis") {
    state.playing = false;
    btnPlayPause.textContent = "Play";
  }

  policy.applyInitPose(data);
  state.mj.mj_forward(model, data);

  renderer.buildFromModel(model);
  drag.bind(model, renderer.getBodyMeshes());
  motionVis.rebuild(state.mj, model, renderer.getBodyGroups());
  imuPanel.bindModel(state.mj, model, {
    rootBody: manifest.root_body || "pelvis",
  });
  await rewardMeter.load(
    state.mj,
    model,
    packUrl,
    manifest.files.rewards,
    manifest.policy_dt,
    policy,
  );
  rewardMeter.reset();
  renderer.sync(data);
  if (policy.hasMotion()) {
    motionVis.sync(policy, data);
    rewardMeter.update(policy, data);
  }
  if (depthCam) {
    depthCam.capture(model, data);
    paintDepthHud();
  }

  const nCkpt = manifest.checkpoints?.length || 1;
  setStatus(
    `Ready: ${packId}\n` +
      `task=${manifest.task_id || "?"} profile=${policy.playProfile} ` +
      `robot=${manifest.robot_xml || "?"}\n` +
      `ckpt=${policy.getCurrentCheckpoint()} (${nCkpt} available)\n` +
      `obs=${manifest.obs_dim} hist=${manifest.history_length} ` +
      `act=${manifest.num_actions} dt=${manifest.policy_dt}s` +
      (depthCam
        ? `\ndepth: zed2i on ${depthCam.getExtrinsics().parent} ` +
          `pos=${depthCam.getExtrinsics().pos.map((x) => x.toFixed(2)).join(",")} ` +
          `fovy=${depthCam.getExtrinsics().fovy}° (follows neck)`
        : ""),
  );
}

function resetSim() {
  if (!state.mj || !state.model || !state.data || !state.policy) return;
  cancelPush("Reset，推力已清零");
  state.mj.mj_resetData(state.model, state.data);
  state.policy.applyInitPose(state.data);
  state.mj.mj_forward(state.model, state.data);
  state.policy.resetHistory();
  state.policy.setCommand(0, 0, 0);
  writeSliders(0, 0, 0);
  cmdEnable.checked = false;
  state.simSteps = 0;
  state.accumulator = 0;
  renderer.sync(state.data);
  if (state.policy.hasMotion()) {
    motionVis.sync(state.policy, state.data);
  }
  rewardMeter.reset();
  if (state.policy.hasMotion()) {
    rewardMeter.update(state.policy, state.data);
  }
  if (depthCam && state.model) {
    depthCam.capture(state.model, state.data);
    paintDepthHud();
  }
  updateClipUi();
  setStatus(`Reset: ${state.packId}`);
}

let loopBusy = false;

async function frame(nowMs: number) {
  requestAnimationFrame(frame);
  if (loopBusy) return;
  loopBusy = true;
  try {
    const dtWall = Math.min(0.05, (nowMs - state.lastFrameMs) / 1000);
    state.lastFrameMs = nowMs;
    updateCommandFromInput(pollGamepad());
    drag.gain = Number(dragGain.value);

    if (state.playing && state.model && state.data && state.policy) {
      const simDt = state.policy.manifest.sim_dt;
      const decimation = state.policy.manifest.decimation;
      state.accumulator += dtWall * state.speed;
      const maxCatchUp = simDt * 40;
      if (state.accumulator > maxCatchUp) state.accumulator = maxCatchUp;

      while (state.accumulator >= simDt) {
        if (state.simSteps % decimation === 0) {
          await state.policy.step(state.model, state.data);
          if (state.playProfile === "kick") updateClipUi();
          rewardMeter.update(state.policy, state.data);
          if (depthCam) paintDepthHud();
        }
        drag.apply(state.data);
        if (state.playProfile === "kick") applyPush(state.data);
        state.mj.mj_step(state.model, state.data);
        state.simSteps += 1;
        state.accumulator -= simDt;
      }
      renderer.sync(state.data);
      if (state.policy.hasMotion()) {
        motionVis.sync(state.policy, state.data);
      }
      jointMeter.update(state.policy, state.model, state.data, nowMs);
      imuPanel.update(state.model, state.data, nowMs);
      rewardMeter.tick(nowMs);
      updateCmdReadout();
    } else if (state.data) {
      drag.apply(state.data);
      renderer.sync(state.data);
      if (state.policy?.hasMotion()) {
        motionVis.sync(state.policy, state.data);
      }
      if (state.policy && state.model) {
        jointMeter.update(state.policy, state.model, state.data, nowMs);
        imuPanel.update(state.model, state.data, nowMs);
      }
    }
    renderer.render();
  } catch (err) {
    console.error(err);
    setStatus(`Runtime error: ${err}`, true);
    state.playing = false;
    btnPlayPause.textContent = "Play";
  } finally {
    loopBusy = false;
  }
}

function wireUi(packs: PackIndex) {
  packSelect.innerHTML = "";
  for (const p of packs.packs) {
    const opt = document.createElement("option");
    opt.value = p.id;
    const tag = [p.play_profile, p.task_id].filter(Boolean).join(" · ");
    opt.textContent = tag ? `${p.id} (${tag})` : p.id;
    packSelect.appendChild(opt);
  }

  packSelect.addEventListener("change", () => {
    void loadPack(packSelect.value).catch((e) => setStatus(String(e), true));
  });

  ckptSelect.addEventListener("change", () => {
    void loadCheckpointOnly(ckptSelect.value).catch((e) =>
      setStatus(String(e), true),
    );
  });

  btnPlayPause.addEventListener("click", () => togglePlaying());

  btnReset.addEventListener("click", () => resetSim());

  btnZero.addEventListener("click", () => {
    writeSliders(0, 0, 0);
    state.policy?.setCommand(0, 0, 0);
  });

  btnPush.addEventListener("click", () => startPush());
  btnCancelPush.addEventListener("click", () => cancelPush());

  btnPlayClip.addEventListener("click", () => {
    if (!state.model || !state.data || !state.policy) return;
    if (state.policy.startPlayClip(state.model, state.data)) {
      updateClipUi();
    }
  });

  btnBlendInit.addEventListener("click", () => {
    if (!state.model || !state.data || !state.policy) return;
    cancelPush("插值到初始位，推力已清零");
    state.policy.startBlendToInit(state.model, state.data, Number(blendDur.value));
    if (!state.playing) {
      state.playing = true;
      btnPlayPause.textContent = "Pause";
    }
    updateClipUi();
  });

  const applyHoldFrameInput = () => {
    if (!state.policy || state.policy.isClipPlaying()) return;
    const raw = Number(holdFrameInput.value);
    if (!Number.isFinite(raw)) return;
    state.policy.setHoldFrame(raw);
    const f = state.policy.getHoldFrame();
    holdFrameInput.value = String(f);
    holdFrameVal.textContent = `/ ${Math.max(0, state.policy.getMotionLength() - 1)}`;
    clipStatus.textContent = state.policy.motionStatus();
  };
  holdFrameInput.addEventListener("change", applyHoldFrameInput);
  holdFrameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      applyHoldFrameInput();
      holdFrameInput.blur();
    }
  });

  showMotion.addEventListener("change", () => {
    updateClipUi();
  });
  motionVisMode.addEventListener("change", () => {
    updateClipUi();
  });

  speedSelect.addEventListener("change", () => {
    state.speed = Number(speedSelect.value);
  });

  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) {
      return;
    }
    state.keys.add(e.code);
    if (e.code === "Space") {
      e.preventDefault();
      togglePlaying();
    }
    if (e.code === "KeyR" && !e.ctrlKey && !e.metaKey) {
      resetSim();
    }
  });
  window.addEventListener("keyup", (e) => {
    state.keys.delete(e.code);
  });
}

async function main() {
  setStatus("Booting MuJoCo WASM…");
  state.mj = await loadMujoco();
  try {
    state.mj.FS.mkdir("/working");
  } catch {
    /* exists */
  }
  try {
    state.mj.FS.mount(state.mj.MEMFS, { root: "." }, "/working");
  } catch {
    /* already */
  }

  setStatus("Fetching pack index…");
  const index = await fetchJson<PackIndex>("/packs/index.json");

  if (!index.packs?.length) {
    setStatus(
      "No packs. Example:\n" +
        "python scripts/pack_web_demo.py --run logs/rsl_rl/.../<run_dir>",
      true,
    );
    return;
  }
  wireUi(index);
  packSelect.value = index.packs[0].id;
  await loadPack(index.packs[0].id);
  // Backend exports new model_*.pt → ONNX every ~8s; refresh dropdown here.
  window.setInterval(() => {
    void refreshCheckpointList();
  }, 5000);
  state.lastFrameMs = performance.now();
  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  setStatus(`Init failed: ${err}`, true);
});
