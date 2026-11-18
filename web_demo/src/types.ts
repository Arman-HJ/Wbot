export type FieldSpec = {
  name: string;
  dim: number;
  slice: [number, number];
  scale: number;
  params?: Record<string, unknown>;
};

export type PlayProfile = "amp_twist" | "kick" | "velocity" | "davis";

export type DepthMeta = {
  height?: number;
  width?: number;
  render_height?: number;
  render_width?: number;
  fovy_deg?: number;
  head_body?: string;
  /** Local offset on ``head_body`` (training CameraSensorCfg.pos). */
  cam_pos?: [number, number, number] | number[];
  /** Local cam quat wxyz (training CameraSensorCfg.quat); MuJoCo look −Z. */
  cam_quat_wxyz?: [number, number, number, number] | number[];
  depth_min_m?: number;
  depth_max_m?: number;
};

export type CommandLimits = {
  lin_vel_x: [number, number];
  lin_vel_y: [number, number];
  ang_vel_z: [number, number];
};

export type MotionMeta = {
  fps: number;
  T: number;
  anchor_body: string;
};

/** Packed kick motion (flat arrays for browser ghost / policy command). */
export type MotionData = {
  fps: number;
  T: number;
  joint_names: string[];
  anchor_body: string;
  anchor_body_index: number;
  root_body?: string;
  root_body_index?: number;
  joint_pos: number[];
  joint_vel: number[];
  anchor_pos_w?: number[];
  anchor_quat_w: number[];
  root_pos_w?: number[];
  root_quat_w?: number[];
};

export type MotionVisMode = "ghost" | "frames";

export type RewardTermKind =
  | "anchor_pos"
  | "anchor_ori"
  | "body_pos"
  | "body_ori"
  | "body_lin_vel"
  | "body_ang_vel"
  | "foot_pos_z"
  | "action_rate"
  | "joint_limit";

export type RewardTermSpec = {
  name: string;
  kind: RewardTermKind;
  weight: number;
  std?: number | Record<string, number>;
  body_names?: string[];
};

export type RewardConfig = {
  anchor_body: string;
  terms: RewardTermSpec[];
};

export type VelocityRewardKind =
  | "track_lin_vel"
  | "track_ang_vel"
  | "body_orientation"
  | "variable_posture"
  | "body_ang_vel_xy"
  | "angular_momentum"
  | "joint_acc"
  | "joint_pos_limits"
  | "action_rate"
  | "feet_gait"
  | "feet_clearance"
  | "feet_slip"
  | "soft_landing"
  | "stand_still"
  | "self_collision"
  | "unsupported";

export type VelocityRewardTerm = {
  name: string;
  kind: VelocityRewardKind;
  weight: number;
  std?: number;
  command_threshold?: number;
  target_height?: number;
  period?: number;
  threshold?: number;
  offset?: number[];
  force_threshold?: number;
  walking_threshold?: number;
  running_threshold?: number;
  body?: string;
  site_names?: string[];
  std_standing?: Record<string, number>;
  std_walking?: Record<string, number>;
  std_running?: Record<string, number>;
  soft_factor?: number;
  contact?: { pattern: string; mode: "subtree" | "body"; history_length: number };
};

export type VelocityRewardConfig = {
  profile: "velocity";
  root_body: string;
  terms: VelocityRewardTerm[];
};

/** ``src.tasks.amp.mdp.rewards`` terms (see ``pack_web_demo._AMP_REWARD_KINDS``). */
export type AmpRewardTerm = {
  name: string;
  kind: string;
  weight: number;
  /** Scalar params from env.yaml; unset ones fall back to the Python defaults. */
  params?: Record<string, number | boolean>;
  joints?: string[];
  body?: string;
  feet?: string[];
  soft_factor?: number;
  contact?: { pattern: string; mode: "subtree" | "body"; history_length: number };
};

export type AmpTurnProfile = { left: number[]; right: number[]; period_s: number };

export type AmpRewardConfig = {
  profile: "amp";
  root_body: string;
  terms: AmpRewardTerm[];
  turn_profiles?: { left_turn: AmpTurnProfile; right_turn: AmpTurnProfile };
};

/** ``pack_web_demo._amp_disc_spec``: discriminator layout shared by all ``*.disc.bin``. */
export type AmpDiscSpec = {
  type: string;
  amp_obs_dim: number;
  joint_names: string[];
  key_body_names: string[];
  label_names: string[];
  velocity_threshold: number;
  reward_coefficient: number;
  task_reward_lerp: number;
  normalizer_epsilon: number;
  normalizer_clip: number;
  parts: { name: string; indices: number[] }[];
  layout: { name: string; shape: number[]; dtype: "f32" | "f16"; offset: number }[];
};

export type CheckpointEntry = {
  name: string;
  step: number | null;
  file: string;
  /** AMP discriminator weights for this checkpoint. */
  disc?: string;
};

/** How the browser assembles the ONNX ``obs`` vector. */
export type ObsLayout = "stacked" | "actor_plus_history";

export type Manifest = {
  id: string;
  task_id?: string;
  play_profile?: PlayProfile;
  robot_xml?: string;
  /** Default ``stacked`` (AMP). HistoryEncoder uses ``actor_plus_history``. */
  obs_layout?: ObsLayout;
  obs_dim: number;
  frame_dim: number;
  history_length: number;
  /** Per-frame dim of ``actor_history`` (HistoryEncoder); defaults to ``frame_dim``. */
  history_frame_dim?: number;
  /** Term names inside each history frame (subset of actor fields). */
  history_terms?: string[];
  num_actions: number;
  joint_names: string[];
  default_joint_pos: number[];
  action_scale: number[];
  field_names?: string[];
  has_velocity_command?: boolean;
  reset_from_motion?: boolean;
  policy_dt: number;
  sim_dt: number;
  decimation: number;
  torso_body: string;
  root_body?: string;
  command_limits?: CommandLimits;
  files: {
    policy: string;
    scene: string;
    alignment: string;
    motion?: string;
    rewards?: string;
    amp_disc?: string;
  };
  motion?: MotionMeta | null;
  rewards?: { n_terms: number; source?: string } | null;
  checkpoints?: CheckpointEntry[];
  default_checkpoint?: string;
  /** DAVIS: ONNX inputs are proprio + depth (not a single obs vector). */
  obs_inputs?: string[];
  proprio_dim?: number;
  depth?: DepthMeta;
};

export type Alignment = {
  meta?: {
    task_id?: string;
    play_profile?: PlayProfile;
    obs_inputs?: string[];
  };
  joints: Array<{
    name: string;
    default_joint_pos: number;
    kp: number;
    kd: number;
  }>;
  observations: {
    obs_layout?: ObsLayout;
    actor: {
      history_length: number;
      concatenated_dim: number;
      frame_dim: number;
      fields: FieldSpec[] | Record<string, Omit<FieldSpec, "name">>;
      obs_scales: Record<string, number>;
      depth?: DepthMeta;
    };
    actor_history?: {
      history_length: number;
      frame_dim: number;
      terms: string[];
    };
  };
  actions: {
    joint_pos: {
      scale_list: number[];
      scale_per_joint: Record<string, number>;
    };
  };
  bodies: {
    torso: string;
    root: string;
  };
  control: {
    sim_dt: number;
    decimation: number;
    policy_dt: number;
  };
};

export type PackIndex = {
  packs: Array<{
    id: string;
    task_id?: string;
    play_profile?: PlayProfile;
    obs_dim?: number;
    num_actions?: number;
    path: string;
  }>;
};
