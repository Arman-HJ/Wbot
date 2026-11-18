"""XML joint order for WBOT 20-DOF AMP. No waist joint."""

WBOT_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
)

# Same six end-effectors as E1 AMP, and these body names exist on WBOT.
WBOT_KEY_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
)

# Leg-only WBOT: arms are welded to the torso, so only the feet carry style.
WBOT_12DOF_JOINT_NAMES = WBOT_JOINT_NAMES[:12]
WBOT_12DOF_KEY_BODY_NAMES = WBOT_KEY_BODY_NAMES[:2]
