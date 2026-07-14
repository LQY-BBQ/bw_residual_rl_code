"""Constants shared by the BW LeRobot data collector.

The 16-D order is the contract shared by:
- LeRobot dataset observation.state
- LeRobot dataset action
- policy runner ACT output
- residual SAC delta output
"""
from __future__ import annotations

JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_yaw_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_gripper_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_yaw_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_gripper_joint",
]

ARM_JOINT_NAMES: list[str] = [name for name in JOINT_NAMES if not name.endswith("_gripper_joint")]
GRIPPER_JOINT_NAMES: list[str] = ["left_gripper_joint", "right_gripper_joint"]
GRIPPER_SHORT_NAMES: list[str] = ["left_gripper", "right_gripper"]
DATASET_JOINT_FEATURE_NAMES: list[str] = [f"{name}.pos" for name in JOINT_NAMES]

CAMERA_NAMES: list[str] = ["env_cam", "left_wrist_cam", "right_wrist_cam"]

JOINT_NAME_ALIASES: dict[str, str] = {
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "left_gripper": "left_gripper_joint",
    "right_gripper": "right_gripper_joint",
    "L_Shoulder_Pitch_Joint": "left_shoulder_pitch_joint",
    "L_Shoulder_Roll_Joint": "left_shoulder_roll_joint",
    "L_Shoulder_Yaw_Joint": "left_shoulder_yaw_joint",
    "L_Elbow_Pitch_Joint": "left_elbow_joint",
    "L_Wrist_Roll_Joint": "left_wrist_roll_joint",
    "L_Wrist_Pitch_Joint": "left_wrist_pitch_joint",
    "L_Wrist_Yaw_Joint": "left_wrist_yaw_joint",
    "R_Shoulder_Pitch_Joint": "right_shoulder_pitch_joint",
    "R_Shoulder_Roll_Joint": "right_shoulder_roll_joint",
    "R_Shoulder_Yaw_Joint": "right_shoulder_yaw_joint",
    "R_Elbow_Pitch_Joint": "right_elbow_joint",
    "R_Wrist_Roll_Joint": "right_wrist_roll_joint",
    "R_Wrist_Pitch_Joint": "right_wrist_pitch_joint",
    "R_Wrist_Yaw_Joint": "right_wrist_yaw_joint",
}

KNOWN_NON_COLLECTION_JOINTS: set[str] = {"pelvis_joint", "head_pitch_joint", "head_yaw_joint"}

OBS_STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEY_PREFIX = "observation.images"
DEFAULT_ROBOT_TYPE = "bw_runtime"

# Extra feature names used only by --mode rl.
CONTROL_SOURCE_KEY = "control_source"
IS_INTERVENTION_KEY = "is_intervention"
HAS_HUMAN_ACTION_KEY = "has_human_action"
ACTION_ACT_KEY = "action.act"
ACTION_RL_DELTA_KEY = "action.rl_delta"
ACTION_HUMAN_KEY = "action.human"
ACTION_EXECUTED_KEY = "action.executed"
REWARD_KEY = "reward"
DONE_KEY = "done"
SUCCESS_KEY = "success"
TIMESTAMP_DIFF_PREFIX = "timing"
