"""Joint and feature constants shared by the BW policy runner."""
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
ARM_JOINT_INDICES: list[int] = [JOINT_NAMES.index(name) for name in ARM_JOINT_NAMES]
GRIPPER_JOINT_INDICES: list[int] = [JOINT_NAMES.index(name) for name in GRIPPER_JOINT_NAMES]
DATASET_JOINT_FEATURE_NAMES: list[str] = [f"{name}.pos" for name in JOINT_NAMES]
CAMERA_NAMES: tuple[str, ...] = ("env_cam", "left_wrist_cam", "right_wrist_cam")
CAMERA_TOPICS: dict[str, str] = {
    "env_cam": "/camera/env_d435/color/image_raw",
    "left_wrist_cam": "/camera/left_d405/color/image_raw",
    "right_wrist_cam": "/camera/right_d405/color/image_raw",
}
CAMERA_SOURCES: dict[str, tuple[int, int, str]] = {
    "env_cam": (640, 480, "rgb8"),
    "left_wrist_cam": (480, 270, "rgb8"),
    "right_wrist_cam": (480, 270, "rgb8"),
}
BW_IMAGE_KEYS: tuple[str, ...] = tuple(
    f"observation.images.{name}" for name in CAMERA_NAMES
)
BW_IMAGE_SHAPES: dict[str, tuple[int, int]] = {
    f"observation.images.{name}": (height, width)
    for name, (width, height, _encoding) in CAMERA_SOURCES.items()
}
BW_IMAGE_HWC_SHAPES: dict[str, tuple[int, int, int]] = {
    key: (height, width, 3)
    for key, (height, width) in BW_IMAGE_SHAPES.items()
}
CAMERA_CONTRACT_VERSION = 3
IMAGE_TRANSFORM = "none_exact_shape"
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
