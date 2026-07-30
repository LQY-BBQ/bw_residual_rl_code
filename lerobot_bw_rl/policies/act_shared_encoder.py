"""Frozen LeRobot ACT visual encoder utilities.

This module is intentionally tied to LeRobot 0.4.4 and the BW three-camera ACT
checkpoint.  It reuses the ACT policy preprocessor, ResNet backbone and
``encoder_img_feat_input_proj`` layer.  ACT parameters are always frozen.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

BW_IMAGE_KEYS: tuple[str, ...] = (
    "observation.images.env_cam",
    "observation.images.left_wrist_cam",
    "observation.images.right_wrist_cam",
)

BW_IMAGE_SHAPES: dict[str, tuple[int, int]] = {
    "observation.images.env_cam": (480, 640),
    "observation.images.left_wrist_cam": (270, 480),
    "observation.images.right_wrist_cam": (270, 480),
}


@dataclass(slots=True)
class FrozenACTBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device
    use_amp: bool
    policy_dir: Path
    image_keys: tuple[str, ...]
    image_shapes: dict[str, tuple[int, int]]
    fingerprint: str

    @property
    def visual_feature_dim(self) -> int:
        dim_model = int(getattr(self.policy.config, "dim_model"))
        return len(self.image_keys) * dim_model


def _import_pretrained_config() -> Any:
    try:
        from lerobot.configs import PreTrainedConfig
    except ImportError:
        from lerobot.configs.policies import PreTrainedConfig
    return PreTrainedConfig


def _import_policy_factory() -> tuple[Any, Any]:
    try:
        from lerobot.policies import get_policy_class, make_pre_post_processors
    except ImportError:
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    return get_policy_class, make_pre_post_processors


def _import_prepare_observation_for_inference() -> Any:
    try:
        from lerobot.policies import prepare_observation_for_inference
    except ImportError:
        from lerobot.policies.utils import prepare_observation_for_inference
    return prepare_observation_for_inference


def resolve_policy_dir(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = (
        root,
        root / "pretrained_model",
        root / "checkpoints" / "last" / "pretrained_model",
        root / "checkpoints" / "last",
    )
    for candidate in candidates:
        direct_ok = (candidate / "config.json").exists() and (candidate / "model.safetensors").exists()
        if direct_ok:
            return candidate
        nested = candidate / "pretrained_model"
        if (nested / "config.json").exists() and (nested / "model.safetensors").exists():
            return nested
    raise FileNotFoundError(f"Could not find a LeRobot pretrained_model under: {root}")


def select_device(requested: str) -> torch.device:
    name = str(requested or "auto").strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(name)


def _device_processor_overrides(policy_dir: Path, filename: str, device: str) -> dict[str, dict[str, str]]:
    path = policy_dir / filename
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    overrides: dict[str, dict[str, str]] = {}
    for step in raw.get("steps", []):
        step_key = step.get("registry_name")
        if step_key is None and "class" in step:
            step_key = str(step["class"]).rsplit(".", 1)[-1]
        if step_key is not None and "device" in step.get("config", {}):
            overrides[str(step_key)] = {"device": device}
    return overrides


def act_policy_fingerprint(policy_dir: str | Path) -> str:
    """Return a stable identity for the ACT feature space.

    Hashing the complete model file once is deliberate: residual checkpoints and
    visual caches must never silently mix features from different ACT weights.
    """
    directory = resolve_policy_dir(policy_dir)
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json"):
        path = directory / name
        if not path.exists():
            continue
        digest.update(name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_bw_act_policy(policy: Any) -> tuple[str, ...]:
    cfg = getattr(policy, "config", None)
    if cfg is None or str(getattr(cfg, "type", "")).lower() != "act":
        raise TypeError("The base policy must be a LeRobot ACT policy.")
    configured = tuple(str(key) for key in getattr(cfg, "image_features", {}).keys())
    if configured != BW_IMAGE_KEYS:
        raise ValueError(
            "This BW residual implementation requires the exact third-generation camera order: "
            f"{list(BW_IMAGE_KEYS)}; checkpoint contains {list(configured)}"
        )
    model = getattr(policy, "model", None)
    for attr in ("backbone", "encoder_img_feat_input_proj"):
        if model is None or not hasattr(model, attr):
            raise AttributeError(f"LeRobot ACT model is missing model.{attr}; LeRobot 0.4.4 is required.")
    state_feature = getattr(cfg, "robot_state_feature", None)
    action_feature = getattr(cfg, "action_feature", None)
    if state_feature is None or int(state_feature.shape[0]) != 16:
        raise ValueError("The ACT checkpoint must use a 16-dimensional observation.state.")
    if action_feature is None or int(action_feature.shape[0]) != 16:
        raise ValueError("The ACT checkpoint must output a 16-dimensional action.")
    shapes = expected_image_shapes_from_policy(policy)
    if shapes != BW_IMAGE_SHAPES:
        raise ValueError(
            "ACT checkpoint image shapes do not match the third-generation BW camera contract: "
            f"expected={BW_IMAGE_SHAPES}, checkpoint={shapes}. Retrain ACT with the new dataset."
        )
    return BW_IMAGE_KEYS


def expected_image_shapes_from_policy(policy: Any) -> dict[str, tuple[int, int]]:
    """Read each camera's CHW training shape from an ACT policy config."""
    cfg = getattr(policy, "config", None)
    image_features = getattr(cfg, "image_features", {}) if cfg is not None else {}
    shapes: dict[str, tuple[int, int]] = {}
    for key in BW_IMAGE_KEYS:
        feature = image_features.get(key)
        shape = tuple(getattr(feature, "shape", ()) or ()) if feature is not None else ()
        if len(shape) != 3 or int(shape[0]) != 3:
            raise ValueError(f"ACT image feature {key!r} must have CHW shape with 3 channels, got {shape}")
        height, width = int(shape[1]), int(shape[2])
        if height <= 0 or width <= 0:
            raise ValueError(f"ACT image feature {key!r} has invalid size {(height, width)}")
        shapes[key] = (height, width)
    return shapes


def load_frozen_act_bundle(
    policy_path: str | Path,
    *,
    device: str = "cuda",
    use_amp: bool = False,
) -> FrozenACTBundle:
    PreTrainedConfig = _import_pretrained_config()
    get_policy_class, make_pre_post_processors = _import_policy_factory()
    policy_dir = resolve_policy_dir(policy_path)
    torch_device = select_device(device)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_dir)
    policy_cfg.device = str(torch_device)
    policy_cfg.use_amp = bool(use_amp and torch_device.type == "cuda")
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_dir, config=policy_cfg)
    policy.to(torch_device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    if hasattr(policy, "reset"):
        policy.reset()
    image_keys = validate_bw_act_policy(policy)
    image_shapes = expected_image_shapes_from_policy(policy)
    pre_overrides = _device_processor_overrides(policy_dir, "policy_preprocessor.json", str(torch_device))
    post_overrides = _device_processor_overrides(policy_dir, "policy_postprocessor.json", "cpu")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_dir),
        preprocessor_overrides=pre_overrides,
        postprocessor_overrides=post_overrides,
    )
    return FrozenACTBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=torch_device,
        use_amp=bool(policy_cfg.use_amp),
        policy_dir=policy_dir,
        image_keys=image_keys,
        image_shapes=image_shapes,
        fingerprint=act_policy_fingerprint(policy_dir),
    )


def _to_uint8_hwc(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        value = image.detach().cpu()
        if value.ndim != 3:
            raise ValueError(f"Expected a 3-D image tensor, got shape={tuple(value.shape)}")
        if value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        array = value.numpy()
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC image, got shape={array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.5:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def raw_observation_from_dataset_item(item: dict[str, Any], image_keys: Iterable[str] = BW_IMAGE_KEYS) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    state = item.get("observation.state")
    if state is None:
        raise KeyError("Dataset item is missing observation.state")
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    observation["observation.state"] = np.asarray(state, dtype=np.float32).reshape(16)
    for key in image_keys:
        if key not in item:
            raise KeyError(f"Dataset item is missing required ACT image: {key}")
        observation[key] = _to_uint8_hwc(item[key])
    return observation


def prepare_raw_observation(
    bundle: FrozenACTBundle,
    observation: dict[str, np.ndarray],
    *,
    task: str = "",
    robot_type: str = "bw_runtime",
) -> dict[str, Any]:
    prepare_observation_for_inference = _import_prepare_observation_for_inference()
    copied = {key: np.asarray(value).copy() for key, value in observation.items()}
    for key, (height, width) in bundle.image_shapes.items():
        if key not in copied:
            raise KeyError(f"Observation is missing required ACT image: {key}")
        image = _to_uint8_hwc(copied[key])
        expected_shape = (height, width, 3)
        if image.shape != expected_shape:
            raise ValueError(
                f"Image {key!r} has shape={image.shape}, expected exact third-generation "
                f"ACT shape={expected_shape}; runtime resizing is disabled"
            )
        copied[key] = image
    prepared = prepare_observation_for_inference(
        copied,
        bundle.device,
        task=task,
        robot_type=robot_type,
    )
    return bundle.preprocessor(prepared)


def concatenate_prepared_observations(items: list[dict[str, Any]], image_keys: Iterable[str] = BW_IMAGE_KEYS) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot concatenate an empty observation batch")
    keys = ["observation.state", *image_keys]
    batch: dict[str, Any] = {}
    for key in keys:
        tensors = [item[key] for item in items]
        batch[key] = torch.cat(tensors, dim=0)
    return batch


def extract_pooled_projected_visual_features(
    bundle: FrozenACTBundle,
    prepared_batch: dict[str, Any],
) -> torch.Tensor:
    """Run the frozen ACT ResNet and 1x1 projector, then pool each camera.

    Output order is always env, left wrist, right wrist, independent of the
    dictionary order stored inside the ACT checkpoint.
    """
    model = bundle.policy.model
    pooled: list[torch.Tensor] = []
    autocast_ctx = (
        torch.autocast(device_type=bundle.device.type)
        if bundle.device.type == "cuda" and bundle.use_amp
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        for key in bundle.image_keys:
            image = prepared_batch[key]
            feature_map = model.backbone(image)["feature_map"]
            projected = model.encoder_img_feat_input_proj(feature_map)
            pooled.append(projected.mean(dim=(-2, -1)))
    return torch.cat(pooled, dim=-1).float()
