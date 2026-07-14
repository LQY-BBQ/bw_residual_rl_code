"""LeRobot 0.4.4 ACT loading and one-pass shared visual inference."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

BW_IMAGE_KEYS: tuple[str, ...] = (
    "observation.images.env_cam",
    "observation.images.left_wrist_cam",
    "observation.images.right_wrist_cam",
)


@dataclass(slots=True)
class PolicyBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device
    use_amp: bool
    policy_dir: Path
    image_keys: tuple[str, ...]
    fingerprint: str

    @property
    def visual_feature_dim(self) -> int:
        return len(self.image_keys) * int(self.policy.config.dim_model)


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


def _device_processor_overrides(policy_dir: Path, config_filename: str, device: str) -> dict[str, dict[str, str]]:
    config_path = policy_dir / config_filename
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    overrides: dict[str, dict[str, str]] = {}
    for step in config.get("steps", []):
        step_key = step.get("registry_name")
        if step_key is None and "class" in step:
            step_key = str(step["class"]).rsplit(".", 1)[-1]
        if step_key is not None and "device" in step.get("config", {}):
            overrides[str(step_key)] = {"device": device}
    return overrides


def resolve_policy_dir(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = (
        root,
        root / "pretrained_model",
        root / "checkpoints" / "last" / "pretrained_model",
        root / "checkpoints" / "last",
    )
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
            return candidate
        nested = candidate / "pretrained_model"
        if (nested / "config.json").exists() and (nested / "model.safetensors").exists():
            return nested
    raise FileNotFoundError(f"Could not find LeRobot pretrained_model under {root}")


def act_policy_fingerprint(policy_dir: str | Path) -> str:
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


def select_device(requested: str) -> torch.device:
    requested = str(requested or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def validate_bw_act_policy(policy: Any) -> tuple[str, ...]:
    cfg = getattr(policy, "config", None)
    if cfg is None or str(getattr(cfg, "type", "")).lower() != "act":
        raise TypeError("The base policy must be a LeRobot ACT policy")
    configured = tuple(str(key) for key in getattr(cfg, "image_features", {}).keys())
    if set(configured) != set(BW_IMAGE_KEYS) or len(configured) != len(BW_IMAGE_KEYS):
        raise ValueError(
            "This runner requires exactly the three BW ACT cameras "
            f"{list(BW_IMAGE_KEYS)}, checkpoint contains {list(configured)}"
        )
    if int(cfg.robot_state_feature.shape[0]) != 16 or int(cfg.action_feature.shape[0]) != 16:
        raise ValueError("The BW ACT checkpoint must use 16-D state and 16-D action")
    model = getattr(policy, "model", None)
    if model is None or not hasattr(model, "encoder_img_feat_input_proj"):
        raise AttributeError("ACT model.encoder_img_feat_input_proj is required; use LeRobot 0.4.4")
    return BW_IMAGE_KEYS


def load_policy_bundle(policy_path: str | Path, *, device: str = "cuda", use_amp: bool = False) -> PolicyBundle:
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
    preprocessor_overrides = _device_processor_overrides(policy_dir, "policy_preprocessor.json", str(torch_device))
    postprocessor_overrides = _device_processor_overrides(policy_dir, "policy_postprocessor.json", "cpu")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_dir),
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
    return PolicyBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=torch_device,
        use_amp=bool(policy_cfg.use_amp),
        policy_dir=policy_dir,
        image_keys=image_keys,
        fingerprint=act_policy_fingerprint(policy_dir),
    )


def expected_image_shapes_from_policy(policy: Any) -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    cfg = getattr(policy, "config", None)
    image_features = getattr(cfg, "image_features", {}) if cfg is not None else {}
    for key, feature in image_features.items():
        shape = tuple(getattr(feature, "shape", ()) or ())
        if len(shape) == 3:
            shapes[str(key)] = (int(shape[1]), int(shape[2]))
    return shapes


def prepare_observation(
    bundle: PolicyBundle,
    observation: dict[str, np.ndarray],
    *,
    task: str = "",
    robot_type: str = "bw_runtime",
) -> dict[str, Any]:
    prepare_observation_for_inference = _import_prepare_observation_for_inference()
    copied = {key: np.asarray(value).copy() for key, value in observation.items()}
    prepared = prepare_observation_for_inference(copied, bundle.device, task=task, robot_type=robot_type)
    return bundle.preprocessor(prepared)


def _policy_action_every_step(bundle: PolicyBundle, prepared_observation: dict[str, Any]) -> Any:
    """Run a fresh ACT forward every control cycle.

    Temporal ensembling remains active when configured.  Without temporal
    ensembling, the first action from the newly predicted chunk is used; the ACT
    action queue is deliberately bypassed.
    """
    policy = bundle.policy
    if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
        return policy.select_action(prepared_observation)
    action_chunk = policy.predict_action_chunk(prepared_observation)
    return action_chunk[:, 0]


def _to_numpy_action(action: Any) -> np.ndarray:
    if hasattr(action, "detach"):
        value = action.detach().cpu().numpy()
    else:
        value = np.asarray(action)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def infer_action(
    bundle: PolicyBundle,
    observation: dict[str, np.ndarray],
    *,
    task: str = "",
    robot_type: str = "bw_runtime",
) -> np.ndarray:
    autocast_ctx = (
        torch.autocast(device_type=bundle.device.type)
        if bundle.device.type == "cuda" and bundle.use_amp
        else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        prepared = prepare_observation(bundle, observation, task=task, robot_type=robot_type)
        action = _policy_action_every_step(bundle, prepared)
        action = bundle.postprocessor(action)
    return _to_numpy_action(action)


class _ProjectedVisualCollector:
    """Capture ACT's projected camera maps during the same forward used for action."""

    def __init__(self, bundle: PolicyBundle) -> None:
        self.bundle = bundle
        self.act_camera_order = tuple(str(key) for key in bundle.policy.config.image_features.keys())
        self.outputs: list[torch.Tensor] = []
        module = bundle.policy.model.encoder_img_feat_input_proj
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise TypeError("ACT image projection hook expected a [B,C,H,W] tensor")
        self.outputs.append(output.detach())

    def close(self) -> None:
        self.handle.remove()

    def pooled_feature(self) -> np.ndarray:
        if len(self.outputs) != len(self.act_camera_order):
            raise RuntimeError(
                "ACT shared vision capture failed: "
                f"captured {len(self.outputs)} maps for {len(self.act_camera_order)} cameras"
            )
        by_key = {key: value for key, value in zip(self.act_camera_order, self.outputs)}
        pooled = [by_key[key].mean(dim=(-2, -1)) for key in self.bundle.image_keys]
        feature = torch.cat(pooled, dim=-1).float()
        if feature.shape[0] != 1:
            raise ValueError(f"Runtime ACT batch size must be 1, got {feature.shape[0]}")
        return feature[0].cpu().numpy().astype(np.float32, copy=False)


def infer_action_with_shared_visual_feature(
    bundle: PolicyBundle,
    observation: dict[str, np.ndarray],
    *,
    task: str = "",
    robot_type: str = "bw_runtime",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ACT action and pooled visual feature from one ACT image pass."""
    collector = _ProjectedVisualCollector(bundle)
    autocast_ctx = (
        torch.autocast(device_type=bundle.device.type)
        if bundle.device.type == "cuda" and bundle.use_amp
        else nullcontext()
    )
    try:
        with torch.inference_mode(), autocast_ctx:
            prepared = prepare_observation(bundle, observation, task=task, robot_type=robot_type)
            action = _policy_action_every_step(bundle, prepared)
            action = bundle.postprocessor(action)
        return _to_numpy_action(action), collector.pooled_feature()
    finally:
        collector.close()


# Historical name retained for code that imported it directly.
infer_action_with_feature = infer_action_with_shared_visual_feature
