"""Build and validate pooled ACT visual-feature caches for BW residual learning."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from policies.act_shared_encoder import (
    BW_IMAGE_KEYS,
    BW_IMAGE_SHAPES,
    FrozenACTBundle,
    concatenate_prepared_observations,
    extract_pooled_projected_visual_features,
    load_frozen_act_bundle,
    prepare_raw_observation,
    raw_observation_from_dataset_item,
)

CACHE_FORMAT_VERSION = 4
CAMERA_CONTRACT_VERSION = 3
IMAGE_TRANSFORM = "none_exact_shape"
EXPECTED_DATASET_FPS = 30.0


@dataclass(slots=True)
class VisualCache:
    directory: Path
    features: np.ndarray
    metadata: dict[str, Any]

    @property
    def feature_dim(self) -> int:
        return int(self.metadata["feature_dim"])

    @property
    def act_fingerprint(self) -> str:
        return str(self.metadata["act_fingerprint"])


def _read_dataset_info(
    root: Path,
    *,
    expected_fps: float = EXPECTED_DATASET_FPS,
) -> tuple[int, dict[str, tuple[int, int, int]], float]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total = int(info.get("total_frames", 0))
    if total <= 0:
        raise ValueError(f"Invalid total_frames={total} in {info_path}")
    dataset_fps = float(info.get("fps", 0.0))
    if dataset_fps <= 0:
        raise ValueError(f"Invalid or missing fps={dataset_fps:g} in {info_path}")
    if expected_fps > 0 and not np.isclose(dataset_fps, expected_fps, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Dataset fps={dataset_fps:g}, expected {expected_fps:g} FPS for BW camera training"
        )
    features = info.get("features", {})
    missing = [key for key in BW_IMAGE_KEYS if key not in features]
    if missing:
        raise ValueError(f"Dataset is missing the three required camera features: {missing}")
    image_shapes: dict[str, tuple[int, int, int]] = {}
    for key in BW_IMAGE_KEYS:
        shape = tuple(int(value) for value in features[key].get("shape", ()))
        if len(shape) != 3 or shape[2] != 3 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Dataset image feature {key!r} must have RGB HWC shape, got {shape}")
        image_shapes[key] = shape
    expected_shapes = {
        key: (height, width, 3)
        for key, (height, width) in BW_IMAGE_SHAPES.items()
    }
    if image_shapes != expected_shapes:
        raise ValueError(
            "Dataset image shapes do not match the third-generation BW camera contract: "
            f"expected={expected_shapes}, dataset={image_shapes}"
        )
    return total, image_shapes, dataset_fps


def default_cache_dir(dataset_root: str | Path, act_fingerprint: str) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    return root / ".bw_act_visual_cache" / act_fingerprint[:16]


def _serialized_shapes(shapes: dict[str, tuple[int, ...]]) -> dict[str, list[int]]:
    return {key: [int(value) for value in shape] for key, shape in shapes.items()}


def _cache_metadata(
    bundle: FrozenACTBundle,
    dataset_root: Path,
    total_frames: int,
    dtype: str,
    source_image_shapes: dict[str, tuple[int, int, int]],
    dataset_fps: float,
) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset_root": str(dataset_root),
        "total_frames": int(total_frames),
        "dataset_fps": float(dataset_fps),
        "feature_dim": int(bundle.visual_feature_dim),
        "dtype": str(dtype),
        "image_keys": list(bundle.image_keys),
        "source_image_shapes": _serialized_shapes(source_image_shapes),
        "policy_image_shapes": _serialized_shapes(bundle.image_shapes),
        "camera_contract_version": CAMERA_CONTRACT_VERSION,
        "image_transform": IMAGE_TRANSFORM,
        "camera_order": ["env_cam", "left_wrist_cam", "right_wrist_cam"],
        "feature_definition": "ACT ResNet layer4 -> encoder_img_feat_input_proj -> spatial mean, cameras concatenated",
        "act_policy_dir": str(bundle.policy_dir),
        "act_fingerprint": bundle.fingerprint,
        "act_dim_model": int(bundle.policy.config.dim_model),
        "act_vision_backbone": str(bundle.policy.config.vision_backbone),
        "act_parameters_frozen": True,
    }


def _write_batch(
    bundle: FrozenACTBundle,
    dataset: Any,
    start: int,
    end: int,
    destination: np.ndarray,
) -> None:
    prepared_items: list[dict[str, Any]] = []
    for index in range(start, end):
        item = dataset[index]
        raw = raw_observation_from_dataset_item(item, bundle.image_keys)
        prepared_items.append(prepare_raw_observation(bundle, raw))
    prepared_batch = concatenate_prepared_observations(prepared_items, bundle.image_keys)
    feature = extract_pooled_projected_visual_features(bundle, prepared_batch)
    feature_np = feature.detach().cpu().numpy().astype(destination.dtype, copy=False)
    destination[start:end] = feature_np


def build_visual_feature_cache(
    *,
    dataset_root: str | Path,
    act_policy_path: str | Path,
    cache_dir: str | Path | None = None,
    repo_id: str | None = None,
    device: str = "cuda",
    use_amp: bool = False,
    batch_size: int = 16,
    dtype: str = "float16",
    overwrite: bool = False,
    video_backend: str | None = None,
) -> VisualCache:
    """Decode each dataset frame and encode it once with the frozen ACT vision stack."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be float16 or float32")
    root = Path(dataset_root).expanduser().resolve()
    total_frames, source_image_shapes, dataset_fps = _read_dataset_info(root)
    bundle = load_frozen_act_bundle(act_policy_path, device=device, use_amp=use_amp)
    out_dir = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir(root, bundle.fingerprint)
    metadata_path = out_dir / "metadata.json"
    feature_path = out_dir / "features.npy"

    if out_dir.exists() and not overwrite:
        return load_visual_feature_cache(
            out_dir,
            expected_total_frames=total_frames,
            expected_act_fingerprint=bundle.fingerprint,
            expected_image_keys=bundle.image_keys,
            expected_source_image_shapes=source_image_shapes,
            expected_policy_image_shapes=bundle.image_shapes,
            expected_dataset_fps=dataset_fps,
        )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import only after the local project has avoided the historical top-level
    # package name `datasets`, which would shadow Hugging Face datasets.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    final_repo_id = repo_id or f"local/{root.name}"
    kwargs: dict[str, Any] = {"repo_id": final_repo_id, "root": root, "download_videos": True}
    if video_backend:
        kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(**kwargs)
    if len(dataset) != total_frames:
        raise ValueError(
            f"LeRobotDataset length ({len(dataset)}) does not match meta/info.json total_frames ({total_frames})."
        )

    tmp_feature_path = out_dir / "features.tmp.npy"
    features = np.lib.format.open_memmap(
        tmp_feature_path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(total_frames, bundle.visual_feature_dim),
    )
    try:
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            _write_batch(bundle, dataset, start, end, features)
            features.flush()
            print(f"[visual-cache] {end}/{total_frames} frames")
        del features
        tmp_feature_path.replace(feature_path)
        metadata = _cache_metadata(
            bundle,
            root,
            total_frames,
            dtype,
            source_image_shapes,
            dataset_fps,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    except BaseException:
        with np.errstate(all="ignore"):
            del features
        if tmp_feature_path.exists():
            tmp_feature_path.unlink()
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return load_visual_feature_cache(
        out_dir,
        expected_total_frames=total_frames,
        expected_act_fingerprint=bundle.fingerprint,
        expected_image_keys=bundle.image_keys,
        expected_source_image_shapes=source_image_shapes,
        expected_policy_image_shapes=bundle.image_shapes,
        expected_dataset_fps=dataset_fps,
    )


def load_visual_feature_cache(
    cache_dir: str | Path,
    *,
    expected_total_frames: int | None = None,
    expected_act_fingerprint: str | None = None,
    expected_image_keys: tuple[str, ...] | list[str] | None = None,
    expected_source_image_shapes: dict[str, tuple[int, int, int]] | None = None,
    expected_policy_image_shapes: dict[str, tuple[int, int]] | None = None,
    expected_dataset_fps: float | None = None,
) -> VisualCache:
    directory = Path(cache_dir).expanduser().resolve()
    metadata_path = directory / "metadata.json"
    feature_path = directory / "features.npy"
    if not metadata_path.exists() or not feature_path.exists():
        raise FileNotFoundError(f"Incomplete ACT visual cache under {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    format_version = int(metadata.get("format_version", -1))
    if format_version != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Visual cache format={format_version} is not the third-generation format="
            f"{CACHE_FORMAT_VERSION}; rebuild the cache."
        )
    if expected_total_frames is not None and int(metadata.get("total_frames", -1)) != int(expected_total_frames):
        raise ValueError("Visual cache frame count does not match the dataset; rebuild the cache.")
    if expected_act_fingerprint and metadata.get("act_fingerprint") != expected_act_fingerprint:
        raise ValueError("Visual cache was built from a different ACT checkpoint; rebuild the cache.")
    cached_image_keys = list(metadata.get("image_keys", []))
    if cached_image_keys != list(BW_IMAGE_KEYS):
        raise ValueError("Visual cache camera order does not match the third-generation BW contract.")
    if expected_image_keys is not None and cached_image_keys != list(expected_image_keys):
        raise ValueError("Visual cache camera order does not match the required ACT camera order.")
    if int(metadata.get("camera_contract_version", -1)) != CAMERA_CONTRACT_VERSION:
        raise ValueError("Visual cache camera contract is stale; rebuild the cache.")
    if metadata.get("image_transform") != IMAGE_TRANSFORM:
        raise ValueError("Visual cache image transform must be none_exact_shape; rebuild the cache.")
    cached_fps = float(metadata.get("dataset_fps", 0.0))
    if not np.isclose(cached_fps, EXPECTED_DATASET_FPS, rtol=0.0, atol=1e-6):
        raise ValueError("Visual cache was not built from a 30 FPS dataset; rebuild the cache.")
    if expected_dataset_fps is not None and not np.isclose(
        cached_fps,
        expected_dataset_fps,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Visual cache dataset FPS does not match the dataset; rebuild the cache.")
    cached_source = {
        key: tuple(int(value) for value in shape)
        for key, shape in metadata.get("source_image_shapes", {}).items()
    }
    cached_policy = {
        key: tuple(int(value) for value in shape)
        for key, shape in metadata.get("policy_image_shapes", {}).items()
    }
    required_source = {
        key: (height, width, 3)
        for key, (height, width) in BW_IMAGE_SHAPES.items()
    }
    if cached_source != required_source or cached_policy != BW_IMAGE_SHAPES:
        raise ValueError(
            "Visual cache image shapes do not match the third-generation BW camera contract; "
            "rebuild the cache."
        )
    if expected_source_image_shapes is not None and cached_source != expected_source_image_shapes:
        raise ValueError("Visual cache source image shapes do not match the dataset; rebuild the cache.")
    if expected_policy_image_shapes is not None and cached_policy != expected_policy_image_shapes:
        raise ValueError("Visual cache target image shapes do not match the ACT checkpoint; rebuild the cache.")
    features = np.load(feature_path, mmap_mode="r")
    expected_shape = (int(metadata["total_frames"]), int(metadata["feature_dim"]))
    if tuple(features.shape) != expected_shape:
        raise ValueError(f"Visual cache shape={features.shape}, expected={expected_shape}")
    if not np.issubdtype(features.dtype, np.floating):
        raise TypeError(f"Visual cache must be floating-point, got {features.dtype}")
    return VisualCache(directory=directory, features=features, metadata=metadata)
