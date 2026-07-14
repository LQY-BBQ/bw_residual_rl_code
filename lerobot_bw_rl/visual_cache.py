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
    FrozenACTBundle,
    concatenate_prepared_observations,
    extract_pooled_projected_visual_features,
    load_frozen_act_bundle,
    prepare_raw_observation,
    raw_observation_from_dataset_item,
)

CACHE_FORMAT_VERSION = 1


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


def _read_dataset_total_frames(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total = int(info.get("total_frames", 0))
    if total <= 0:
        raise ValueError(f"Invalid total_frames={total} in {info_path}")
    features = info.get("features", {})
    missing = [key for key in BW_IMAGE_KEYS if key not in features]
    if missing:
        raise ValueError(f"Dataset is missing the three required camera features: {missing}")
    return total


def default_cache_dir(dataset_root: str | Path, act_fingerprint: str) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    return root / ".bw_act_visual_cache" / act_fingerprint[:16]


def _cache_metadata(bundle: FrozenACTBundle, dataset_root: Path, total_frames: int, dtype: str) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset_root": str(dataset_root),
        "total_frames": int(total_frames),
        "feature_dim": int(bundle.visual_feature_dim),
        "dtype": str(dtype),
        "image_keys": list(bundle.image_keys),
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
    total_frames = _read_dataset_total_frames(root)
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
        metadata = _cache_metadata(bundle, root, total_frames, dtype)
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
    )


def load_visual_feature_cache(
    cache_dir: str | Path,
    *,
    expected_total_frames: int | None = None,
    expected_act_fingerprint: str | None = None,
    expected_image_keys: tuple[str, ...] | list[str] | None = None,
) -> VisualCache:
    directory = Path(cache_dir).expanduser().resolve()
    metadata_path = directory / "metadata.json"
    feature_path = directory / "features.npy"
    if not metadata_path.exists() or not feature_path.exists():
        raise FileNotFoundError(f"Incomplete ACT visual cache under {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("format_version", -1)) != CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported visual cache format in {metadata_path}")
    if expected_total_frames is not None and int(metadata.get("total_frames", -1)) != int(expected_total_frames):
        raise ValueError("Visual cache frame count does not match the dataset; rebuild the cache.")
    if expected_act_fingerprint and metadata.get("act_fingerprint") != expected_act_fingerprint:
        raise ValueError("Visual cache was built from a different ACT checkpoint; rebuild the cache.")
    if expected_image_keys is not None and list(metadata.get("image_keys", [])) != list(expected_image_keys):
        raise ValueError("Visual cache camera order does not match the required ACT camera order.")
    features = np.load(feature_path, mmap_mode="r")
    expected_shape = (int(metadata["total_frames"]), int(metadata["feature_dim"]))
    if tuple(features.shape) != expected_shape:
        raise ValueError(f"Visual cache shape={features.shape}, expected={expected_shape}")
    if not np.issubdtype(features.dtype, np.floating):
        raise TypeError(f"Visual cache must be floating-point, got {features.dtype}")
    return VisualCache(directory=directory, features=features, metadata=metadata)
