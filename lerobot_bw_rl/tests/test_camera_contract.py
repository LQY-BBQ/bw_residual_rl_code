from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from policies.act_shared_encoder import (
    BW_IMAGE_KEYS,
    BW_IMAGE_SHAPES,
    validate_bw_act_policy,
)
from visual_cache import (
    CACHE_FORMAT_VERSION,
    CAMERA_CONTRACT_VERSION,
    IMAGE_TRANSFORM,
    _read_dataset_info,
    load_visual_feature_cache,
)
from train_residual_sac import load_bc_initialization


SOURCE_SHAPES = {
    "observation.images.env_cam": (480, 640, 3),
    "observation.images.left_wrist_cam": (270, 480, 3),
    "observation.images.right_wrist_cam": (270, 480, 3),
}


def _fake_policy(shapes: dict[str, tuple[int, int]], keys=BW_IMAGE_KEYS) -> SimpleNamespace:
    features = {
        key: SimpleNamespace(shape=(3, *shapes[key]))
        for key in keys
    }
    config = SimpleNamespace(
        type="act",
        image_features=features,
        robot_state_feature=SimpleNamespace(shape=(16,)),
        action_feature=SimpleNamespace(shape=(16,)),
    )
    model = SimpleNamespace(backbone=object(), encoder_img_feat_input_proj=object())
    return SimpleNamespace(config=config, model=model)


def _write_dataset_info(directory: Path, shapes: dict[str, tuple[int, int, int]]) -> None:
    meta_dir = directory / "meta"
    meta_dir.mkdir()
    info = {
        "total_frames": 10,
        "fps": 30,
        "features": {
            key: {"dtype": "video", "shape": list(shape)}
            for key, shape in shapes.items()
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _cache_metadata() -> dict:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "total_frames": 2,
        "dataset_fps": 30.0,
        "feature_dim": 12,
        "dtype": "float16",
        "image_keys": list(BW_IMAGE_KEYS),
        "source_image_shapes": {key: list(shape) for key, shape in SOURCE_SHAPES.items()},
        "policy_image_shapes": {key: list(shape) for key, shape in BW_IMAGE_SHAPES.items()},
        "camera_contract_version": CAMERA_CONTRACT_VERSION,
        "image_transform": IMAGE_TRANSFORM,
        "act_fingerprint": "test-act",
        "feature_definition": "test",
    }


def _write_cache(directory: Path, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "features.npy", np.zeros((2, 12), dtype=np.float16))
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_act_policy_must_use_exact_third_generation_shapes_and_order() -> None:
    assert validate_bw_act_policy(_fake_policy(BW_IMAGE_SHAPES)) == BW_IMAGE_KEYS

    old_shapes = dict(BW_IMAGE_SHAPES)
    old_shapes["observation.images.env_cam"] = (270, 480)
    with pytest.raises(ValueError, match="image shapes do not match"):
        validate_bw_act_policy(_fake_policy(old_shapes))

    reordered = tuple(reversed(BW_IMAGE_KEYS))
    with pytest.raises(ValueError, match="exact third-generation camera order"):
        validate_bw_act_policy(_fake_policy(BW_IMAGE_SHAPES, keys=reordered))


def test_dataset_metadata_accepts_exact_mixed_camera_shapes(tmp_path: Path) -> None:
    _write_dataset_info(tmp_path, SOURCE_SHAPES)

    total_frames, shapes, fps = _read_dataset_info(tmp_path)

    assert total_frames == 10
    assert shapes == SOURCE_SHAPES
    assert fps == 30.0


def test_dataset_metadata_rejects_previous_environment_shape(tmp_path: Path) -> None:
    old_shapes = dict(SOURCE_SHAPES)
    old_shapes["observation.images.env_cam"] = (270, 480, 3)
    _write_dataset_info(tmp_path, old_shapes)

    with pytest.raises(ValueError, match="third-generation BW camera contract"):
        _read_dataset_info(tmp_path)


def test_visual_cache_accepts_only_current_camera_contract(tmp_path: Path) -> None:
    _write_cache(tmp_path, _cache_metadata())

    cache = load_visual_feature_cache(
        tmp_path,
        expected_total_frames=2,
        expected_act_fingerprint="test-act",
        expected_image_keys=BW_IMAGE_KEYS,
        expected_source_image_shapes=SOURCE_SHAPES,
        expected_policy_image_shapes=BW_IMAGE_SHAPES,
        expected_dataset_fps=30.0,
    )

    assert cache.features.shape == (2, 12)


def test_visual_cache_rejects_previous_format(tmp_path: Path) -> None:
    metadata = _cache_metadata()
    metadata["format_version"] = CACHE_FORMAT_VERSION - 1
    _write_cache(tmp_path, metadata)

    with pytest.raises(ValueError, match="not the third-generation format"):
        load_visual_feature_cache(tmp_path)


def test_visual_cache_rejects_changed_source_shape(tmp_path: Path) -> None:
    metadata = _cache_metadata()
    metadata["source_image_shapes"]["observation.images.env_cam"] = [270, 480, 3]
    _write_cache(tmp_path, metadata)

    with pytest.raises(ValueError, match="third-generation BW camera contract"):
        load_visual_feature_cache(tmp_path)


def test_sac_initialization_rejects_previous_bc_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "residual_bc.pt"
    torch.save(
        {
            "config": {"format_version": 2, "policy_type": "residual_bc"},
            "actor": {},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="third-generation residual BC checkpoint"):
        load_bc_initialization(checkpoint_path)
