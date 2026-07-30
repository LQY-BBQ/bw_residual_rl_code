from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from lerobot_bw_policy_runner.constants import (
    BW_IMAGE_HWC_SHAPES,
    BW_IMAGE_KEYS,
    BW_IMAGE_SHAPES,
    CAMERA_CONTRACT_VERSION,
    IMAGE_TRANSFORM,
)
from lerobot_bw_policy_runner.image_utils import ImageConversionError, ros_image_to_rgb
from lerobot_bw_policy_runner.infer_node import _validate_residual_pair
from lerobot_bw_policy_runner.policy_loader import validate_bw_act_policy


def _image(*, height: int, width: int, encoding: str, step: int, data: bytes) -> SimpleNamespace:
    return SimpleNamespace(height=height, width=width, encoding=encoding, step=step, data=data)


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
    model = SimpleNamespace(encoder_img_feat_input_proj=object())
    return SimpleNamespace(config=config, model=model)


def test_yuyv_black_white_and_padding() -> None:
    msg = _image(
        height=2,
        width=2,
        encoding="yuv422_yuy2",
        step=6,
        data=bytes(
            [
                16, 128, 235, 128, 99, 98,
                235, 128, 16, 128, 97, 96,
            ]
        ),
    )

    rgb = ros_image_to_rgb(msg)

    np.testing.assert_array_equal(
        rgb,
        np.asarray(
            [
                [[0, 0, 0], [255, 255, 255]],
                [[255, 255, 255], [0, 0, 0]],
            ],
            dtype=np.uint8,
        ),
    )
    assert rgb.flags.c_contiguous


def test_rgb_row_padding_is_removed() -> None:
    msg = _image(
        height=2,
        width=1,
        encoding="rgb8",
        step=5,
        data=bytes([1, 2, 3, 99, 98, 4, 5, 6, 97, 96]),
    )

    rgb = ros_image_to_rgb(msg)

    np.testing.assert_array_equal(
        rgb,
        np.asarray([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.uint8),
    )


def test_invalid_step_is_rejected() -> None:
    msg = _image(height=1, width=2, encoding="rgb8", step=5, data=bytes(6))

    with pytest.raises(ImageConversionError, match="smaller than packed row"):
        ros_image_to_rgb(msg)


def test_act_policy_requires_exact_third_generation_shapes_and_order() -> None:
    assert validate_bw_act_policy(_fake_policy(BW_IMAGE_SHAPES)) == BW_IMAGE_KEYS

    previous_shapes = dict(BW_IMAGE_SHAPES)
    previous_shapes["observation.images.env_cam"] = (270, 480)
    with pytest.raises(ValueError, match="image shapes do not match"):
        validate_bw_act_policy(_fake_policy(previous_shapes))

    with pytest.raises(ValueError, match="exact third-generation BW camera order"):
        validate_bw_act_policy(_fake_policy(BW_IMAGE_SHAPES, tuple(reversed(BW_IMAGE_KEYS))))


def _compatible_bundles() -> tuple[SimpleNamespace, SimpleNamespace]:
    act = SimpleNamespace(
        fingerprint="act-fingerprint",
        image_keys=BW_IMAGE_KEYS,
        visual_feature_dim=1536,
        image_shapes=BW_IMAGE_SHAPES,
    )
    residual = SimpleNamespace(
        policy_type="residual_rl",
        act_fingerprint="act-fingerprint",
        image_keys=BW_IMAGE_KEYS,
        visual_feature_dim=1536,
        source_image_shapes=BW_IMAGE_HWC_SHAPES,
        policy_image_shapes=BW_IMAGE_SHAPES,
        camera_contract_version=CAMERA_CONTRACT_VERSION,
        image_transform=IMAGE_TRANSFORM,
        dataset_fps=30.0,
    )
    return act, residual


def test_residual_checkpoint_must_match_camera_contract() -> None:
    act, residual = _compatible_bundles()
    _validate_residual_pair("act_residual_rl", act, residual)

    residual.image_transform = "opencv_resize"
    with pytest.raises(ValueError, match="image transform"):
        _validate_residual_pair("act_residual_rl", act, residual)


def test_default_runtime_camera_contract_is_d435_and_two_d405() -> None:
    from lerobot_bw_policy_runner.config import default_config_path, load_config

    config = load_config(default_config_path(), robot_sn="BW_TEST123")
    stream = config.inference.camera_stream
    assert stream is not None
    assert config.inference.fps == 30.0
    assert stream.expected_fps == 30.0
    assert stream.require_new_frames
    assert config.robot.input_topics.cameras["env_cam"] == "/camera/env_d435/color/image_raw"
    env = stream.sources["env_cam"]
    assert (env.width, env.height, env.encoding) == (640, 480, "rgb8")


def test_runtime_rejects_non_30_fps_override() -> None:
    from lerobot_bw_policy_runner.config import default_config_path, load_config

    with pytest.raises(ValueError, match="must match.*expected_fps"):
        load_config(default_config_path(), robot_sn="BW_TEST123", fps=15)


def test_runtime_rejects_previous_hdmi_contract(tmp_path) -> None:
    from lerobot_bw_policy_runner.config import default_config_path, load_config

    raw = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    raw["robot"]["input_topics"]["cameras"]["env_cam"] = "/env_camera/image_raw"
    raw["inference"]["camera_stream"]["sources"]["env_cam"] = {
        "width": 480,
        "height": 270,
        "encoding": "bgr8",
    }
    config_path = tmp_path / "old_camera.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="third-generation BW contract"):
        load_config(config_path, robot_sn="BW_TEST123")
