from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_bw_data_collector.image_utils import ImageConversionError, ros_image_to_rgb


def _image(*, height: int, width: int, encoding: str, step: int, data: bytes) -> SimpleNamespace:
    return SimpleNamespace(height=height, width=width, encoding=encoding, step=step, data=data)


def test_yuyv_black_and_white_pixels_convert_to_rgb() -> None:
    msg = _image(
        height=1,
        width=2,
        encoding="yuv422_yuy2",
        step=4,
        data=bytes([16, 128, 235, 128]),
    )

    rgb = ros_image_to_rgb(msg)

    np.testing.assert_array_equal(
        rgb,
        np.asarray([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8),
    )
    assert rgb.flags.c_contiguous


@pytest.mark.parametrize("encoding", ["yuv422_yuy2", "YUYV", "yuy2"])
def test_yuyv_aliases_and_row_padding(encoding: str) -> None:
    msg = _image(
        height=2,
        width=2,
        encoding=encoding,
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


def test_bgr8_conversion_honors_padded_image_step() -> None:
    msg = _image(
        height=2,
        width=2,
        encoding="bgr8",
        step=8,
        data=bytes(
            [
                1, 2, 3, 4, 5, 6, 99, 98,
                10, 20, 30, 40, 50, 60, 97, 96,
            ]
        ),
    )

    rgb = ros_image_to_rgb(msg)

    np.testing.assert_array_equal(
        rgb,
        np.asarray(
            [
                [[3, 2, 1], [6, 5, 4]],
                [[30, 20, 10], [60, 50, 40]],
            ],
            dtype=np.uint8,
        ),
    )
    assert rgb.flags.c_contiguous


@pytest.mark.parametrize(
    ("encoding", "pixel_bytes", "expected_rgb"),
    [
        ("rgb8", [1, 2, 3], [1, 2, 3]),
        ("8UC3", [1, 2, 3], [1, 2, 3]),
        ("rgba8", [1, 2, 3, 200], [1, 2, 3]),
        ("bgra8", [1, 2, 3, 200], [3, 2, 1]),
        ("mono8", [7], [7, 7, 7]),
        ("8UC1", [7], [7, 7, 7]),
    ],
)
def test_supported_encodings_honor_row_padding(
    encoding: str,
    pixel_bytes: list[int],
    expected_rgb: list[int],
) -> None:
    data = bytes(pixel_bytes + [91, 92, 93])
    msg = _image(height=1, width=1, encoding=encoding, step=len(data), data=data)

    rgb = ros_image_to_rgb(msg)

    np.testing.assert_array_equal(rgb, np.asarray([[expected_rgb]], dtype=np.uint8))
    assert rgb.flags.c_contiguous


def test_step_smaller_than_packed_row_is_rejected() -> None:
    msg = _image(height=1, width=2, encoding="rgb8", step=5, data=bytes(6))

    with pytest.raises(ImageConversionError, match="smaller than packed row"):
        ros_image_to_rgb(msg)


def test_short_buffer_is_rejected_using_height_times_step() -> None:
    msg = _image(height=2, width=2, encoding="rgb8", step=8, data=bytes(15))

    with pytest.raises(ImageConversionError, match="15 < 16"):
        ros_image_to_rgb(msg)


def test_odd_yuyv_width_is_rejected() -> None:
    msg = _image(height=1, width=3, encoding="yuyv", step=6, data=bytes(6))

    with pytest.raises(ImageConversionError, match="width must be even"):
        ros_image_to_rgb(msg)


@pytest.mark.parametrize(
    ("height", "width", "encoding", "bytes_per_pixel", "expected_shape"),
    [
        (480, 640, "rgb8", 3, (480, 640, 3)),
        (270, 480, "rgb8", 3, (270, 480, 3)),
    ],
)
def test_verified_camera_resolutions(
    height: int,
    width: int,
    encoding: str,
    bytes_per_pixel: int,
    expected_shape: tuple[int, int, int],
) -> None:
    step = width * bytes_per_pixel
    msg = _image(
        height=height,
        width=width,
        encoding=encoding,
        step=step,
        data=bytes(height * step),
    )

    rgb = ros_image_to_rgb(msg)

    assert rgb.shape == expected_shape
    assert rgb.dtype == np.uint8
    assert rgb.flags.c_contiguous
