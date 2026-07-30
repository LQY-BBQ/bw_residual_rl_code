"""ROS Image conversion helpers."""
from __future__ import annotations

import cv2
import numpy as np


class ImageConversionError(ValueError):
    """Raised when a ROS Image message cannot be converted into RGB uint8."""


def _packed_rows(
    msg: object,
    raw: np.ndarray,
    *,
    height: int,
    width: int,
    bytes_per_pixel: int,
    encoding: str,
) -> np.ndarray:
    """Return packed image bytes with ROS row padding removed."""
    packed_row_bytes = width * bytes_per_pixel
    step_value = getattr(msg, "step", packed_row_bytes)
    try:
        step = int(step_value)
    except (TypeError, ValueError) as exc:
        raise ImageConversionError(f"Invalid image step for {encoding}: {step_value!r}") from exc
    if step < packed_row_bytes:
        raise ImageConversionError(
            f"Image step for {encoding} is smaller than packed row: {step} < {packed_row_bytes}"
        )
    required = height * step
    if raw.size < required:
        raise ImageConversionError(
            f"Image buffer too small for {encoding}: {raw.size} < {required} "
            f"(height={height}, step={step})"
        )
    return raw[:required].reshape(height, step)[:, :packed_row_bytes]


def _yuyv_to_rgb(packed: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Convert packed Y0 U Y1 V (BT.601 limited range) to RGB."""
    yuyv = np.ascontiguousarray(packed.reshape(height, width, 2))
    return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)


def ros_image_to_rgb(msg: object) -> np.ndarray:
    """Convert a ROS sensor_msgs/Image-like object to RGB uint8 HWC."""
    height = int(getattr(msg, "height", 0))
    width = int(getattr(msg, "width", 0))
    encoding = str(getattr(msg, "encoding", "")).lower()
    data = getattr(msg, "data", None)
    if height <= 0 or width <= 0:
        raise ImageConversionError(f"Invalid image size: height={height}, width={width}")
    if data is None:
        raise ImageConversionError("Image message has no data field")
    raw = np.frombuffer(bytes(data), dtype=np.uint8)

    if encoding in {"rgb8", "8uc3"}:
        packed = _packed_rows(
            msg, raw, height=height, width=width, bytes_per_pixel=3, encoding=encoding
        )
        return np.ascontiguousarray(packed.reshape(height, width, 3))
    if encoding == "bgr8":
        packed = _packed_rows(
            msg, raw, height=height, width=width, bytes_per_pixel=3, encoding=encoding
        )
        return np.ascontiguousarray(packed.reshape(height, width, 3)[:, :, ::-1])
    if encoding == "rgba8":
        packed = _packed_rows(
            msg, raw, height=height, width=width, bytes_per_pixel=4, encoding=encoding
        )
        return np.ascontiguousarray(packed.reshape(height, width, 4)[:, :, :3])
    if encoding == "bgra8":
        packed = _packed_rows(
            msg, raw, height=height, width=width, bytes_per_pixel=4, encoding=encoding
        )
        return np.ascontiguousarray(packed.reshape(height, width, 4)[:, :, [2, 1, 0]])
    if encoding in {"mono8", "8uc1"}:
        packed = _packed_rows(
            msg, raw, height=height, width=width, bytes_per_pixel=1, encoding=encoding
        )
        gray = packed.reshape(height, width)
        return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
    if encoding in {"yuv422_yuy2", "yuyv", "yuy2"}:
        if width % 2 != 0:
            raise ImageConversionError(f"YUYV image width must be even, got {width}")
        packed = _packed_rows(
            msg, raw, height=height, width=width, bytes_per_pixel=2, encoding=encoding
        )
        return np.ascontiguousarray(_yuyv_to_rgb(packed, height=height, width=width))
    raise ImageConversionError(
        f"Unsupported image encoding {getattr(msg, 'encoding', None)!r}. "
        "Supported encodings: rgb8, bgr8, rgba8, bgra8, mono8, yuv422_yuy2/yuyv/yuy2."
    )
