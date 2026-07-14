"""ROS Image conversion helpers."""
from __future__ import annotations

import numpy as np


class ImageConversionError(ValueError):
    """Raised when a ROS Image message cannot be converted into RGB uint8."""


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
        expected = height * width * 3
        if raw.size < expected:
            raise ImageConversionError(f"Image buffer too small for {encoding}: {raw.size} < {expected}")
        return np.ascontiguousarray(raw[:expected].reshape(height, width, 3))
    if encoding == "bgr8":
        expected = height * width * 3
        if raw.size < expected:
            raise ImageConversionError(f"Image buffer too small for bgr8: {raw.size} < {expected}")
        return np.ascontiguousarray(raw[:expected].reshape(height, width, 3)[:, :, ::-1])
    if encoding == "rgba8":
        expected = height * width * 4
        if raw.size < expected:
            raise ImageConversionError(f"Image buffer too small for rgba8: {raw.size} < {expected}")
        return np.ascontiguousarray(raw[:expected].reshape(height, width, 4)[:, :, :3])
    if encoding == "bgra8":
        expected = height * width * 4
        if raw.size < expected:
            raise ImageConversionError(f"Image buffer too small for bgra8: {raw.size} < {expected}")
        return np.ascontiguousarray(raw[:expected].reshape(height, width, 4)[:, :, [2, 1, 0]])
    if encoding in {"mono8", "8uc1"}:
        expected = height * width
        if raw.size < expected:
            raise ImageConversionError(f"Image buffer too small for {encoding}: {raw.size} < {expected}")
        gray = raw[:expected].reshape(height, width)
        return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
    raise ImageConversionError(
        f"Unsupported image encoding {getattr(msg, 'encoding', None)!r}. Publish rgb8 or bgr8 image_raw."
    )
