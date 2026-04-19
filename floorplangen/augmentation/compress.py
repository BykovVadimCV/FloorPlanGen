"""Stage G — JPEG compression round-trip (DESIGN §10.3 Stage G)."""
from __future__ import annotations

import cv2
import numpy as np


def apply_jpeg(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    if preset.jpeg_quality_range is None:
        return img
    if rng.random() >= preset.jpeg_prob:
        return img
    lo, hi = preset.jpeg_quality_range
    q = int(rng.integers(lo, hi + 1))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    if img.ndim == 2:
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    else:
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img
