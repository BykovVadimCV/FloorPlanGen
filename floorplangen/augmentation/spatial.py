"""Spatial augmentation applied identically to image AND mask.

Includes random horizontal/vertical flip and small rotation. These are
kept separate from the degradation pipeline (blur, noise, JPEG) which
is image-only. YOLO label coordinates are recalculated from the
transformed mask pixels.

The flip/rotation RNG is seeded from rng["augment"] but drawn before the
degradation ops so that changing the degradation preset does not shift
the spatial transform. Use a fixed offset (spawn a sub-generator).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..types import PIXEL_BACKGROUND


def apply_spatial(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    flip_prob: float = 0.5,
    vflip_prob: float = 0.3,
    rotate_max_deg: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply flip + small rotation to both image and mask.

    Returns (image, mask, transform_meta) where transform_meta records what
    was applied so callers can undo or log the operation.
    """
    meta: dict = {}
    h, w = image.shape[:2]

    # Horizontal flip
    if rng.random() < flip_prob:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        meta["hflip"] = True

    # Vertical flip
    if rng.random() < vflip_prob:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        meta["vflip"] = True

    # Small rotation (use INTER_NEAREST for mask to preserve exact pixel values)
    if rotate_max_deg > 0.0:
        angle = float(rng.uniform(-rotate_max_deg, rotate_max_deg))
        if abs(angle) > 0.05:
            cx, cy = w / 2.0, h / 2.0
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            bg_val = _border_value_for(image)
            image = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=bg_val)
            mask = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=int(PIXEL_BACKGROUND))
            meta["rotate_deg"] = angle

    return image, mask, meta


def _border_value_for(img: np.ndarray) -> tuple[int, ...] | int:
    if img.ndim == 2:
        return 255
    return (255, 255, 255)


def yolo_from_mask(mask: np.ndarray,
                   class_to_pixel: dict[int, int],
                   image_size: tuple[int, int]) -> list[str]:
    """Recompute YOLO labels from the (possibly-transformed) mask.

    class_to_pixel: {yolo_class_id: pixel_value}
    e.g. {0: 128, 1: 192}  (window=128, door=192)
    """
    h, w = image_size
    lines: list[str] = []
    for cls_id, pixel_val in class_to_pixel.items():
        region = (mask == pixel_val).astype(np.uint8)
        if region.sum() == 0:
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 2 or bh < 2:
                continue
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines
