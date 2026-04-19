"""Stages B, E, F — Paper, scanner, illumination (DESIGN §10.3)."""
from __future__ import annotations

import cv2
import numpy as np


def apply_yellowing(img: np.ndarray, preset, rng: np.random.Generator,
                    force_mono: bool = False) -> np.ndarray:
    lo, hi = preset.yellowing_strength_range
    if hi <= 0.0 or force_mono or img.ndim == 2:
        return img
    strength = float(rng.uniform(lo, hi))
    if strength <= 0.0:
        return img
    f = img.astype(np.float32) / 255.0
    # BGR order (OpenCV): b=0, g=1, r=2
    r_shift = strength * float(rng.uniform(0.55, 1.00))
    g_shift = strength * float(rng.uniform(0.20, 0.50))
    b_shift = strength * float(rng.uniform(0.10, 0.30))
    f[..., 2] = np.clip(f[..., 2] + r_shift * (1.0 - f[..., 2]), 0.0, 1.0)
    f[..., 1] = np.clip(f[..., 1] + g_shift * (1.0 - f[..., 1]), 0.0, 1.0)
    f[..., 0] = np.clip(f[..., 0] - b_shift * f[..., 0], 0.0, 1.0)
    return (f * 255.0).astype(np.uint8)


def apply_foxing(img: np.ndarray, preset, rng: np.random.Generator,
                 force_mono: bool = False) -> np.ndarray:
    if rng.random() >= preset.foxing_prob:
        return img
    h, w = img.shape[:2]
    n = min(40, int(rng.poisson(12.0)))
    if n <= 0:
        return img
    out = img.astype(np.float32).copy()
    for _ in range(n):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        a = int(rng.integers(3, 15))
        b = int(rng.integers(3, 15))
        angle = float(rng.uniform(0.0, 180.0))
        overlay = np.zeros_like(out)
        if out.ndim == 2:
            cv2.ellipse(overlay, (cx, cy), (a, b), angle, 0, 360, color=80, thickness=-1)
        else:
            hue = rng.uniform(25, 40)
            sat = rng.uniform(0.4, 0.7)
            val = rng.uniform(0.55, 0.80)
            hsv = np.uint8([[[int(hue / 2), int(sat * 255), int(val * 255)]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.ellipse(overlay, (cx, cy), (a, b), angle, 0, 360,
                        color=tuple(bgr), thickness=-1)
        feather = max(1, int(min(a, b) * 0.4) | 1)
        overlay = cv2.GaussianBlur(overlay, (feather, feather), 0)
        mask = (overlay.sum(axis=-1, keepdims=True) > 0) if overlay.ndim == 3 else (overlay > 0)
        alpha = 0.45
        out = np.where(mask, out * (1 - alpha) + overlay * alpha, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_fold_lines(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    if rng.random() >= preset.fold_line_prob:
        return img
    h, w = img.shape[:2]
    n = int(rng.integers(1, 4))
    out = img.copy()
    for _ in range(n):
        horizontal = rng.random() < 0.6
        thick = int(rng.integers(1, 4))
        opacity = float(rng.uniform(0.3, 0.7))
        overlay = out.astype(np.float32).copy()
        if horizontal:
            y = int(rng.uniform(0.15, 0.85) * h)
            cv2.line(overlay, (0, y), (w, y),
                     color=(60, 60, 60) if out.ndim == 3 else 60,
                     thickness=thick)
        else:
            x = int(rng.uniform(0.15, 0.85) * w)
            cv2.line(overlay, (x, 0), (x, h),
                     color=(60, 60, 60) if out.ndim == 3 else 60,
                     thickness=thick)
        out = np.clip(out.astype(np.float32) * (1 - opacity)
                      + overlay * opacity, 0, 255).astype(np.uint8)
    return out


def apply_roller_marks(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    if rng.random() >= preset.roller_mark_prob:
        return img
    h, w = img.shape[:2]
    n = int(rng.integers(1, 5))
    out = img.astype(np.float32).copy()
    for _ in range(n):
        y = int(rng.uniform(0.0, 1.0) * h)
        band_h = int(rng.integers(2, 7))
        shift = float(rng.uniform(-12, 8))
        y0 = max(0, y - band_h // 2)
        y1 = min(h, y + band_h // 2 + 1)
        out[y0:y1] = np.clip(out[y0:y1] + shift, 0, 255)
    # Feather edges with a mild 1px blur
    out = cv2.GaussianBlur(out, (3, 3), 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_vignette(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    lo, hi = preset.vignette_strength_range
    if hi <= 0.0:
        return img
    strength = float(rng.uniform(lo, hi))
    if strength <= 0.0:
        return img
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)
    mask = 1.0 - strength * np.clip(r - 0.5, 0.0, 1.0)
    if img.ndim == 3:
        mask = mask[..., None]
    out = img.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_illumination(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    if rng.random() >= preset.illumination_gradient_prob:
        return img
    s = float(preset.illumination_gradient_strength)
    if s <= 0.0:
        return img
    h, w = img.shape[:2]
    grid = rng.uniform(1.0 - s, 1.0 + s, size=(4, 4)).astype(np.float32)
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    if img.ndim == 3:
        field = field[..., None]
    out = img.astype(np.float32) * field
    return np.clip(out, 0, 255).astype(np.uint8)
