"""Stages C + D — Blur and noise (DESIGN §10.3)."""
from __future__ import annotations

import cv2
import numpy as np


def apply_blur(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    img = _apply_gaussian_blur(img, preset, rng)
    img = _apply_motion_blur(img, preset, rng)
    return img


def apply_noise(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    img = _apply_gaussian_noise(img, preset, rng)
    img = _apply_salt_pepper(img, preset, rng)
    return img


def _apply_gaussian_blur(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    lo, hi = preset.gaussian_blur_sigma_range
    if hi <= 0.0:
        return img
    sigma = float(rng.uniform(lo, hi))
    if sigma < 0.3:
        return img
    k = max(3, int(round(sigma * 3.0)) | 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def _apply_motion_blur(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    lo, hi = preset.motion_blur_length_range
    if hi <= 0 or preset.motion_blur_prob <= 0.0:
        return img
    if rng.random() >= preset.motion_blur_prob:
        return img
    length = int(rng.integers(max(1, lo), hi + 1))
    if length < 2:
        return img
    angle = float(rng.uniform(0.0, 180.0))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2.0, length / 2.0), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    s = kernel.sum()
    if s <= 0:
        return img
    kernel /= s
    return cv2.filter2D(img, -1, kernel)


def _apply_gaussian_noise(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    lo, hi = preset.gaussian_noise_sigma_range
    if hi <= 0.0:
        return img
    sigma = float(rng.uniform(lo, hi))
    if sigma <= 0.0:
        return img
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_salt_pepper(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    p = float(preset.salt_pepper_prob)
    if p <= 0.0:
        return img
    h, w = img.shape[:2]
    mask = rng.random((h, w))
    salt = mask < (p / 2.0)
    pepper = mask > (1.0 - p / 2.0)
    out = img.copy()
    if out.ndim == 2:
        out[salt] = 255
        out[pepper] = 0
    else:
        out[salt] = 255
        out[pepper] = 0
    return out
