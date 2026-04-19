"""Stage A — Geometric distortions (DESIGN §10.3 Stage A)."""
from __future__ import annotations

import cv2
import numpy as np


def apply_geometric(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    img = _apply_skew(img, preset, rng)
    img = _apply_perspective(img, preset, rng)
    img = _apply_barrel(img, preset, rng)
    return img


def _border_value(img: np.ndarray) -> tuple[int, ...] | int:
    if img.ndim == 2:
        return 255
    return (255, 255, 255)


def _apply_skew(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    max_deg = float(preset.skew_angle_deg_max)
    if max_deg <= 0.0:
        return img
    angle = float(rng.uniform(-max_deg, max_deg))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=_border_value(img))


def _apply_perspective(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    strength = float(preset.perspective_warp_strength)
    if strength <= 0.0:
        return img
    h, w = img.shape[:2]
    dx = strength * w
    dy = strength * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - rng.uniform(0, dx), h - rng.uniform(0, dy)],
        [rng.uniform(0, dx), h - rng.uniform(0, dy)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=_border_value(img))


def _apply_barrel(img: np.ndarray, preset, rng: np.random.Generator) -> np.ndarray:
    k = float(preset.barrel_distortion_k)
    if abs(k) < 1e-6:
        return img
    h, w = img.shape[:2]
    fx = fy = max(h, w)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.array([k, 0.0, 0.0, 0.0], dtype=np.float32)
    return cv2.undistort(img, K, dist, newCameraMatrix=K)
