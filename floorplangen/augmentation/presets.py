"""Augmentation presets (DESIGN §10.2) and top-level driver (§10.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blur_noise import apply_blur, apply_noise
from .compress import apply_jpeg
from .geometric import apply_geometric
from .paper import (
    apply_foxing,
    apply_fold_lines,
    apply_illumination,
    apply_roller_marks,
    apply_vignette,
    apply_yellowing,
)


@dataclass(frozen=True)
class AugmentationPreset:
    name: str

    # Geometric
    skew_angle_deg_max: float = 0.0
    perspective_warp_strength: float = 0.0
    barrel_distortion_k: float = 0.0

    # Blur
    gaussian_blur_sigma_range: tuple[float, float] = (0.0, 0.0)
    motion_blur_length_range: tuple[int, int] = (0, 0)
    motion_blur_prob: float = 0.0

    # Noise
    gaussian_noise_sigma_range: tuple[float, float] = (0.0, 0.0)
    salt_pepper_prob: float = 0.0

    # Compression
    jpeg_quality_range: tuple[int, int] | None = None
    jpeg_prob: float = 0.0

    # Paper / scanner
    yellowing_strength_range: tuple[float, float] = (0.0, 0.0)
    foxing_prob: float = 0.0
    fold_line_prob: float = 0.0
    roller_mark_prob: float = 0.0
    vignette_strength_range: tuple[float, float] = (0.0, 0.0)

    # Illumination
    illumination_gradient_prob: float = 0.0
    illumination_gradient_strength: float = 0.0


CLEAN = AugmentationPreset(
    name="clean",
    gaussian_noise_sigma_range=(0.0, 2.0),
    jpeg_quality_range=(92, 99),
    jpeg_prob=0.3,
)


MEDIUM = AugmentationPreset(
    name="medium",
    skew_angle_deg_max=3.5,
    perspective_warp_strength=0.008,
    barrel_distortion_k=0.012,
    gaussian_blur_sigma_range=(0.3, 1.2),
    motion_blur_length_range=(0, 3),
    motion_blur_prob=0.08,
    gaussian_noise_sigma_range=(1.0, 8.0),
    salt_pepper_prob=0.0005,
    jpeg_quality_range=(55, 85),
    jpeg_prob=0.75,
    yellowing_strength_range=(0.02, 0.14),
    foxing_prob=0.12,
    fold_line_prob=0.10,
    roller_mark_prob=0.18,
    vignette_strength_range=(0.0, 0.12),
    illumination_gradient_prob=0.20,
    illumination_gradient_strength=0.08,
)


HEAVY = AugmentationPreset(
    name="heavy",
    skew_angle_deg_max=5.0,
    perspective_warp_strength=0.018,
    barrel_distortion_k=0.025,
    gaussian_blur_sigma_range=(0.6, 2.0),
    motion_blur_length_range=(2, 6),
    motion_blur_prob=0.22,
    gaussian_noise_sigma_range=(4.0, 18.0),
    salt_pepper_prob=0.002,
    jpeg_quality_range=(40, 72),
    jpeg_prob=0.85,
    yellowing_strength_range=(0.10, 0.38),
    foxing_prob=0.55,
    fold_line_prob=0.45,
    roller_mark_prob=0.30,
    vignette_strength_range=(0.05, 0.28),
    illumination_gradient_prob=0.60,
    illumination_gradient_strength=0.20,
)


_REGISTRY: dict[str, AugmentationPreset] = {
    "clean": CLEAN,
    "medium": MEDIUM,
    "heavy": HEAVY,
}


def get_preset(name: str) -> AugmentationPreset:
    try:
        return _REGISTRY[name]
    except KeyError as ex:
        raise KeyError(f"Unknown augmentation preset: {name!r}") from ex


def apply_augmentation(canvas: np.ndarray, preset_name: str,
                       rng: np.random.Generator,
                       force_mono: bool = False) -> np.ndarray:
    """Apply the fixed-order degradation pipeline (DESIGN §10.3)."""
    preset = get_preset(preset_name)
    was_gray = (canvas.ndim == 2)

    # Stage A — Geometric
    img = apply_geometric(canvas, preset, rng)
    # Stage B — Paper / ink pre-effects
    img = apply_yellowing(img, preset, rng, force_mono=force_mono)
    img = apply_foxing(img, preset, rng, force_mono=force_mono)
    img = apply_fold_lines(img, preset, rng)
    # Stage C — Blur
    img = apply_blur(img, preset, rng)
    # Stage D — Noise
    img = apply_noise(img, preset, rng)
    # Stage E — Scanner surface
    img = apply_roller_marks(img, preset, rng)
    img = apply_vignette(img, preset, rng)
    # Stage F — Illumination
    img = apply_illumination(img, preset, rng)
    # Stage G — Compression
    img = apply_jpeg(img, preset, rng)

    # §10.4 — monochrome must stay monochrome
    if force_mono and img.ndim == 3:
        img = img.mean(axis=2).astype(np.uint8)
    elif was_gray and img.ndim == 3:
        img = img.mean(axis=2).astype(np.uint8)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img
