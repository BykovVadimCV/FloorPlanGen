"""Wall thickness sampling (DESIGN §5.4)."""
from __future__ import annotations

import numpy as np

# (px at 512 reference) — Thin / Medium / Wide
THICKNESS_CLASSES_PX_AT_512: dict[str, tuple[float, float]] = {
    "T": (3.0, 5.0),
    "M": (5.0, 8.0),
    "W": (8.0, 14.0),
}

# Per-image class probabilities for exterior / interior
EXTERIOR_CLASS_PROBS = {"T": 0.10, "M": 0.55, "W": 0.35}
INTERIOR_CLASS_PROBS = {"T": 0.40, "M": 0.50, "W": 0.10}


def sample_thickness(rng: np.random.Generator, image_size: tuple[int, int],
                     is_exterior: bool) -> float:
    probs = EXTERIOR_CLASS_PROBS if is_exterior else INTERIOR_CLASS_PROBS
    names = list(probs.keys())
    p = np.array([probs[n] for n in names], dtype=float)
    p = p / p.sum()
    cls = str(rng.choice(names, p=p))
    lo, hi = THICKNESS_CLASSES_PX_AT_512[cls]
    scale = min(image_size) / 512.0
    return float(rng.uniform(lo, hi)) * scale


def per_image_thickness_profile(rng: np.random.Generator,
                                image_size: tuple[int, int]) -> tuple[float, float]:
    """Fixed per-image exterior/interior thickness for consistency (§5.4.3)."""
    ext = sample_thickness(rng, image_size, is_exterior=True)
    intr = sample_thickness(rng, image_size, is_exterior=False)
    # Interior ≤ exterior to match real plans
    if intr > ext:
        intr = ext * 0.75
    return ext, intr
