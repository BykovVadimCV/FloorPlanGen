"""Room-type distribution (DESIGN §3.3)."""
from __future__ import annotations

import numpy as np

ROOM_TYPE_WEIGHTS: dict[str, float] = {
    "bedroom":  0.30,
    "living":   0.20,
    "kitchen":  0.18,
    "bathroom": 0.14,
    "hall":     0.10,
    "storage":  0.05,
    "balcony":  0.03,
}


def sample_room_type(rng: np.random.Generator,
                     used: dict[str, int] | None = None) -> str:
    names = list(ROOM_TYPE_WEIGHTS.keys())
    probs = np.array([ROOM_TYPE_WEIGHTS[n] for n in names], dtype=float)
    probs = probs / probs.sum()
    # Cap how many of each type (bathroom at most 2; kitchen at most 2)
    if used:
        caps = {"kitchen": 2, "bathroom": 2, "balcony": 2, "living": 2}
        for i, n in enumerate(names):
            if n in caps and used.get(n, 0) >= caps[n]:
                probs[i] = 0.0
        if probs.sum() <= 0:
            return "bedroom"
        probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))
