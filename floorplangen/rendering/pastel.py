"""Pastel wall-colour picker (Guidelines §5.8).

~5% of plans ship with coloured walls; two stylistic variants:

  - "solid"       : walls and openings both rendered in the same pastel ink.
  - "hollow_fill" : only the hollow-wall interior takes the pastel tint while
                    outlines and openings remain black — a common modern-era
                    CAD convention.
"""
from __future__ import annotations

import numpy as np


_PASTEL_PALETTE: tuple[tuple[int, int, int], ...] = (
    (214, 163, 163),  # dusty rose
    (196, 176, 136),  # warm sand
    (156, 175, 136),  # sage
    (136, 168, 196),  # powder blue
    (186, 152, 196),  # lavender
    (210, 180, 140),  # tan
    (164, 188, 166),  # mint
    (204, 170, 140),  # peach
)


def pick_pastel(rng: np.random.Generator
                ) -> tuple[str, tuple[int, int, int]]:
    """Return (mode, rgb) for the current sample."""
    mode = "solid" if rng.random() < 0.60 else "hollow_fill"
    rgb = _PASTEL_PALETTE[int(rng.integers(0, len(_PASTEL_PALETTE)))]
    return mode, rgb
