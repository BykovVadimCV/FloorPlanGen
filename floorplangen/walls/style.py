"""Per-sample wall draw-style choice.

The style (solid / hollow / hatch) is chosen *once* per image so that all walls
share a consistent visual tier, matching how a single draftsman/CAD system
would have produced the sheet.
"""
from __future__ import annotations

import numpy as np

from ..linework import LineworkSpec


def sample_wall_style(rng: np.random.Generator, spec: LineworkSpec) -> str:
    return spec.sample_wall_style(rng)
