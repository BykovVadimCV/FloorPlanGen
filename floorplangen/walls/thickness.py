"""Per-wall thickness sampling.

Guidelines §0.3 is explicit: BTI plans do not reliably distinguish load-bearing
walls from partitions by line thickness. We therefore draw a thickness for each
wall independently from `LineworkSpec.sample_wall_thickness_mm`, using the
exterior flag only as a soft hint toward the `capital` band.
"""
from __future__ import annotations

import numpy as np

from ..linework import LineworkSpec, px_per_mm


def sample_wall_thickness_px(rng: np.random.Generator,
                             image_size: tuple[int, int],
                             spec: LineworkSpec,
                             is_exterior: bool) -> float:
    """Return a wall-band thickness in pixels for a single wall segment."""
    mm = spec.sample_wall_thickness_mm(rng, hint_exterior=is_exterior)
    return float(mm * px_per_mm(image_size))
