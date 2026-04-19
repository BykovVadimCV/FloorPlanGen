"""Canvas initialisation."""
from __future__ import annotations

import numpy as np

from ..themes.base import EraTheme


def render_canvas(image_size: tuple[int, int], theme: EraTheme,
                  force_mono: bool) -> np.ndarray:
    h, w = image_size
    if force_mono or theme.colour_mode == "mono":
        # Return grayscale single-channel
        v = int(round(0.299 * theme.paper_rgb[0]
                      + 0.587 * theme.paper_rgb[1]
                      + 0.114 * theme.paper_rgb[2]))
        return np.full((h, w), v, dtype=np.uint8)
    # 3-channel
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[..., 0] = theme.paper_rgb[0]
    canvas[..., 1] = theme.paper_rgb[1]
    canvas[..., 2] = theme.paper_rgb[2]
    return canvas
