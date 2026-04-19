"""Icon compositing onto canvas (DESIGN §7.3, pre-multiplied alpha)."""
from __future__ import annotations

import numpy as np

from ..themes.base import EraTheme
from ..types import PlacedIcon


def render_icons(canvas: np.ndarray, icons: list[PlacedIcon],
                 theme: EraTheme, rng) -> None:
    h, w = canvas.shape[:2]
    channels = 1 if canvas.ndim == 2 else canvas.shape[2]

    for icon in icons:
        rgba = icon.rgba  # (ih, iw, 4) uint8, pre-multiplied
        ih, iw = rgba.shape[:2]
        x0, y0 = icon.top_left
        x1 = x0 + iw
        y1 = y0 + ih
        # Clip to canvas
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = iw - max(0, x1 - w)
        sy1 = ih - max(0, y1 - h)
        cx0 = max(0, x0)
        cy0 = max(0, y0)
        cx1 = min(w, x1)
        cy1 = min(h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        slab = rgba[sy0:sy1, sx0:sx1]  # (n, m, 4)
        alpha = slab[..., 3:4].astype(np.float32) / 255.0  # (n, m, 1)
        rgb = slab[..., :3].astype(np.float32)              # stored straight RGB
        # Convert to grayscale if canvas is single-channel
        if channels == 1:
            gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                    + 0.114 * rgb[..., 2])[..., None]
            fg = gray * alpha  # pre-multiply at composite-time
            dst = canvas[cy0:cy1, cx0:cx1].astype(np.float32)[..., None]
            out = fg + dst * (1.0 - alpha)
            canvas[cy0:cy1, cx0:cx1] = np.clip(out[..., 0], 0, 255).astype(np.uint8)
        else:
            # RGB to BGR for OpenCV canvas
            bgr = rgb[..., ::-1]
            fg = bgr * alpha
            dst = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
            out = fg + dst * (1.0 - alpha)
            canvas[cy0:cy1, cx0:cx1] = np.clip(out, 0, 255).astype(np.uint8)
