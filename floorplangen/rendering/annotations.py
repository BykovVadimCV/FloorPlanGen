"""Dimension annotation rendering (DESIGN §8.5)."""
from __future__ import annotations

import cv2
import numpy as np

from ..themes.base import EraTheme
from ..types import Annotation


def render_annotations(canvas: np.ndarray, annotations: list[Annotation],
                       theme: EraTheme, rng) -> None:
    channels = 1 if canvas.ndim == 2 else canvas.shape[2]
    if channels == 1:
        ink = int(round(0.299 * theme.ink_rgb[0]
                        + 0.587 * theme.ink_rgb[1]
                        + 0.114 * theme.ink_rgb[2]))
    else:
        ink = (int(theme.ink_rgb[2]), int(theme.ink_rgb[1]), int(theme.ink_rgb[0]))

    for ann in annotations:
        for poly in ann.polygons:
            if poly.is_empty:
                continue
            coords = np.array(list(poly.exterior.coords), dtype=np.int32)
            if coords.shape[0] >= 3:
                # Draw as filled polygon for arrowheads + text glyphs
                cv2.fillPoly(canvas, [coords], color=ink)
            else:
                # Polyline fallback
                cv2.polylines(canvas, [coords], False, color=ink, thickness=1,
                              lineType=cv2.LINE_AA)
