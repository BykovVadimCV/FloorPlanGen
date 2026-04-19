"""Opening symbol rendering (DESIGN §6.9).

Window and door line thickness is always 1px (matching hollow wall contours).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..themes.base import EraTheme
from ..types import Opening, WallGraph

# Fixed contour thickness — must match hollow wall contour_thk in walls.py
_CONTOUR_THK = 1


def _ink_color(theme: EraTheme, channels: int):
    if channels == 1:
        v = int(round(0.299 * theme.ink_rgb[0]
                      + 0.587 * theme.ink_rgb[1]
                      + 0.114 * theme.ink_rgb[2]))
        return int(v)
    return (int(theme.ink_rgb[2]), int(theme.ink_rgb[1]), int(theme.ink_rgb[0]))


def _paper_color(theme: EraTheme, channels: int):
    if channels == 1:
        v = int(round(0.299 * theme.paper_rgb[0]
                      + 0.587 * theme.paper_rgb[1]
                      + 0.114 * theme.paper_rgb[2]))
        return int(v)
    return (int(theme.paper_rgb[2]), int(theme.paper_rgb[1]), int(theme.paper_rgb[0]))


def render_openings(canvas: np.ndarray, openings: list[Opening],
                    walls: WallGraph, theme: EraTheme, rng) -> None:
    channels = 1 if canvas.ndim == 2 else canvas.shape[2]
    ink = _ink_color(theme, channels)
    paper = _paper_color(theme, channels)
    thk = _CONTOUR_THK  # always matches hollow wall contour

    for op in openings:
        # Fill opening with paper colour (white-out the wall band)
        coords = np.array(list(op.polygon.exterior.coords), dtype=np.int32)
        cv2.fillPoly(canvas, [coords], color=paper)

        # Draw outline of the opening rectangle
        cv2.polylines(canvas, [coords], isClosed=True,
                      color=ink, thickness=thk, lineType=cv2.LINE_AA)

        xs = coords[:, 0]
        ys = coords[:, 1]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        if op.kind == "window":
            # Two parallel sill lines at 1/3 and 2/3 along the short axis
            if (x1 - x0) >= (y1 - y0):  # horizontal opening
                y_a = int(y0 + (y1 - y0) / 3)
                y_b = int(y0 + 2 * (y1 - y0) / 3)
                cv2.line(canvas, (x0, y_a), (x1, y_a),
                         color=ink, thickness=thk, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x0, y_b), (x1, y_b),
                         color=ink, thickness=thk, lineType=cv2.LINE_AA)
            else:
                x_a = int(x0 + (x1 - x0) / 3)
                x_b = int(x0 + 2 * (x1 - x0) / 3)
                cv2.line(canvas, (x_a, y0), (x_a, y1),
                         color=ink, thickness=thk, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x_b, y0), (x_b, y1),
                         color=ink, thickness=thk, lineType=cv2.LINE_AA)

        elif op.kind == "door":
            if op.swing_arc is not None and not op.swing_arc.is_empty:
                try:
                    arc_coords = np.array(
                        list(op.swing_arc.exterior.coords), dtype=np.int32
                    )
                    cv2.polylines(canvas, [arc_coords], isClosed=True,
                                  color=ink, thickness=thk, lineType=cv2.LINE_AA)
                except Exception:
                    pass
