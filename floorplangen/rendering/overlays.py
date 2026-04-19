"""Era-specific room-interior overlays.

These are graphic conventions applied *after* rooms, walls and openings have
been drawn but *before* text labels:

  - Balcony 45° hatching (Soviet only, per Guidelines §1.11 / §5.2).
  - Built-in wardrobe "X" diagonal fill (all eras, for storage rooms).

The overlays only touch the image canvas — they are not part of the class
mask (the mask rules are driven by icon footprints via `PlacedIcon`).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..linework import LineworkSpec, px_per_mm
from ..themes.base import EraTheme
from ..types import Room


def _ink_color(theme: EraTheme, channels: int):
    if channels == 1:
        v = int(round(0.299 * theme.ink_rgb[0]
                      + 0.587 * theme.ink_rgb[1]
                      + 0.114 * theme.ink_rgb[2]))
        return int(v)
    return (int(theme.ink_rgb[2]), int(theme.ink_rgb[1]), int(theme.ink_rgb[0]))


def _room_mask(canvas_shape, room: Room) -> np.ndarray:
    h, w = canvas_shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in room.polygon.exterior.coords],
        dtype=np.int32,
    )
    cv2.fillPoly(m, [pts], 1)
    return m


def _draw_hatch_45(canvas: np.ndarray, room: Room, spacing_px: float,
                   color, thickness: int) -> None:
    h, w = canvas.shape[:2]
    mask = _room_mask(canvas.shape, room)
    spacing = max(2.0, spacing_px)
    # 45° lines y = x - c: run c from -w to h in steps of spacing*sqrt(2)
    step = spacing * math.sqrt(2.0)
    c = -float(w)
    layer = np.zeros_like(canvas)
    while c <= h:
        x0 = 0
        y0 = int(round(c))
        x1 = w
        y1 = int(round(c + w))
        cv2.line(layer, (x0, y0), (x1, y1), color=color,
                 thickness=thickness, lineType=cv2.LINE_AA)
        c += step
    mask_bool = mask.astype(bool)
    if canvas.ndim == 2:
        canvas[mask_bool] = layer[mask_bool]
    else:
        canvas[mask_bool] = layer[mask_bool]


def render_era_overlays(canvas: np.ndarray, rooms: list[Room],
                        theme: EraTheme, spec: LineworkSpec) -> None:
    channels = 1 if canvas.ndim == 2 else canvas.shape[2]
    color = _ink_color(theme, channels)
    h, w = canvas.shape[:2]
    ppm = px_per_mm((h, w))

    if spec.balcony.fill == "hatch_45":
        spacing_px = spec.hatch_spacing_mm * ppm * 3.0  # balcony hatch is airier
        for room in rooms:
            if room.room_type != "balcony":
                continue
            _draw_hatch_45(canvas, room, spacing_px, color=color, thickness=1)
