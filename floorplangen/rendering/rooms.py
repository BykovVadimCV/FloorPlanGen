"""Room fill rendering for colour_mode=='colour' or 'tint'."""
from __future__ import annotations

import numpy as np
import cv2

from ..themes.base import EraTheme
from ..types import Room


def render_rooms(canvas: np.ndarray, rooms: list[Room],
                 theme: EraTheme, rng: np.random.Generator) -> None:
    if canvas.ndim == 2:
        return  # monochrome — no room fills
    if theme.colour_mode == "mono":
        return
    if theme.colour_mode == "tint" and theme.room_tint_rgb:
        _fill_all_rooms(canvas, rooms, theme.room_tint_rgb)
        return
    if theme.colour_mode == "colour" and theme.room_palette:
        palette = theme.room_palette
        for i, room in enumerate(rooms):
            colour = palette[i % len(palette)]
            # BGR order for OpenCV
            bgr = (colour[2], colour[1], colour[0])
            _fill_room(canvas, room, bgr)
        return


def _fill_all_rooms(canvas: np.ndarray, rooms: list[Room],
                    rgb: tuple[int, int, int]) -> None:
    bgr = (rgb[2], rgb[1], rgb[0])
    for room in rooms:
        _fill_room(canvas, room, bgr)


def _fill_room(canvas: np.ndarray, room: Room, bgr: tuple[int, ...]) -> None:
    try:
        coords = np.array(
            [[int(round(x)), int(round(y))]
             for x, y in room.polygon.exterior.coords],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [coords], color=bgr)
    except Exception:
        pass
