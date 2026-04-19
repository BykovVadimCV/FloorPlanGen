"""Wall rendering — solid / hollow / hatch (DESIGN §5.5, §9.3/4/5)."""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.ops import unary_union

from ..themes.base import EraTheme
from ..types import WallGraph
from .stroke import draw_hand_stroke


def _ink_color(theme: EraTheme, channels: int):
    if channels == 1:
        v = int(round(0.299 * theme.ink_rgb[0]
                      + 0.587 * theme.ink_rgb[1]
                      + 0.114 * theme.ink_rgb[2]))
        return int(v)
    return (int(theme.ink_rgb[2]), int(theme.ink_rgb[1]), int(theme.ink_rgb[0]))


def _draw_polygon_outline(canvas: np.ndarray, polygon, color, thickness: int) -> None:
    """Draw the exterior ring of a Shapely polygon as a polyline."""
    try:
        coords = np.array(
            [[int(round(x)), int(round(y))] for x, y in polygon.exterior.coords],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [coords], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
    except Exception:
        pass


def render_walls(canvas: np.ndarray, walls: WallGraph,
                 theme: EraTheme, rng) -> None:
    channels = 1 if canvas.ndim == 2 else canvas.shape[2]
    color = _ink_color(theme, channels)
    lw_lo, lw_hi = theme.line_weight_range

    # Hollow walls — collect all segments and render together or separately
    hollow_segs = [s for s in walls.segments if s.style == "hollow"]
    # 20% chance: individual bands (clean rectangle corners)
    # 80% chance: union mode (lines terminate at junctions, no crossing)
    junction_mode = "individual" if (rng.random() < 0.20) else "union"

    # Contour thickness for hollow walls and openings: 1px (thin line)
    contour_thk = 1

    if hollow_segs:
        # Give hollow-style walls thicker bands for more visible white space.
        # Scale thickness by 1.6× relative to nominal so the gap looks airy.
        bands = []
        for seg in hollow_segs:
            # Temporarily scale the band — don't mutate the segment, just buffer
            half = max(seg.thickness_px * 0.8, seg.thickness_px / 2.0 + 1.0)
            b = seg.centreline.buffer(half, cap_style=2, join_style=2)
            bands.append(b)

        if junction_mode == "union":
            # Merge all hollow bands — shared interior boundaries disappear,
            # lines cut off cleanly where walls meet.
            merged = unary_union(bands)
            geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
            for geom in geoms:
                if geom.is_empty:
                    continue
                _draw_polygon_outline(canvas, geom, color, contour_thk)
        else:
            # Individual bands: each wall drawn as its own rectangle outline
            for band in bands:
                if band.is_empty:
                    continue
                _draw_polygon_outline(canvas, band, color, contour_thk)

    # Solid and hatch walls — segment by segment
    for seg in walls.segments:
        coords = list(seg.centreline.coords)
        if len(coords) < 2:
            continue
        pts = np.array([[round(x), round(y)] for x, y in coords], dtype=np.int32)

        if seg.style == "solid":
            thk = max(1, int(round(seg.thickness_px)))
            if theme.stroke_model == "hand":
                for i in range(len(pts) - 1):
                    draw_hand_stroke(canvas,
                                     (float(pts[i][0]), float(pts[i][1])),
                                     (float(pts[i + 1][0]), float(pts[i + 1][1])),
                                     thickness=float(thk),
                                     color=color,
                                     rng=rng,
                                     overshoot_px=theme.overshoot_px,
                                     wobble_px=min(1.2, max(0.4, thk * 0.15)),
                                     thickness_jitter=0.30)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]),
                             color=color, thickness=thk, lineType=cv2.LINE_AA)

        elif seg.style == "hatch":
            thk = max(2, int(round(seg.thickness_px)))
            for i in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]),
                         color=color, thickness=thk, lineType=cv2.LINE_AA)
            for i in range(len(pts) - 1):
                a = pts[i].astype(float)
                b = pts[i + 1].astype(float)
                d = b - a
                L = math.hypot(d[0], d[1])
                if L < 6.0:
                    continue
                step = theme.hatch_spacing_px
                n_hatch = max(1, int(L / step))
                for k in range(n_hatch):
                    t = (k + 0.5) / n_hatch
                    p = a + d * t
                    p1 = (int(p[0] - 3), int(p[1] - 3))
                    p2 = (int(p[0] + 3), int(p[1] + 3))
                    cv2.line(canvas, p1, p2, color=color, thickness=1,
                             lineType=cv2.LINE_AA)
