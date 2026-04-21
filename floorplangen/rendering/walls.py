"""Wall rendering — solid poché / hollow double-line / diagonal hatch.

Style mix is per-era (Guidelines §5.1); the hollow band's contour thickness
and the hatch spacing are scaled from the LineworkSpec paper-millimetre
values.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.ops import unary_union

from ..linework import LineworkSpec, px_per_mm
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
                 theme: EraTheme, rng,
                 spec: LineworkSpec) -> None:
    channels = 1 if canvas.ndim == 2 else canvas.shape[2]
    color = _ink_color(theme, channels)
    h, w = canvas.shape[:2]
    ppm = px_per_mm((h, w))
    contour_thk = max(1, int(round(spec.wall.outline_px * ppm)))

    # Hollow walls — render as polygon outlines so the interior stays paper.
    hollow_segs = [s for s in walls.segments if s.style == "hollow"]
    junction_mode = "individual" if (rng.random() < 0.20) else "union"

    if hollow_segs:
        bands = []
        for seg in hollow_segs:
            half = max(seg.thickness_px * 0.5, 1.0)
            b = seg.centreline.buffer(half, cap_style=2, join_style=2)
            bands.append(b)

        if junction_mode == "union":
            merged = unary_union(bands)
            geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
            for geom in geoms:
                if geom.is_empty:
                    continue
                _draw_polygon_outline(canvas, geom, color, contour_thk)
        else:
            for band in bands:
                if band.is_empty:
                    continue
                _draw_polygon_outline(canvas, band, color, contour_thk)

    # Solid poché and hatch walls — segment by segment.
    for seg in walls.segments:
        coords = list(seg.centreline.coords)
        if len(coords) < 2:
            continue
        pts = np.array([[round(x), round(y)] for x, y in coords], dtype=np.int32)

        if seg.style == "solid":
            thk = max(1, int(round(seg.thickness_px)))
            if theme.stroke_model == "hand":
                wobble = min(1.2, max(0.4, spec.stroke.wobble_amp_mm * ppm))
                overshoot = spec.stroke.overshoot_mm * ppm
                for i in range(len(pts) - 1):
                    over = overshoot if rng.random() < spec.stroke.overshoot_prob else 0.0
                    draw_hand_stroke(canvas,
                                     (float(pts[i][0]), float(pts[i][1])),
                                     (float(pts[i + 1][0]), float(pts[i + 1][1])),
                                     thickness=float(thk),
                                     color=color,
                                     rng=rng,
                                     overshoot_px=over,
                                     wobble_px=wobble,
                                     thickness_jitter=spec.stroke.thickness_jitter)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]),
                             color=color, thickness=thk, lineType=cv2.LINE_AA)

        elif seg.style == "hatch":
            thk = max(2, int(round(seg.thickness_px)))
            for i in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]),
                         color=color, thickness=thk, lineType=cv2.LINE_AA)
            step = max(2.0, spec.hatch_spacing_mm * ppm)
            for i in range(len(pts) - 1):
                a = pts[i].astype(float)
                b = pts[i + 1].astype(float)
                d = b - a
                L = math.hypot(d[0], d[1])
                if L < 6.0:
                    continue
                n_hatch = max(1, int(L / step))
                for k in range(n_hatch):
                    t = (k + 0.5) / n_hatch
                    p = a + d * t
                    p1 = (int(p[0] - 3), int(p[1] - 3))
                    p2 = (int(p[0] + 3), int(p[1] + 3))
                    cv2.line(canvas, p1, p2, color=color, thickness=1,
                             lineType=cv2.LINE_AA)
