"""Wall rendering — solid poché / hollow double-line / diagonal hatch.

Style mix is per-era (Guidelines §5.1); the hollow band's contour thickness
and the hatch spacing are scaled from the LineworkSpec paper-millimetre
values.

Openings (windows, doors) are subtracted from the hollow wall union before
the contour is drawn so the wall edge wraps cleanly around each opening —
i.e. openings *replace* a stretch of wall rather than overlaying it.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from ..linework import LineworkSpec, px_per_mm
from ..themes.base import EraTheme
from ..types import Opening, WallGraph
from .stroke import draw_hand_stroke


def _ink_color(theme: EraTheme, channels: int):
    if channels == 1:
        v = int(round(0.299 * theme.ink_rgb[0]
                      + 0.587 * theme.ink_rgb[1]
                      + 0.114 * theme.ink_rgb[2]))
        return int(v)
    return (int(theme.ink_rgb[2]), int(theme.ink_rgb[1]), int(theme.ink_rgb[0]))


def _rgb_to_cv(rgb: tuple[int, int, int], channels: int):
    if channels == 1:
        v = int(round(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]))
        return int(v)
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _draw_polygon_outline(canvas: np.ndarray, polygon: Polygon, color,
                          thickness: int) -> None:
    if polygon.is_empty:
        return
    try:
        ext = np.array(
            [[int(round(x)), int(round(y))] for x, y in polygon.exterior.coords],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [ext], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
        for interior in polygon.interiors:
            hole = np.array(
                [[int(round(x)), int(round(y))] for x, y in interior.coords],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [hole], isClosed=True, color=color,
                          thickness=thickness, lineType=cv2.LINE_AA)
    except Exception:
        pass


def _fill_polygon(canvas: np.ndarray, polygon: Polygon, color) -> None:
    if polygon.is_empty:
        return
    try:
        ext = np.array(
            [[int(round(x)), int(round(y))] for x, y in polygon.exterior.coords],
            dtype=np.int32,
        )
        holes = [
            np.array([[int(round(x)), int(round(y))] for x, y in r.coords],
                     dtype=np.int32)
            for r in polygon.interiors
        ]
        cv2.fillPoly(canvas, [ext], color=color, lineType=cv2.LINE_AA)
        for h in holes:
            # Carve holes back to paper by drawing a paper fill — caller should
            # handle per-polygon holes via subtraction before this point.
            cv2.fillPoly(canvas, [h], color=color, lineType=cv2.LINE_AA)
    except Exception:
        pass


def _iter_geoms(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g


def render_walls(canvas: np.ndarray, walls: WallGraph,
                 theme: EraTheme, rng,
                 spec: LineworkSpec,
                 openings: list[Opening] | None = None,
                 pastel_mode: str | None = None,
                 pastel_rgb: tuple[int, int, int] | None = None) -> None:
    """Render wall bands. Openings are subtracted so wall edges wrap around them.

    pastel_mode:
      - None         : monochrome ink rendering.
      - "solid"      : walls filled solid in pastel (ink-colour swapped).
      - "hollow_fill": hollow walls get pastel interior fill but keep black
                       outlines; solid walls still rendered in pastel too so
                       they read as a cohesive palette.
    """
    channels = 1 if canvas.ndim == 2 else canvas.shape[2]
    ink = _ink_color(theme, channels)
    pastel = _rgb_to_cv(pastel_rgb, channels) if pastel_rgb else None
    h, w = canvas.shape[:2]
    ppm = px_per_mm((h, w))
    contour_thk = max(1, int(round(spec.wall.outline_px * ppm)))

    # Build opening-subtraction mask: a single union polygon covering every
    # opening slab so its geometry is excluded from wall contours and fills.
    openings_union = None
    if openings:
        polys = [op.polygon for op in openings if op.polygon is not None
                 and not op.polygon.is_empty]
        if polys:
            try:
                openings_union = unary_union(polys)
            except Exception:
                openings_union = None

    # ── Hollow walls — render outline of the band (minus openings). ─────────
    hollow_segs = [s for s in walls.segments if s.style == "hollow"]
    if hollow_segs:
        bands = []
        for seg in hollow_segs:
            half = max(seg.thickness_px * 0.5, 0.6)
            b = seg.centreline.buffer(half, cap_style=2, join_style=2)
            bands.append(b)
        merged = unary_union(bands)

        if openings_union is not None:
            try:
                merged = merged.difference(openings_union)
            except Exception:
                pass

        outline_color = ink
        fill_color = None
        if pastel_mode == "hollow_fill" and pastel is not None:
            fill_color = pastel
        elif pastel_mode == "solid" and pastel is not None:
            outline_color = pastel
            fill_color = pastel

        for geom in _iter_geoms(merged):
            if fill_color is not None:
                ext = np.array(
                    [[int(round(x)), int(round(y))] for x, y in geom.exterior.coords],
                    dtype=np.int32,
                )
                holes = [
                    np.array([[int(round(x)), int(round(y))] for x, y in r.coords],
                             dtype=np.int32)
                    for r in geom.interiors
                ]
                cv2.fillPoly(canvas, [ext], color=fill_color, lineType=cv2.LINE_AA)
                for hole in holes:
                    # nothing to do — outline pass paints them black/pastel too
                    pass
            _draw_polygon_outline(canvas, geom, outline_color, contour_thk)

    # ── Solid poché / hatch walls — segment by segment. ────────────────────
    solid_colour = pastel if pastel is not None else ink
    for seg in walls.segments:
        coords = list(seg.centreline.coords)
        if len(coords) < 2:
            continue
        pts = np.array([[round(x), round(y)] for x, y in coords], dtype=np.int32)

        if seg.style == "solid":
            thk = max(1, int(round(seg.thickness_px)))
            if theme.stroke_model == "hand" and pastel_mode is None:
                wobble = min(1.2, max(0.4, spec.stroke.wobble_amp_mm * ppm))
                overshoot = spec.stroke.overshoot_mm * ppm
                for i in range(len(pts) - 1):
                    over = overshoot if rng.random() < spec.stroke.overshoot_prob else 0.0
                    draw_hand_stroke(canvas,
                                     (float(pts[i][0]), float(pts[i][1])),
                                     (float(pts[i + 1][0]), float(pts[i + 1][1])),
                                     thickness=float(thk),
                                     color=solid_colour,
                                     rng=rng,
                                     overshoot_px=over,
                                     wobble_px=wobble,
                                     thickness_jitter=spec.stroke.thickness_jitter)
            else:
                # Draw as a buffered polygon so caps are flat/mitre, not round.
                half = max(thk * 0.5, 0.6)
                band = seg.centreline.buffer(half, cap_style=2, join_style=2)
                if openings_union is not None:
                    try:
                        band = band.difference(openings_union)
                    except Exception:
                        pass
                for geom in _iter_geoms(band):
                    ext = np.array(
                        [[int(round(x)), int(round(y))] for x, y in geom.exterior.coords],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(canvas, [ext], color=solid_colour,
                                 lineType=cv2.LINE_AA)

        elif seg.style == "hatch":
            thk = max(1, int(round(seg.thickness_px)))
            # Outline band as two parallel lines + diagonal fill
            half = max(thk * 0.5, 0.6)
            band = seg.centreline.buffer(half, cap_style=2, join_style=2)
            if openings_union is not None:
                try:
                    band = band.difference(openings_union)
                except Exception:
                    pass
            for geom in _iter_geoms(band):
                _draw_polygon_outline(canvas, geom, solid_colour, contour_thk)
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
                    cv2.line(canvas, p1, p2, color=solid_colour, thickness=1,
                             lineType=cv2.LINE_AA)
