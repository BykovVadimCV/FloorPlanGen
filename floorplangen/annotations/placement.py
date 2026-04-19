"""Dimension annotation placement (DESIGN §8)."""
from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon, box

from ..config import GeneratorConfig
from ..types import Annotation, Footprint, Room, WallGraph


P_NO_ANNOTATIONS = 0.30


def _arrow_polygon(ax: float, ay: float, bx: float, by: float,
                   theme_style: str) -> list[Polygon]:
    """Return a list of polygons for a dimension run from a→b."""
    dx = bx - ax
    dy = by - ay
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    shaft_thickness = 1.0
    polys: list[Polygon] = []

    # Shaft as a thin rectangle
    hw = shaft_thickness / 2.0
    shaft = Polygon([
        (ax + nx * hw, ay + ny * hw),
        (bx + nx * hw, by + ny * hw),
        (bx - nx * hw, by - ny * hw),
        (ax - nx * hw, ay - ny * hw),
    ])
    polys.append(shaft)

    # Terminals
    term_len = 8.0
    if theme_style == "tick_oblique":
        # Short oblique ticks at each end (45° relative to shaft)
        for ex, ey in [(ax, ay), (bx, by)]:
            cos45 = math.cos(math.pi / 4.0)
            sin45 = math.sin(math.pi / 4.0)
            tip1x = ex + (ux * cos45 + nx * sin45) * term_len / 2.0
            tip1y = ey + (uy * cos45 + ny * sin45) * term_len / 2.0
            tip2x = ex - (ux * cos45 + nx * sin45) * term_len / 2.0
            tip2y = ey - (uy * cos45 + ny * sin45) * term_len / 2.0
            tick = Polygon([
                (tip1x, tip1y), (ex + nx * 0.5, ey + ny * 0.5),
                (tip2x, tip2y), (ex - nx * 0.5, ey - ny * 0.5),
            ])
            polys.append(tick)
    else:
        # Filled arrowheads
        for ex, ey, sign in [(ax, ay, -1), (bx, by, 1)]:
            tipx = ex + sign * ux * 0.0
            tipy = ey + sign * uy * 0.0
            backx = ex - sign * ux * term_len
            backy = ey - sign * uy * term_len
            leftx = backx + nx * term_len * 0.4
            lefty = backy + ny * term_len * 0.4
            rightx = backx - nx * term_len * 0.4
            righty = backy - ny * term_len * 0.4
            head = Polygon([(tipx, tipy), (leftx, lefty), (rightx, righty)])
            polys.append(head)

    # Dummy text glyph block (rectangle) near the midpoint
    mx = (ax + bx) / 2.0
    my = (ay + by) / 2.0
    label = Polygon([
        (mx - 8, my - 5), (mx + 8, my - 5),
        (mx + 8, my + 5), (mx - 8, my + 5),
    ])
    polys.append(label)
    return polys


def place_dimensions(footprint: Footprint, rooms: list[Room], walls: WallGraph,
                     cfg: GeneratorConfig, rng: np.random.Generator,
                     era: str) -> list[Annotation]:
    if rng.random() < P_NO_ANNOTATIONS:
        return []
    style = "tick_oblique" if era == "soviet" else "arrow_filled"

    # Annotate a subset of the exterior (axis-aligned) walls with a dim line
    candidates = []
    for seg in walls.segments:
        if not seg.is_exterior or seg.is_diagonal:
            continue
        a = abs(seg.angle_deg)
        if not (a < 1.5 or abs(a - 90.0) < 1.5):
            continue
        if seg.centreline.length < 60.0:
            continue
        candidates.append(seg)

    if not candidates:
        return []

    rng.shuffle(candidates)
    max_ann = max(1, int(rng.integers(1, 5)))
    out: list[Annotation] = []
    offset = 14.0  # px offset from the wall
    for seg in candidates[:max_ann]:
        coords = list(seg.centreline.coords)
        (x0, y0), (x1, y1) = coords[0], coords[-1]
        dx = x1 - x0
        dy = y1 - y0
        L = math.hypot(dx, dy) or 1.0
        nx = -dy / L
        ny = dx / L
        # Push outward from the footprint centroid
        cx, cy = footprint.polygon.centroid.x, footprint.polygon.centroid.y
        mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        if (mid[0] - cx) * nx + (mid[1] - cy) * ny < 0:
            nx, ny = -nx, -ny
        ax = x0 + nx * offset
        ay = y0 + ny * offset
        bx = x1 + nx * offset
        by = y1 + ny * offset
        polys = _arrow_polygon(ax, ay, bx, by, theme_style=style)
        minx = min(p[0] for p in [(ax, ay), (bx, by)])
        miny = min(p[1] for p in [(ax, ay), (bx, by)])
        maxx = max(p[0] for p in [(ax, ay), (bx, by)])
        maxy = max(p[1] for p in [(ax, ay), (bx, by)])
        out.append(Annotation(polygons=polys, text="L",
                              bbox=(minx, miny, maxx, maxy)))
    return out
