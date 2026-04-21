"""Wall graph construction.

Builds a WallGraph from a Footprint and list of Rooms in four passes:
  1. Exterior ring segmentation (with T-junction splitting).
  2. Interior room-boundary edges (deduplicating shared edges).
  3. Diagonal wall insertion (delegates to `diagonal.py`).
  4. (Informational) junction typing.

Per Guidelines §0.3 we do *not* couple drawn wall thickness to structural
role: each segment samples its own thickness from the era's LineworkSpec,
using the exterior flag only as a soft probabilistic hint.

Wall bands are buffered polygons — shapely union handles L/T/X junction
clean-up at render time, so no explicit miter/butt geometry is needed here.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Polygon

from ..config import GeneratorConfig
from ..linework import LineworkSpec, get_linework
from ..types import Footprint, Room, WallGraph, WallSegment
from .style import sample_wall_style
from .thickness import sample_wall_thickness_px


def _segments_from_ring(ring_coords: list[tuple[float, float]]
                        ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    out = []
    n = len(ring_coords)
    for i in range(n - 1):
        a = ring_coords[i]
        b = ring_coords[i + 1]
        if a == b:
            continue
        out.append((a, b))
    return out


def _canonical_edge(a: tuple[float, float],
                    b: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    a_r = (round(a[0]), round(a[1]))
    b_r = (round(b[0]), round(b[1]))
    return (a_r, b_r) if a_r <= b_r else (b_r, a_r)


def _split_edge_by_vertices(a: tuple[float, float], b: tuple[float, float],
                            interior_pts: set[tuple[int, int]]
                            ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length < 1.0:
        return [(a, b)]
    ts: list[float] = []
    for (px, py) in interior_pts:
        t = ((px - ax) * dx + (py - ay) * dy) / (length * length)
        if 0.01 < t < 0.99:
            cross = abs((px - ax) * dy - (py - ay) * dx) / length
            if cross < 1.5:
                ts.append(t)
    if not ts:
        return [(a, b)]
    ts = sorted(set(round(t, 4) for t in ts))
    pts = [a] + [(ax + t * dx, ay + t * dy) for t in ts] + [b]
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _angle_between(a, b) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def build_wall_graph(footprint: Footprint, rooms: list[Room],
                     cfg: GeneratorConfig,
                     rng_walls: np.random.Generator,
                     rng_diag: np.random.Generator | None = None,
                     era: str = "transitional") -> WallGraph:
    spec = get_linework(era)
    image_style = sample_wall_style(rng_walls, spec)

    # Pass 1 — exterior ring segmentation
    exterior_coords = list(footprint.polygon.exterior.coords)
    exterior_edges = _segments_from_ring(exterior_coords)

    interior_vertices: set[tuple[int, int]] = set()
    for r in rooms:
        for (x, y) in r.polygon.exterior.coords:
            interior_vertices.add((round(x), round(y)))

    graph = WallGraph()

    for a, b in exterior_edges:
        for sa, sb in _split_edge_by_vertices(a, b, interior_vertices):
            seg = _make_segment(sa, sb, cfg.image_size, image_style,
                                is_exterior=True, rng=rng_walls, spec=spec)
            if seg is not None:
                graph.segments.append(seg)

    # Pass 2 — interior boundaries, deduplicating shared edges
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for a, b in exterior_edges:
        seen.add(_canonical_edge(a, b))

    interior_edges_raw = []
    for room in rooms:
        coords = list(room.polygon.exterior.coords)
        for a, b in _segments_from_ring(coords):
            key = _canonical_edge(a, b)
            if key in seen:
                continue
            seen.add(key)
            interior_edges_raw.append((a, b))

    for a, b in interior_edges_raw:
        for sa, sb in _split_edge_by_vertices(a, b, interior_vertices):
            seg = _make_segment(sa, sb, cfg.image_size, image_style,
                                is_exterior=False, rng=rng_walls, spec=spec)
            if seg is not None:
                graph.segments.append(seg)

    # Pass 3 — diagonal wall injection
    if rng_diag is not None:
        from .diagonal import inject_diagonal_walls
        inject_diagonal_walls(graph, footprint, rooms, cfg, rng_diag,
                              image_style=image_style, spec=spec)

    return graph


def _make_segment(a, b, image_size: tuple[int, int], style: str,
                  is_exterior: bool, rng: np.random.Generator,
                  spec: LineworkSpec) -> WallSegment | None:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length < 2.0:
        return None
    centreline = LineString([a, b])
    angle = _angle_between(a, b)
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    is_diag = abs(angle) > 1.0 and abs(angle - 90.0) > 1.0 and abs(angle + 90.0) > 1.0
    thickness = sample_wall_thickness_px(rng, image_size, spec, is_exterior)
    return WallSegment(
        centreline=centreline,
        thickness_px=thickness,
        style=style,
        is_exterior=is_exterior,
        is_diagonal=is_diag,
        angle_deg=float(angle),
        no_annotate=False,
    )
