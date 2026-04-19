"""Footprint generation pipeline — 5 stages + validator (DESIGN §2.3–§2.6)."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from ..types import Footprint
from .primitives import build_primitive, scale_to_canvas, select_primitive
from .validator import validate


def _make_valid_polygon(geom) -> Polygon | None:
    """§2.8: make_valid can return GeometryCollection; extract largest Polygon."""
    try:
        from shapely import get_parts
        from shapely.ops import make_valid
    except Exception:  # pragma: no cover
        return geom if isinstance(geom, Polygon) else None
    try:
        result = make_valid(geom)
    except Exception:
        return None
    if isinstance(result, Polygon):
        return result if not result.is_empty else None
    if isinstance(result, MultiPolygon):
        polys = [p for p in result.geoms if not p.is_empty]
        return max(polys, key=lambda p: p.area) if polys else None
    # GeometryCollection fallback
    polys = [g for g in get_parts(result) if isinstance(g, Polygon) and not g.is_empty]
    return max(polys, key=lambda p: p.area) if polys else None


def _round_pixels(poly: Polygon) -> Polygon:
    coords = [(round(x), round(y)) for x, y in poly.exterior.coords]
    new_poly = Polygon(coords)
    if not new_poly.is_valid:
        new_poly = new_poly.buffer(0)
        if isinstance(new_poly, MultiPolygon):
            new_poly = max(new_poly.geoms, key=lambda p: p.area)
    new_poly = new_poly.simplify(0.5, preserve_topology=True)
    return new_poly if isinstance(new_poly, Polygon) else poly


def _cutout_rect(poly: Polygon, rng: np.random.Generator,
                 d_max: float) -> Polygon | None:
    """Sample a rectangular notch adjacent to the bounding box boundary."""
    minx, miny, maxx, maxy = poly.bounds
    bw = maxx - minx
    bh = maxy - miny
    side = int(rng.integers(0, 4))  # 0=left 1=right 2=bottom 3=top
    depth = float(rng.uniform(0.10, max(0.12, d_max)))
    width = float(rng.uniform(0.10, max(0.12, d_max)))
    if side in (0, 1):  # vertical side
        d_px = depth * bw
        w_px = width * bh
        y_pos = miny + rng.uniform(0.05, 0.95) * (bh - w_px)
        if side == 0:
            return box(minx, y_pos, minx + d_px, y_pos + w_px)
        return box(maxx - d_px, y_pos, maxx, y_pos + w_px)
    d_px = depth * bh
    w_px = width * bw
    x_pos = minx + rng.uniform(0.05, 0.95) * (bw - w_px)
    if side == 2:
        return box(x_pos, miny, x_pos + w_px, miny + d_px)
    return box(x_pos, maxy - d_px, x_pos + w_px, maxy)


def _extrusion_rect(poly: Polygon, rng: np.random.Generator) -> Polygon | None:
    minx, miny, maxx, maxy = poly.bounds
    bw = maxx - minx
    bh = maxy - miny
    side = int(rng.integers(0, 4))
    along = float(rng.uniform(0.06, 0.18))
    depth = float(rng.uniform(0.04, 0.12))
    if side in (0, 1):
        w_px = along * bh
        d_px = depth * bw
        y_pos = miny + rng.uniform(0.1, 0.9) * (bh - w_px)
        if side == 0:
            return box(minx - d_px, y_pos, minx, y_pos + w_px)
        return box(maxx, y_pos, maxx + d_px, y_pos + w_px)
    w_px = along * bw
    d_px = depth * bh
    x_pos = minx + rng.uniform(0.1, 0.9) * (bw - w_px)
    if side == 2:
        return box(x_pos, miny - d_px, x_pos + w_px, miny)
    return box(x_pos, maxy, x_pos + w_px, maxy + d_px)


def _bevel_corner(poly: Polygon, vertex_idx: int, size: float) -> Polygon | None:
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    n = len(coords)
    v = coords[vertex_idx % n]
    p = coords[(vertex_idx - 1) % n]
    q = coords[(vertex_idx + 1) % n]
    # Only chamfer 90°-ish convex corners
    vx, vy = v
    ax, ay = p[0] - vx, p[1] - vy
    bx, by = q[0] - vx, q[1] - vy
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    if la == 0 or lb == 0:
        return None
    ax, ay = ax / la, ay / la
    bx, by = bx / lb, by / lb
    return Polygon([
        (vx + ax * size, vy + ay * size),
        (vx, vy),
        (vx + bx * size, vy + by * size),
    ])


def generate_footprint(image_size: tuple[int, int],
                       aggression: float,
                       rng: np.random.Generator,
                       margin: int = 24,
                       forced_primitive: str | None = None) -> Footprint:
    """Produce a validated footprint polygon with integer-pixel vertices."""
    aggression = float(max(0.0, min(1.0, aggression)))
    aspect_ratio = float(rng.uniform(0.65, 1.55))
    if forced_primitive is not None:
        primitive_id = forced_primitive
    else:
        primitive_id = select_primitive(aggression, rng)
    primitive_fallback: str | None = None

    unit = build_primitive(primitive_id, rng)
    poly = scale_to_canvas(unit, image_size, aspect_ratio, margin)
    poly = _make_valid_polygon(poly) or _fallback_rect(image_size, margin)

    # Stage 2 — rectangular cutouts
    max_cuts = int(round(aggression * rng.triangular(0.5, 3.0, 5.0)))
    d_max = aggression * float(rng.uniform(0.15, 0.40))
    cuts_applied = 0
    z_reject_streak = 0
    for _ in range(max(0, max_cuts)):
        cut = _cutout_rect(poly, rng, d_max)
        if cut is None:
            continue
        candidate = _make_valid_polygon(poly.difference(cut))
        if candidate is not None and validate(candidate, image_size):
            poly = candidate
            cuts_applied += 1
            z_reject_streak = 0
        else:
            z_reject_streak += 1
            if primitive_id == "Z" and z_reject_streak >= 2:
                # §2.8 fallback: retry with L primitive
                primitive_fallback = "L"
                unit = build_primitive("L", rng)
                poly = scale_to_canvas(unit, image_size, aspect_ratio, margin)
                poly = _make_valid_polygon(poly) or _fallback_rect(image_size, margin)
                z_reject_streak = 0

    # Stage 3 — optional extrusions
    extrusions_applied = 0
    if aggression > 0.4 and cuts_applied < round(aggression * 2):
        for _ in range(int(rng.integers(1, 3))):
            ext = _extrusion_rect(poly, rng)
            if ext is None:
                continue
            candidate = _make_valid_polygon(unary_union([poly, ext]))
            if candidate is not None and validate(candidate, image_size):
                poly = candidate
                extrusions_applied += 1

    # Stage 4 — corner bevelling (only if primitive=BEVEL or high aggression)
    bevel_corners: list[int] = []
    if primitive_id == "BEVEL" or aggression > 0.65:
        coords = list(poly.exterior.coords)[:-1]
        n_corners = int(rng.integers(1, min(4, max(1, len(coords) - 3)) + 1))
        chosen = list(rng.choice(len(coords), size=min(n_corners, len(coords)),
                                 replace=False))
        size = float(rng.uniform(8.0, 22.0)) * min(image_size) / 512.0
        for idx in chosen:
            bev = _bevel_corner(poly, int(idx), size)
            if bev is None:
                continue
            candidate = _make_valid_polygon(poly.difference(bev))
            if candidate is not None and validate(candidate, image_size):
                poly = candidate
                bevel_corners.append(int(idx))

    # Stage 5 — normalise
    poly = _round_pixels(poly)
    poly = _make_valid_polygon(poly) or _fallback_rect(image_size, margin)
    if not validate(poly, image_size):
        poly = _fallback_rect(image_size, margin)
        primitive_fallback = primitive_fallback or "RECT_fallback"

    minx, miny, maxx, maxy = poly.bounds
    return Footprint(
        polygon=poly,
        primitive_id=primitive_id,
        n_cuts_applied=cuts_applied,
        n_extrusions_applied=extrusions_applied,
        bevel_corners=bevel_corners,
        aggression=aggression,
        bounding_box=(int(minx), int(miny), int(maxx), int(maxy)),
        primitive_fallback=primitive_fallback,
    )


def _fallback_rect(image_size: tuple[int, int], margin: int) -> Polygon:
    h, w = image_size
    return box(margin, margin, w - margin, h - margin)
