"""Opening placement (DESIGN §6).

Windows-first placement on eligible (axis-aligned, non-diagonal, long-enough) wall
segments. Doors placed on interior walls with a swing arc. Diagonal walls are
explicitly excluded (§4.7, no_annotate=True).
"""
from __future__ import annotations

import math

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union

from ..config import GeneratorConfig
from ..types import Footprint, Opening, Room, WallGraph, WallSegment

MIN_WINDOW_LEN_PX_AT_512 = 50.0
MIN_DOOR_WALL_LEN_PX_AT_512 = 55.0
WINDOW_LEN_PX_AT_512 = (40.0, 110.0)
DOOR_WIDTH_PX_AT_512 = (32.0, 48.0)


def _scale(image_size: tuple[int, int], v: float) -> float:
    return v * min(image_size) / 512.0


def _eligible_for_opening(seg: WallSegment, image_size) -> bool:
    if seg.no_annotate or seg.is_diagonal:
        return False
    # Axis-aligned only (§6.2): require angle ~0 or ~90 within ±1°
    a = abs(seg.angle_deg)
    if not (a < 1.5 or abs(a - 90.0) < 1.5):
        return False
    if seg.centreline.length < _scale(image_size, MIN_WINDOW_LEN_PX_AT_512):
        return False
    return True


def _segment_axis_aligned_bbox(seg: WallSegment) -> tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) along-axis bounds of the centreline."""
    xs = [p[0] for p in seg.centreline.coords]
    ys = [p[1] for p in seg.centreline.coords]
    return min(xs), min(ys), max(xs), max(ys)


def _slab_polygon_along_wall(seg: WallSegment, along_t0: float, along_t1: float,
                             thickness: float) -> Polygon:
    """Rectangle slab aligned with wall centreline, spanning t in [t0, t1]."""
    (x0, y0), (x1, y1) = list(seg.centreline.coords)[0], list(seg.centreline.coords)[-1]
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    half = thickness / 2.0
    p_a_x = x0 + ux * along_t0
    p_a_y = y0 + uy * along_t0
    p_b_x = x0 + ux * along_t1
    p_b_y = y0 + uy * along_t1
    return Polygon([
        (p_a_x + nx * half, p_a_y + ny * half),
        (p_b_x + nx * half, p_b_y + ny * half),
        (p_b_x - nx * half, p_b_y - ny * half),
        (p_a_x - nx * half, p_a_y - ny * half),
    ])


def _door_swing_arc(seg: WallSegment, along_t0: float, along_t1: float,
                    thickness: float, open_inward: bool) -> Polygon:
    """Pie-slice swing arc attached at the hinge point."""
    coords = list(seg.centreline.coords)
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    if not open_inward:
        nx, ny = -nx, -ny
    hinge = (x0 + ux * along_t0, y0 + uy * along_t0)
    door_w = along_t1 - along_t0
    # Create a 90° pie sector
    n_steps = 14
    pts = [hinge]
    for i in range(n_steps + 1):
        ang = (math.pi / 2.0) * i / n_steps
        # Direction vector: starts along the wall (toward the door leaf)
        # and sweeps toward the normal
        cx = ux * math.cos(ang) + nx * math.sin(ang)
        cy = uy * math.cos(ang) + ny * math.sin(ang)
        pts.append((hinge[0] + cx * door_w, hinge[1] + cy * door_w))
    pts.append(hinge)
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if isinstance(poly, Polygon) else Polygon()


def place_openings(footprint: Footprint, rooms: list[Room],
                   walls: WallGraph, cfg: GeneratorConfig,
                   rng: np.random.Generator, era: str) -> list[Opening]:
    openings: list[Opening] = []
    occupied = []  # list of (wall_idx, t0, t1) to prevent overlap on same wall

    # ── Windows first on exterior walls (§6.7) ────────────────────────────
    exterior_indices = [i for i, s in enumerate(walls.segments)
                        if s.is_exterior and _eligible_for_opening(s, cfg.image_size)]
    rng.shuffle(exterior_indices)
    win_lo, win_hi = [_scale(cfg.image_size, v) for v in WINDOW_LEN_PX_AT_512]
    min_exterior_cap = max(1, int(len(exterior_indices) * 0.55))

    for i in exterior_indices[:min_exterior_cap]:
        seg = walls.segments[i]
        L = seg.centreline.length
        if L < win_lo * 1.2:
            continue
        w_hi = min(win_hi, L * 0.7)
        if w_hi <= win_lo:
            continue
        w = float(rng.uniform(win_lo, w_hi))
        margin = max(8.0, L * 0.10)
        t_hi = L - margin - w
        if t_hi <= margin:
            continue
        t0 = float(rng.uniform(margin, t_hi))
        t1 = t0 + w
        slab = _slab_polygon_along_wall(seg, t0, t1, seg.thickness_px * 1.05)
        openings.append(Opening(kind="window", polygon=slab, wall_index=i,
                                bbox=slab.bounds))
        occupied.append((i, t0, t1))

    # ── Doors on interior walls (§6.6) ─────────────────────────────────────
    interior_indices = [i for i, s in enumerate(walls.segments)
                        if (not s.is_exterior) and _eligible_for_opening(s, cfg.image_size)]
    rng.shuffle(interior_indices)
    door_lo, door_hi = [_scale(cfg.image_size, v) for v in DOOR_WIDTH_PX_AT_512]
    max_doors = min(len(interior_indices), max(1, len(rooms) - 1))

    for i in interior_indices[:max_doors]:
        seg = walls.segments[i]
        L = seg.centreline.length
        min_len = _scale(cfg.image_size, MIN_DOOR_WALL_LEN_PX_AT_512)
        if L < min_len:
            continue
        w_hi = min(door_hi, L * 0.55)
        if w_hi <= door_lo:
            continue
        w = float(rng.uniform(door_lo, w_hi))
        margin = max(8.0, L * 0.10)
        t_hi = L - margin - w
        if t_hi <= margin:
            continue
        t0 = float(rng.uniform(margin, t_hi))
        t1 = t0 + w
        slab = _slab_polygon_along_wall(seg, t0, t1, seg.thickness_px * 1.05)
        open_inward = bool(rng.random() < 0.5)
        try:
            arc = _door_swing_arc(seg, t0, t1, seg.thickness_px, open_inward)
            # Clip the arc to the footprint so it never extends outside
            arc = arc.intersection(footprint.polygon)
            if not isinstance(arc, Polygon) or arc.is_empty:
                arc = None
        except Exception:
            arc = None
        openings.append(Opening(kind="door", polygon=slab, wall_index=i,
                                swing_arc=arc, bbox=slab.bounds))
        occupied.append((i, t0, t1))

    return openings
