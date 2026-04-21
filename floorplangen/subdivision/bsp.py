"""Recursive BSP subdivision (DESIGN §3.2, §3.5, §3.6)."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ..config import GeneratorConfig
from ..types import Footprint, Room
from .room_types import sample_room_type
from .split import half_plane_split

MIN_AREA_PX_AT_512 = 1800.0
MAX_ASPECT = 3.6
MIN_SIDE_PX_AT_512 = 22.0


def _scale_min(image_size, base):
    return base * (min(image_size) / 512.0) ** 2


def _scale_linear(image_size, base):
    return base * min(image_size) / 512.0


def _pca_axis(coords: np.ndarray, rng: np.random.Generator,
              jitter_deg: float = 8.0) -> float:
    """Return split-axis angle (radians) perpendicular to the PCA dominant axis.

    §3.5.2: we *split across* the longest extent, so the split-line orientation
    runs along the minor axis (i.e., perpendicular to the dominant direction).
    """
    if coords.shape[0] < 2:
        return float(rng.uniform(0, math.pi))
    mean = coords.mean(axis=0)
    centred = coords - mean
    cov = np.cov(centred.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return float(rng.uniform(0, math.pi))
    # dominant direction
    major = eigvecs[:, np.argmax(eigvals)]
    angle_along_major = math.atan2(major[1], major[0])
    # Split line runs perpendicular to the dominant axis (cuts across it)
    split_angle = angle_along_major + math.pi / 2.0
    # §3.5.4 axis jitter
    split_angle += math.radians(rng.uniform(-jitter_deg, jitter_deg))
    return split_angle


def _mrr_sides(poly: Polygon) -> tuple[float, float]:
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)[:4]
        s1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        s2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        return min(s1, s2), max(s1, s2)
    except Exception:
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        return min(w, h), max(w, h)


def _accept_split(cell: Polygon, a: Polygon | None, b: Polygon | None,
                  image_size) -> bool:
    if a is None or b is None:
        return False
    if a.is_empty or b.is_empty:
        return False
    total = a.area + b.area
    # Allow small loss from rounding but not huge
    if total < 0.95 * cell.area:
        return False
    min_area = _scale_min(image_size, MIN_AREA_PX_AT_512)
    if a.area < min_area or b.area < min_area:
        return False
    min_side = _scale_linear(image_size, MIN_SIDE_PX_AT_512)
    for p in (a, b):
        s_min, s_max = _mrr_sides(p)
        if s_min < min_side:
            return False
        if s_min > 0 and s_max / s_min > MAX_ASPECT:
            return False
    return True


def _subdivide_cell(cell: Polygon, depth: int, max_depth: int,
                    target_count: int, cells: list[Polygon],
                    rng: np.random.Generator, image_size) -> None:
    # Termination: target count, depth limit, or min-area guard
    if len(cells) + 1 >= target_count or depth >= max_depth:
        cells.append(cell)
        return
    min_area = _scale_min(image_size, MIN_AREA_PX_AT_512)
    if cell.area < 2.0 * min_area:
        cells.append(cell)
        return

    # Simplify for PCA stability on STAIR-like geometry (§3.8)
    simp = cell.simplify(2.0, preserve_topology=True)
    if not isinstance(simp, Polygon) or simp.is_empty:
        simp = cell
    coords = np.array(list(simp.exterior.coords)[:-1], dtype=float)

    # Axis-aligned splits first — keeps interior walls rectangular.
    # PCA angle is a last resort so diagonal BSP walls are rare.
    angle_candidates = [
        math.pi / 2.0,                          # vertical cut
        0.0,                                     # horizontal cut
        math.pi / 2.0 + math.radians(rng.uniform(-3, 3)),  # slight vertical jitter
        0.0 + math.radians(rng.uniform(-3, 3)),             # slight horizontal jitter
        _pca_axis(coords, rng, jitter_deg=2.0),  # PCA fallback
    ]
    centroid = cell.centroid
    cx, cy = centroid.x, centroid.y

    best: tuple[Polygon, Polygon] | None = None
    for angle in angle_candidates:
        for pos_frac in (0.5, 0.45, 0.55, 0.40, 0.60):
            # Offset origin along the normal to shift the split line
            nx = -math.sin(angle)
            ny = math.cos(angle)
            bounds_proj = []
            for x, y in coords:
                bounds_proj.append((x - cx) * nx + (y - cy) * ny)
            span = max(bounds_proj) - min(bounds_proj)
            offset = (pos_frac - 0.5) * span
            origin = (cx + nx * offset, cy + ny * offset)
            a, b = half_plane_split(cell, origin, angle)
            if _accept_split(cell, a, b, image_size):
                best = (a, b)  # type: ignore[assignment]
                break
        if best is not None:
            break

    if best is None:
        cells.append(cell)
        return

    a, b = best
    # Recurse, alternating so the tree stays reasonably balanced
    _subdivide_cell(a, depth + 1, max_depth, target_count, cells, rng, image_size)
    _subdivide_cell(b, depth + 1, max_depth, target_count, cells, rng, image_size)


def subdivide(footprint: Footprint, cfg: GeneratorConfig,
              rng: np.random.Generator) -> list[Room]:
    target = int(rng.integers(cfg.min_rooms, cfg.max_rooms + 1))
    cells: list[Polygon] = []
    _subdivide_cell(footprint.polygon, 0, 12, target, cells, rng, cfg.image_size)
    if not cells:
        cells = [footprint.polygon]

    # Assign room types
    used: dict[str, int] = {}
    rooms: list[Room] = []
    # Sort cells by area descending to give large cells higher-weight types
    order = sorted(range(len(cells)), key=lambda i: -cells[i].area)
    for rank, idx in enumerate(order):
        cell = cells[idx]
        rt = sample_room_type(rng, used)
        used[rt] = used.get(rt, 0) + 1
        rooms.append(Room(polygon=cell, room_type=rt, area_px=float(cell.area),
                          idx=idx, depth=rank))
    rooms.sort(key=lambda r: r.idx)
    return rooms
