"""Primitive footprint polygons (DESIGN §2.2).

All primitives are parameterised in a normalised 1x1 unit square. They are scaled
to the target canvas by the caller.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from shapely.geometry import Polygon

PrimitiveId = str

# Probability weights (§2.2.2) at aggression=1.0
PRIMITIVE_WEIGHTS: dict[PrimitiveId, float] = {
    "RECT":  0.05,
    "L":     0.28,
    "T":     0.18,
    "U":     0.17,
    "Z":     0.10,
    "STAIR": 0.12,
    "BEVEL": 0.10,
}


def _rect() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def _l_shape(rng: np.random.Generator) -> Polygon:
    ax = float(rng.uniform(0.30, 0.60))  # arm thickness x (remaining)
    ay = float(rng.uniform(0.30, 0.60))  # arm thickness y
    # Remove upper-right quadrant
    return Polygon([
        (0, 0), (1, 0), (1, 1 - ay),
        (ax, 1 - ay), (ax, 1), (0, 1),
    ])


def _t_shape(rng: np.random.Generator) -> Polygon:
    stem_w = float(rng.uniform(0.30, 0.55))
    stem_h = float(rng.uniform(0.40, 0.65))
    x0 = (1 - stem_w) / 2
    x1 = x0 + stem_w
    # Cross-bar on top (height 1-stem_h); stem hangs down
    return Polygon([
        (x0, 0),                  (x1, 0),
        (x1, stem_h),             (1, stem_h),
        (1, 1),                   (0, 1),
        (0, stem_h),              (x0, stem_h),
    ])


def _u_shape(rng: np.random.Generator) -> Polygon:
    arm_w = float(rng.uniform(0.22, 0.38))
    gap_w = 1 - 2 * arm_w
    notch_d = float(rng.uniform(0.40, 0.70))
    # Courtyard cut from top
    return Polygon([
        (0, 0), (1, 0), (1, 1),
        (1 - arm_w, 1), (1 - arm_w, 1 - notch_d),
        (arm_w,     1 - notch_d), (arm_w,     1),
        (0, 1),
    ])


def _z_shape(rng: np.random.Generator) -> Polygon:
    step = float(rng.uniform(0.30, 0.50))
    thick = float(rng.uniform(0.35, 0.55))
    # Z-profile: remove top-right and bottom-left rectangles
    # Keep the full polygon as an 8-gon for validator simplicity
    return Polygon([
        (step, 0),           (1, 0),
        (1, thick),          (step + (1 - step), thick),  # noop vertex
        (1, 1),              (0, 1),
        (0, 1 - thick),      (step, 1 - thick),
    ])


def _stair_shape(rng: np.random.Generator) -> Polygon:
    n_steps = int(rng.integers(2, 4))  # 2 or 3
    step_w = 1.0 / n_steps
    step_h = float(rng.uniform(0.12, 0.22))
    coords: list[tuple[float, float]] = [(0, 0), (1, 0)]
    # Step down from right
    for i in range(n_steps):
        x_right = 1.0 - i * step_w
        x_left = 1.0 - (i + 1) * step_w
        y_bot = 1.0 - (i + 1) * step_h
        y_top = 1.0 - i * step_h
        # Right-face rises from y_top-? no — we walk the top edge going left.
        # Build corner run: go up to (x_right, y_top), then left to (x_left, y_top)
        coords.append((x_right, y_top))
        coords.append((x_left, y_top))
    coords.append((0, coords[-1][1]))  # close the left side
    return Polygon(coords)


def _bevel_shape(rng: np.random.Generator) -> Polygon:
    b = float(rng.uniform(0.08, 0.18))  # bevel fraction
    return Polygon([
        (b, 0), (1 - b, 0),
        (1, b), (1, 1 - b),
        (1 - b, 1), (b, 1),
        (0, 1 - b), (0, b),
    ])


_BUILDERS: dict[PrimitiveId, Callable[[np.random.Generator], Polygon]] = {
    "RECT":  lambda _r: _rect(),
    "L":     _l_shape,
    "T":     _t_shape,
    "U":     _u_shape,
    "Z":     _z_shape,
    "STAIR": _stair_shape,
    "BEVEL": _bevel_shape,
}


def select_primitive(aggression: float, rng: np.random.Generator) -> PrimitiveId:
    """§2.2.2: at aggression=0 only RECT; blend toward full weights at 1.0."""
    if aggression <= 0.0:
        return "RECT"
    rect_w = max(0.0, 1.0 - aggression) * 0.90  # dominant at low aggression
    remaining = 1.0 - rect_w
    weights = {k: v for k, v in PRIMITIVE_WEIGHTS.items() if k != "RECT"}
    total = sum(weights.values())
    pool = [("RECT", rect_w)] + [(k, remaining * v / total) for k, v in weights.items()]
    names = [n for n, _ in pool]
    probs = np.array([w for _, w in pool], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def build_primitive(primitive_id: PrimitiveId,
                    rng: np.random.Generator) -> Polygon:
    return _BUILDERS[primitive_id](rng)


def scale_to_canvas(unit_poly: Polygon,
                    canvas_size: tuple[int, int],
                    aspect_ratio: float,
                    margin: int) -> Polygon:
    """Scale unit polygon to fill canvas (minus margin) at the given AR."""
    from shapely.affinity import scale as _shp_scale, translate as _shp_translate

    h, w = canvas_size
    avail_w = w - 2 * margin
    avail_h = h - 2 * margin
    # Derive bounding-box size matching aspect ratio
    if aspect_ratio >= 1.0:
        box_w = avail_w
        box_h = avail_w / aspect_ratio
        if box_h > avail_h:
            box_h = avail_h
            box_w = avail_h * aspect_ratio
    else:
        box_h = avail_h
        box_w = avail_h * aspect_ratio
        if box_w > avail_w:
            box_w = avail_w
            box_h = avail_w / aspect_ratio
    scaled = _shp_scale(unit_poly, xfact=box_w, yfact=box_h, origin=(0, 0))
    tx = margin + (avail_w - box_w) / 2.0
    ty = margin + (avail_h - box_h) / 2.0
    return _shp_translate(scaled, xoff=tx, yoff=ty)
