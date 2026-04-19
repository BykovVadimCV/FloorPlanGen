"""Polygon rasterisation via rasterio (DESIGN §11.3, §11.10)."""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

try:
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover - rasterio is a hard dep
    _HAS_RASTERIO = False


def _to_polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    # GeometryCollection etc.
    try:
        from shapely import get_parts
        return [p for p in get_parts(geom) if isinstance(p, Polygon) and not p.is_empty]
    except Exception:  # pragma: no cover
        return []


def fill_polygon(mask: np.ndarray, polygon, value: int) -> None:
    """Fill polygon interior with `value`, in place.

    Operates on a (H, W) uint8 mask. Uses rasterio scanline (all_touched=False)
    for platform-stable pixel-perfect output. Silently ignores empty polygons.
    """
    polys = _to_polygons(polygon)
    if not polys:
        return
    h, w = mask.shape
    if _HAS_RASTERIO:
        transform = from_bounds(0, 0, w, h, w, h)
        burned = rasterize(
            [(p, 1) for p in polys],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
        mask[burned != 0] = value
    else:  # pragma: no cover - emergency fallback using PIL
        from PIL import Image, ImageDraw
        img = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        for p in polys:
            xs, ys = p.exterior.xy
            draw.polygon(list(zip(xs, ys)), fill=1)
            for ring in p.interiors:
                xs, ys = ring.xy
                draw.polygon(list(zip(xs, ys)), fill=0)
        arr = np.asarray(img)
        mask[arr != 0] = value


def rasterize_polygon(polygon, image_size: tuple[int, int]) -> np.ndarray:
    """Return a (H, W) bool mask of the polygon interior."""
    h, w = image_size
    out = np.zeros((h, w), dtype=np.uint8)
    fill_polygon(out, polygon, 1)
    return out.astype(bool)


def wall_band_polygon(centreline, half_width: float) -> Polygon:
    """Filled band polygon (§11.4).

    cap_style=2 (flat), join_style=2 (mitre) — matches butt joins used in scan
    and digital eras. For hollow walls the band covers the entire stroke-gap-stroke
    region so that class-1 pixels include the interior white gap.
    """
    hw = max(float(half_width), 0.5)
    return centreline.buffer(hw, cap_style=2, join_style=2)
