"""Icon placement (DESIGN §7.6, §7.7, §7.9)."""
from __future__ import annotations

import math

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

from ..config import GeneratorConfig
from ..types import Footprint, PlacedIcon, Room, WallGraph
from .loader import IconAsset, IconPack, get_cached_pack
from .mapping import ROOM_ICON_CATEGORIES


def _rotate_rgba(rgba: np.ndarray, angle_deg: int) -> np.ndarray:
    if angle_deg % 360 == 0:
        return rgba
    k = (angle_deg % 360) // 90
    return np.rot90(rgba, k=k).copy()


def _icon_footprint_polygon(icon: IconAsset, top_left: tuple[int, int],
                             rgba_shape: tuple[int, int]) -> Polygon:
    h, w = rgba_shape
    fp = icon.sidecar.get("footprint")
    if not fp:
        # Use the bounding box of the rgba
        x0, y0 = top_left
        return box(x0, y0, x0 + w, y0 + h)
    # Normalised polygon → pixel polygon
    x0, y0 = top_left
    coords = [(x0 + px * w, y0 + py * h) for px, py in fp]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if isinstance(poly, Polygon) else box(x0, y0, x0 + w, y0 + h)


def place_icons(footprint: Footprint, rooms: list[Room], walls: WallGraph,
                cfg: GeneratorConfig, rng: np.random.Generator,
                era: str) -> list[PlacedIcon]:
    try:
        pack = get_cached_pack(cfg.icon_pack_dir)
    except FileNotFoundError:
        return []

    h_img, w_img = cfg.image_size
    canvas_area = float(h_img * w_img)
    placed: list[PlacedIcon] = []

    for room in rooms:
        cats = ROOM_ICON_CATEGORIES.get(room.room_type, [])
        room_frac = room.area_px / canvas_area
        # Per-room icon budget (§7.9)
        n_icons = max(0, min(3, int(rng.integers(1, 4))))
        if not cats:
            continue

        # Simple placement: pick a spot inside the room interior, random rotation
        minx, miny, maxx, maxy = room.polygon.bounds
        for _ in range(n_icons):
            cat = str(rng.choice(cats))
            eligible = pack.eligible(cat, era, min_area_frac=room_frac)
            if not eligible:
                continue
            asset = eligible[int(rng.integers(0, len(eligible)))]
            angle = int(rng.choice(asset.allow_rotation or [0]))
            rgba = _rotate_rgba(asset.rgba, angle)
            ih, iw = rgba.shape[:2]
            # Shrink to fit the room if needed
            room_w = maxx - minx
            room_h = maxy - miny
            max_dim = min(room_w, room_h) * 0.40
            scale = 1.0
            if max(iw, ih) > max_dim and max_dim > 4:
                scale = max_dim / max(iw, ih)
            if scale < 1.0:
                new_w = max(4, int(iw * scale))
                new_h = max(4, int(ih * scale))
                # Simple area resize via nearest (avoid importing PIL here)
                try:
                    import cv2
                    rgba = cv2.resize(rgba, (new_w, new_h),
                                      interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
                ih, iw = rgba.shape[:2]
            # Random position inside the room, offset from walls
            pad = 4
            max_x = int(maxx - iw - pad)
            min_x = int(minx + pad)
            max_y = int(maxy - ih - pad)
            min_y = int(miny + pad)
            if max_x <= min_x or max_y <= min_y:
                continue
            for _attempt in range(6):
                x = int(rng.integers(min_x, max_x))
                y = int(rng.integers(min_y, max_y))
                fp_poly = _icon_footprint_polygon(asset, (x, y), (ih, iw))
                if not room.polygon.contains(fp_poly.buffer(-1)):
                    # Check looser — the buffer may be too strict
                    if room.polygon.intersection(fp_poly).area / max(fp_poly.area, 1.0) < 0.85:
                        continue
                placed.append(PlacedIcon(
                    category=cat, stem=asset.stem, rgba=rgba,
                    top_left=(x, y), footprint_polygon=fp_poly,
                    rotation_deg=angle,
                ))
                break

    return placed


def composite_icons(*args, **kwargs):
    """Backwards-compat alias — actual compositing lives in rendering.icons."""
    raise NotImplementedError("Use floorplangen.rendering.render_icons")
