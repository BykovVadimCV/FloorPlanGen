"""OCR-readable text rendering — room labels and area annotations.

Scans FONTS_DIR for .ttf/.otf files, picks one per image, then renders:
  - Room type label  (e.g. "Кухня" / "Kitchen")
  - Room area        (e.g. "12.4 м²")

Text is rendered with PIL (supports Unicode/Cyrillic) and composited onto
the OpenCV canvas.  Falls back to PIL's built-in bitmap font if no .ttf
is found — no crash, just lower fidelity.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from ..themes.base import EraTheme
from ..types import Room


# Russian / English room-type labels
_LABELS_RU: dict[str, str] = {
    "bedroom": "Спальня",
    "living":  "Гостиная",
    "kitchen": "Кухня",
    "bathroom": "Санузел",
    "hall":    "Прихожая",
    "storage": "Кладовая",
    "balcony": "Балкон",
    "stair":   "Лестница",
}
_LABELS_EN: dict[str, str] = {
    "bedroom": "Bedroom",
    "living":  "Living",
    "kitchen": "Kitchen",
    "bathroom": "Bathroom",
    "hall":    "Hall",
    "storage": "Storage",
    "balcony": "Balcony",
    "stair":   "Stair",
}


def _discover_fonts(fonts_dir: Path) -> list[Path]:
    if not fonts_dir.exists():
        return []
    return sorted(fonts_dir.rglob("*.ttf")) + sorted(fonts_dir.rglob("*.otf"))


def _load_font(fonts_dir: Path, size_px: int,
               rng: np.random.Generator) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    if not _HAS_PIL:
        return None  # type: ignore[return-value]
    fonts = _discover_fonts(fonts_dir)
    if fonts:
        path = fonts[int(rng.integers(0, len(fonts)))]
        try:
            return ImageFont.truetype(str(path), size=size_px)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        return ImageFont.load_default()


def _ink_tuple(theme: EraTheme, alpha: int = 200) -> tuple[int, int, int, int]:
    r, g, b = theme.ink_rgb
    return (r, g, b, alpha)


def render_room_labels(canvas: np.ndarray, rooms: list[Room],
                       theme: EraTheme, rng: np.random.Generator,
                       fonts_dir: Path,
                       language: str = "ru",
                       px_per_sqm: float = 1.0) -> None:
    """Render room type + area text centred in each room."""
    if not _HAS_PIL:
        return

    h, w = canvas.shape[:2]
    scale = min(h, w) / 512.0
    base_size = max(8, int(12 * scale))
    font = _load_font(fonts_dir, base_size, rng)
    if font is None:
        return

    labels = _LABELS_RU if language == "ru" else _LABELS_EN

    # Work on a PIL RGBA overlay
    is_gray = (canvas.ndim == 2)
    if is_gray:
        pil_canvas = PILImage.fromarray(canvas, mode="L").convert("RGBA")
    else:
        pil_canvas = PILImage.fromarray(
            cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        ).convert("RGBA")

    overlay = PILImage.new("RGBA", pil_canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ink = _ink_tuple(theme, alpha=200)

    for room in rooms:
        label = labels.get(room.room_type, room.room_type.capitalize())
        area_m2 = room.area_px * px_per_sqm
        area_str = f"{area_m2:.1f}" if area_m2 >= 1.0 else ""
        if language == "ru" and area_str:
            area_str += " м²"
        elif area_str:
            area_str += " m²"

        # Room bounding box — skip very small rooms
        minx, miny, maxx, maxy = room.polygon.bounds
        room_w = maxx - minx
        room_h = maxy - miny
        if room_w < base_size * 3 or room_h < base_size * 2:
            continue

        # Fit font to room width
        try:
            bb = draw.textbbox((0, 0), label, font=font)
            txt_w = bb[2] - bb[0]
            if txt_w > room_w * 0.85:
                fit_size = max(6, int(base_size * room_w * 0.85 / max(txt_w, 1)))
                font_fit = _load_font_size(font, fit_size)
            else:
                font_fit = font
        except Exception:
            font_fit = font

        # Centroid for placement
        cx = int(room.polygon.centroid.x)
        cy = int(room.polygon.centroid.y)

        try:
            bb1 = draw.textbbox((0, 0), label, font=font_fit)
            tw1 = bb1[2] - bb1[0]
            th1 = bb1[3] - bb1[1]
            x1 = cx - tw1 // 2
            y1 = cy - th1 - 2
            draw.text((x1, y1), label, font=font_fit, fill=ink)

            if area_str:
                bb2 = draw.textbbox((0, 0), area_str, font=font_fit)
                tw2 = bb2[2] - bb2[0]
                x2 = cx - tw2 // 2
                y2 = cy + 2
                draw.text((x2, y2), area_str, font=font_fit, fill=ink)
        except Exception:
            pass

    # Composite overlay onto canvas
    merged = PILImage.alpha_composite(pil_canvas, overlay).convert("RGB")
    result = cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)
    if is_gray:
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        canvas[:] = result_gray
    else:
        canvas[:] = result


def _load_font_size(existing_font, new_size: int):
    """Try to reload the same font at a different size."""
    if not _HAS_PIL:
        return existing_font
    try:
        path = existing_font.path  # FreeTypeFont has .path
        return ImageFont.truetype(path, size=new_size)
    except Exception:
        try:
            return ImageFont.load_default(size=new_size)
        except TypeError:
            return ImageFont.load_default()
