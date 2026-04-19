"""Era 2 — Modern Digital theme (§9.4)."""
from __future__ import annotations

import numpy as np

from .base import EraTheme
from .registry import register_theme

# Pastel room palettes used when colour_mode=="colour"
_ROOM_PALETTES = [
    [(255, 230, 220), (220, 240, 255), (220, 255, 225), (255, 255, 210),
     (240, 220, 255), (210, 240, 240), (255, 240, 210)],
    [(255, 200, 180), (180, 220, 255), (190, 255, 200), (255, 255, 180),
     (230, 200, 255), (180, 240, 230), (255, 230, 190)],
]


class DigitalThemePlugin:
    era_id = "digital"

    def build_theme(self, rng: np.random.Generator,
                    image_size: tuple[int, int]) -> EraTheme:
        scale = min(image_size) / 512.0
        roll = rng.random()
        if roll < 0.20:
            # Full colour — room fills with pastel palette
            palette = _ROOM_PALETTES[int(rng.integers(0, len(_ROOM_PALETTES)))]
            wall_r = int(rng.integers(20, 60))
            wall_g = int(rng.integers(20, 60))
            wall_b = int(rng.integers(100, 180))
            return EraTheme(
                era_id=self.era_id,
                paper_rgb=(252, 252, 252),
                ink_rgb=(wall_r, wall_g, wall_b),
                accent_rgb=(40, 120, 200),
                stroke_model="solid",
                line_weight_range=(1.0 * scale, 1.8 * scale),
                overshoot_px=0.0,
                colour_mode="colour",
                room_palette=palette,
                wall_colour_rgb=(wall_r, wall_g, wall_b),
                annotation_style="arrow_filled",
                hatch_spacing_px=4.0 * scale,
                hollow_gap_px=2.5 * scale,
            )
        elif roll < 0.55:
            # Warm tint
            return EraTheme(
                era_id=self.era_id,
                paper_rgb=(255, 255, 255),
                ink_rgb=(10, 10, 10),
                accent_rgb=(40, 120, 200),
                stroke_model="solid",
                line_weight_range=(1.0 * scale, 1.6 * scale),
                overshoot_px=0.0,
                colour_mode="tint",
                room_tint_rgb=(245, 242, 232),
                annotation_style="arrow_filled",
                hatch_spacing_px=4.0 * scale,
                hollow_gap_px=2.5 * scale,
            )
        else:
            return EraTheme(
                era_id=self.era_id,
                paper_rgb=(255, 255, 255),
                ink_rgb=(10, 10, 10),
                accent_rgb=(40, 120, 200),
                stroke_model="solid",
                line_weight_range=(1.0 * scale, 1.6 * scale),
                overshoot_px=0.0,
                colour_mode="mono",
                annotation_style="arrow_filled",
                hatch_spacing_px=4.0 * scale,
                hollow_gap_px=2.5 * scale,
            )


register_theme(DigitalThemePlugin())
