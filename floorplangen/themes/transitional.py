"""Transitional-era theme (Guidelines §2).

Mix of hand-drawn retrace, typewritten overlays and early CAD output, all run
through a photocopier/scanner pipeline. Paper palette drifts cream -> tan.
"""
from __future__ import annotations

import numpy as np

from .base import EraTheme
from .registry import register_theme


class TransitionalThemePlugin:
    era_id = "transitional"

    def build_theme(self, rng: np.random.Generator,
                    image_size: tuple[int, int]) -> EraTheme:
        scale = min(image_size) / 512.0
        # Paper: cream -> lightly yellowed per Guidelines §2.4
        paper_r = int(rng.integers(235, 255))
        paper_g = paper_r - int(rng.integers(2, 10))
        paper_b = paper_g - int(rng.integers(4, 14))
        ink_v = int(rng.integers(20, 55))
        lw_lo = 1.2 * scale
        lw_hi = 2.3 * scale

        use_colour = rng.random() < 0.15
        if use_colour:
            wall_r = int(rng.integers(10, 50))
            wall_g = int(rng.integers(10, 50))
            wall_b = int(rng.integers(10, 50))
            palette = [
                (int(rng.integers(220, 250)), int(rng.integers(220, 250)),
                 int(rng.integers(200, 240)))
                for _ in range(7)
            ]
            return EraTheme(
                era_id=self.era_id,
                paper_rgb=(paper_r, paper_g, paper_b),
                ink_rgb=(wall_r, wall_g, wall_b),
                accent_rgb=(200, 40, 45),
                stroke_model="solid",
                line_weight_range=(lw_lo, lw_hi),
                overshoot_px=0.0,
                colour_mode="colour",
                room_palette=palette,
                wall_colour_rgb=(wall_r, wall_g, wall_b),
                annotation_style="tick",
                hatch_spacing_px=5.0 * scale,
                hollow_gap_px=3.0 * scale,
            )

        return EraTheme(
            era_id=self.era_id,
            paper_rgb=(paper_r, paper_g, paper_b),
            ink_rgb=(ink_v, ink_v, ink_v),
            accent_rgb=(200, 40, 45),
            stroke_model="solid",
            line_weight_range=(lw_lo, lw_hi),
            overshoot_px=0.0,
            colour_mode="mono",
            annotation_style="tick",
            hatch_spacing_px=5.0 * scale,
            hollow_gap_px=3.0 * scale,
        )


register_theme(TransitionalThemePlugin())
