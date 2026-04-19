"""Era 3 — Soviet hand-drawn theme (§9.5)."""
from __future__ import annotations

import numpy as np

from .base import EraTheme
from .registry import register_theme


class SovietThemePlugin:
    era_id = "soviet"

    def build_theme(self, rng: np.random.Generator,
                    image_size: tuple[int, int]) -> EraTheme:
        scale = min(image_size) / 512.0
        # Yellowed paper with variation
        paper_r = int(rng.integers(226, 246))
        paper_g = paper_r - int(rng.integers(6, 16))
        paper_b = paper_g - int(rng.integers(10, 26))
        ink_v = int(rng.integers(30, 65))
        return EraTheme(
            era_id=self.era_id,
            paper_rgb=(paper_r, paper_g, paper_b),
            ink_rgb=(ink_v, ink_v, ink_v),
            accent_rgb=(140, 70, 40),
            stroke_model="hand",
            line_weight_range=(2.0 * scale, 3.4 * scale),
            overshoot_px=2.0 * scale,
            colour_mode="mono",
            room_tint_rgb=None,
            annotation_style="tick_oblique",
            hatch_spacing_px=6.0 * scale,
            hollow_gap_px=3.5 * scale,
        )


register_theme(SovietThemePlugin())
