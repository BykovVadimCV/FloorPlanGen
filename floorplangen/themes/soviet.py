"""Soviet-era theme (Guidelines §1).

Aged калька / diazo cream paper, sepia or violet-blue ink, hand-drawn strokes
with jitter and overshoots. Stroke imperfections are driven here via the
theme's `stroke_model='hand'` + `overshoot_px` fields; LineworkSpec governs
the geometric distributions (door angle, window count, ...).
"""
from __future__ import annotations

import numpy as np

from .base import EraTheme
from .registry import register_theme


class SovietThemePlugin:
    era_id = "soviet"

    def build_theme(self, rng: np.random.Generator,
                    image_size: tuple[int, int]) -> EraTheme:
        scale = min(image_size) / 512.0
        # Aged калька / diazo cream paper — §1.2 palette
        paper_r = int(rng.integers(226, 246))
        paper_g = paper_r - int(rng.integers(6, 16))
        paper_b = paper_g - int(rng.integers(10, 26))
        # Ink: sepia-brown or violet-blue (diazo) or faded black
        ink_variant = int(rng.integers(0, 3))
        if ink_variant == 0:
            ink_rgb = (55, 40, 25)   # sepia-brown
        elif ink_variant == 1:
            ink_rgb = (40, 55, 140)  # violet-blue diazo
        else:
            v = int(rng.integers(30, 65))
            ink_rgb = (v, v, v)      # faded black
        return EraTheme(
            era_id=self.era_id,
            paper_rgb=(paper_r, paper_g, paper_b),
            ink_rgb=ink_rgb,
            accent_rgb=(195, 32, 26),  # красный карандаш
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
