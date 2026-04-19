"""Era 1 — Post-2000 Scan theme (§9.3)."""
from __future__ import annotations

import numpy as np

from .base import EraTheme
from .registry import register_theme


class ScanThemePlugin:
    era_id = "scan"

    def build_theme(self, rng: np.random.Generator,
                    image_size: tuple[int, int]) -> EraTheme:
        scale = min(image_size) / 512.0
        paper_v = int(rng.integers(246, 255))
        ink_v = int(rng.integers(20, 55))
        lw_lo = 1.2 * scale
        lw_hi = 2.3 * scale

        # ~15% of scans are from colour-printed originals
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
                paper_rgb=(paper_v, paper_v, paper_v),
                ink_rgb=(wall_r, wall_g, wall_b),
                accent_rgb=(190, 70, 70),
                stroke_model="solid",
                line_weight_range=(lw_lo, lw_hi),
                overshoot_px=0.0,
                colour_mode="colour",
                room_palette=palette,
                wall_colour_rgb=(wall_r, wall_g, wall_b),
                annotation_style="arrow_filled",
                hatch_spacing_px=5.0 * scale,
                hollow_gap_px=3.0 * scale,
            )

        return EraTheme(
            era_id=self.era_id,
            paper_rgb=(paper_v, paper_v, paper_v),
            ink_rgb=(ink_v, ink_v, ink_v),
            accent_rgb=(190, 70, 70),
            stroke_model="solid",
            line_weight_range=(lw_lo, lw_hi),
            overshoot_px=0.0,
            colour_mode="mono",
            annotation_style="arrow_filled",
            hatch_spacing_px=5.0 * scale,
            hollow_gap_px=3.0 * scale,
        )


register_theme(ScanThemePlugin())
