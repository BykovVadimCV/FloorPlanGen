"""EraTheme dataclass (DESIGN §9.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class EraTheme:
    era_id: str
    # Palette
    paper_rgb: tuple[int, int, int] = (255, 255, 255)
    ink_rgb: tuple[int, int, int] = (30, 30, 30)
    accent_rgb: tuple[int, int, int] = (200, 60, 60)
    # Stroke
    stroke_model: Literal["solid", "hand"] = "solid"
    line_weight_range: tuple[float, float] = (1.0, 2.0)   # px at image scale
    overshoot_px: float = 0.0
    # Fills
    colour_mode: Literal["mono", "tint", "colour"] = "mono"
    room_tint_rgb: tuple[int, int, int] | None = None
    # Per-room fill palette (None = no room fills; non-empty = cycle through colours)
    room_palette: list[tuple[int, int, int]] = field(default_factory=list)
    wall_colour_rgb: tuple[int, int, int] | None = None  # None = use ink_rgb
    # Annotation
    annotation_style: Literal["arrow_filled", "tick_oblique", "tick"] = "arrow_filled"
    # Compatibility markers
    hatch_spacing_px: float = 5.0
    hollow_gap_px: float = 3.0


class EraThemePlugin(Protocol):
    era_id: str
    def build_theme(self, rng, image_size: tuple[int, int]) -> EraTheme: ...
