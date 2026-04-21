"""Generator configuration (DESIGN §1.3.1, §1.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Era mix default per Guidelines §5.9 (30% soviet / 40% transitional / 30% modern).
ERA_MIX_DEFAULT: dict[str, float] = {
    "soviet": 0.30,
    "transitional": 0.40,
    "modern": 0.30,
}
ERA_TO_PRESET: dict[str, str] = {
    "modern": "clean",
    "transitional": "medium",
    "soviet": "heavy",
}


@dataclass
class GeneratorConfig:
    # Image
    image_size: tuple[int, int] = (512, 512)
    monochrome_prob: float = 0.70
    # Sampling
    era: Literal["soviet", "transitional", "modern"] | None = None
    era_mix: dict[str, float] = field(default_factory=lambda: dict(ERA_MIX_DEFAULT))
    aggression: float | None = None  # None => sampled Uniform(0, 1)
    forced_primitive: str | None = None  # debug/testing: force a specific primitive id
    # Augmentation
    augmentation_preset: Literal["clean", "medium", "heavy"] | None = None
    # Assets
    icon_pack_dir: Path = Path(__file__).resolve().parent.parent / "icons"
    fonts_dir: Path = Path(__file__).resolve().parent.parent / "FONTS"
    # Text labels
    text_language: str = "ru"   # "ru" or "en"
    text_prob: float = 0.70     # probability of rendering room labels
    # Layout
    min_rooms: int = 3
    max_rooms: int = 7
    # Canvas padding
    canvas_margin: int = 24

    def resolved_era(self, rng) -> str:
        if self.era is not None:
            return self.era
        names = list(self.era_mix.keys())
        weights = [self.era_mix[n] for n in names]
        total = sum(weights)
        weights = [w / total for w in weights]
        return rng.choice(names, p=weights)

    def resolved_preset(self, era: str) -> str:
        if self.augmentation_preset is not None:
            return self.augmentation_preset
        return ERA_TO_PRESET[era]

    def resolved_aggression(self, rng) -> float:
        if self.aggression is not None:
            return float(self.aggression)
        return float(rng.uniform(0.0, 1.0))
