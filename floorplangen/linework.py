"""Per-era linework specification.

Encodes the visual-language rules from Guidelines.md §1 (Soviet), §2
(Transitional) and §3 (Modern), with §5.1/§5.2 as master tables.

The three eras model a single drifting convention, so most parameters are
simple probability distributions over categorical choices or ranges over
continuous values. A single `LineworkSpec` instance is built once per sample
and threaded through wall / opening / icon placement so the eras stay
internally consistent.

Coordinate note: all "paper millimetres" in the guidelines assume a 1:100
architectural scale. We approximate the paper→image-pixel mapping as
``px_per_mm = min(image_size) / 80.0`` — a 512 px render therefore assumes a
~80 mm wide slab of paper, which comfortably fits the 60×75 mm plan of a
45 m² flat with margin for the dimension chains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


EraId = Literal["soviet", "transitional", "modern"]

# ── Paper ↔ pixel mapping (shared by every era) ─────────────────────────────
PAPER_MM_FOR_MIN_SIDE = 80.0  # nominal A4 apartment slab


def px_per_mm(image_size: tuple[int, int]) -> float:
    return min(image_size) / PAPER_MM_FOR_MIN_SIDE


# ── Master linework tables (Guidelines §5.1 / §5.2) ─────────────────────────
@dataclass(frozen=True)
class WallWeights:
    """Wall-band widths in paper millimetres at 1:100.

    Real masonry walls (400–640 mm brick) and partitions (50–120 mm) compress
    to these paper values. Per §0.3 the exterior/interior split is *not* a
    reliable structural marker: pick one per wall independently.
    """
    capital_mm: tuple[float, float]  # thick band (capital / несущая)
    partition_mm: tuple[float, float]  # thin band (перегородка)
    capital_share: float  # probability a given wall is drawn "capital-thick"
    outline_px: float  # ink weight of the band outline itself


@dataclass(frozen=True)
class DoorSpec:
    """Door leaf drawing style."""
    angle_deg_choices: tuple[float, ...]  # e.g. (30.0, 90.0)
    angle_deg_weights: tuple[float, ...]
    arc_weight_px: float  # swing-arc line weight


@dataclass(frozen=True)
class WindowSpec:
    line_count_choices: tuple[int, ...]  # 2 / 3 / 4
    line_count_weights: tuple[float, ...]
    frame_weight_px: float


@dataclass(frozen=True)
class VentShaftSpec:
    """Vent-shaft hatching style — a per-era tell (§1.11 / §3.7 / §5.9)."""
    style: Literal["cross_hatch", "diagonal_45", "solid"]


@dataclass(frozen=True)
class StrokeSpec:
    """Hand-drawn stroke imperfection budget."""
    thickness_jitter: float  # ±fraction of nominal
    wobble_amp_mm: float     # lateral wobble amplitude on paper
    overshoot_mm: float      # endpoint extension past junction
    overshoot_prob: float    # probability of overshoot at an endpoint


@dataclass(frozen=True)
class BalconySpec:
    """Balcony drawing style inside the slab polygon."""
    fill: Literal["hatch_45", "empty"]


@dataclass(frozen=True)
class FurnitureSpec:
    """Which fixture categories are allowed in this era."""
    allow_radiators: bool
    allow_free_furniture: bool  # bed/sofa/etc. — absent on BTI plans


@dataclass(frozen=True)
class AnnotationSpec:
    tick_style: Literal["tick_oblique", "tick", "arrow_filled"]


@dataclass(frozen=True)
class LineworkSpec:
    era: EraId
    wall: WallWeights
    wall_style_mix: dict[str, float]  # {solid, hollow, hatch}
    door: DoorSpec
    window: WindowSpec
    vent: VentShaftSpec
    stroke: StrokeSpec
    balcony: BalconySpec
    furniture: FurnitureSpec
    annotation: AnnotationSpec
    hatch_spacing_mm: float = 2.0

    # ── Convenience samplers ────────────────────────────────────────────────
    def sample_wall_thickness_mm(self, rng: np.random.Generator,
                                 hint_exterior: bool) -> float:
        """Return a wall thickness in paper mm, randomised per §0.3.

        `hint_exterior` nudges (does not force) the sample toward the capital
        band — real plans tend to have most capital walls on the perimeter,
        but interior load-bearing walls exist and exterior non-capital walls
        exist on balconies/extensions.
        """
        base = self.wall.capital_share
        p_capital = min(0.95, base + 0.20) if hint_exterior else max(0.05, base - 0.20)
        if rng.random() < p_capital:
            lo, hi = self.wall.capital_mm
        else:
            lo, hi = self.wall.partition_mm
        return float(rng.uniform(lo, hi))

    def sample_wall_style(self, rng: np.random.Generator) -> str:
        names = list(self.wall_style_mix.keys())
        probs = np.array([self.wall_style_mix[n] for n in names], dtype=float)
        probs = probs / probs.sum()
        return str(rng.choice(names, p=probs))

    def sample_door_angle_deg(self, rng: np.random.Generator) -> float:
        probs = np.array(self.door.angle_deg_weights, dtype=float)
        probs = probs / probs.sum()
        return float(rng.choice(self.door.angle_deg_choices, p=probs))

    def sample_window_line_count(self, rng: np.random.Generator) -> int:
        probs = np.array(self.window.line_count_weights, dtype=float)
        probs = probs / probs.sum()
        return int(rng.choice(self.window.line_count_choices, p=probs))


# ── Per-era presets ─────────────────────────────────────────────────────────
def _soviet_spec() -> LineworkSpec:
    # §1.4, §1.6, §5.1, §5.2
    return LineworkSpec(
        era="soviet",
        wall=WallWeights(
            capital_mm=(1.2, 1.8),
            partition_mm=(0.5, 0.9),
            capital_share=0.40,
            outline_px=0.25,
        ),
        wall_style_mix={"solid": 0.10, "hollow": 0.65, "hatch": 0.25},
        door=DoorSpec(
            angle_deg_choices=(30.0, 90.0),
            angle_deg_weights=(0.40, 0.60),
            arc_weight_px=0.25,
        ),
        window=WindowSpec(
            line_count_choices=(2, 3),
            line_count_weights=(0.35, 0.65),
            frame_weight_px=0.25,
        ),
        vent=VentShaftSpec(style="cross_hatch"),
        stroke=StrokeSpec(
            thickness_jitter=0.15,
            wobble_amp_mm=0.25,
            overshoot_mm=0.7,
            overshoot_prob=0.50,
        ),
        balcony=BalconySpec(fill="hatch_45"),
        furniture=FurnitureSpec(
            allow_radiators=False,
            allow_free_furniture=False,
        ),
        annotation=AnnotationSpec(tick_style="tick_oblique"),
        hatch_spacing_mm=2.0,
    )


def _transitional_spec() -> LineworkSpec:
    # §2.5, §2.7, §5.1
    return LineworkSpec(
        era="transitional",
        wall=WallWeights(
            capital_mm=(1.0, 1.6),
            partition_mm=(0.4, 0.8),
            capital_share=0.50,
            outline_px=0.20,
        ),
        wall_style_mix={"solid": 0.40, "hollow": 0.55, "hatch": 0.05},
        door=DoorSpec(
            angle_deg_choices=(30.0, 90.0),
            angle_deg_weights=(0.30, 0.70),
            arc_weight_px=0.20,
        ),
        window=WindowSpec(
            line_count_choices=(3, 4),
            line_count_weights=(0.55, 0.45),
            frame_weight_px=0.20,
        ),
        vent=VentShaftSpec(style="diagonal_45"),
        stroke=StrokeSpec(
            thickness_jitter=0.05,
            wobble_amp_mm=0.05,
            overshoot_mm=0.2,
            overshoot_prob=0.20,
        ),
        balcony=BalconySpec(fill="empty"),
        furniture=FurnitureSpec(
            allow_radiators=True,
            allow_free_furniture=False,
        ),
        annotation=AnnotationSpec(tick_style="tick"),
        hatch_spacing_mm=1.5,
    )


def _modern_spec() -> LineworkSpec:
    # §3.4, §3.6, §5.1
    return LineworkSpec(
        era="modern",
        wall=WallWeights(
            capital_mm=(0.8, 1.4),
            partition_mm=(0.35, 0.7),
            capital_share=0.55,
            outline_px=0.20,
        ),
        wall_style_mix={"solid": 0.50, "hollow": 0.50, "hatch": 0.00},
        door=DoorSpec(
            angle_deg_choices=(30.0, 90.0),
            angle_deg_weights=(0.50, 0.50),
            arc_weight_px=0.20,
        ),
        window=WindowSpec(
            line_count_choices=(3, 4),
            line_count_weights=(0.25, 0.75),
            frame_weight_px=0.20,
        ),
        vent=VentShaftSpec(style="solid"),
        stroke=StrokeSpec(
            thickness_jitter=0.0,
            wobble_amp_mm=0.0,
            overshoot_mm=0.0,
            overshoot_prob=0.0,
        ),
        balcony=BalconySpec(fill="empty"),
        furniture=FurnitureSpec(
            allow_radiators=False,
            allow_free_furniture=False,
        ),
        annotation=AnnotationSpec(tick_style="tick"),
        hatch_spacing_mm=1.0,
    )


_SPECS: dict[str, LineworkSpec] = {
    "soviet": _soviet_spec(),
    "transitional": _transitional_spec(),
    "modern": _modern_spec(),
}


def get_linework(era: str) -> LineworkSpec:
    try:
        return _SPECS[era]
    except KeyError as ex:
        raise KeyError(f"Unknown era {era!r}; expected one of {sorted(_SPECS)}") from ex
