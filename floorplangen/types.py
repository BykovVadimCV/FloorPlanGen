"""Core dataclasses and constants (DESIGN §1.2, §1.3, §2.7, §3.9, §4.8, §5.10, §7.12)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from shapely.geometry import LineString, Polygon

# ── Class pixel values (§1.2) ────────────────────────────────────────────────
CLASS_BACKGROUND = 0
CLASS_WALL = 1
CLASS_WINDOW = 2
CLASS_DOOR = 3
CLASS_FURNITURE = 4

PIXEL_BACKGROUND = 0
PIXEL_WALL = 64
PIXEL_WINDOW = 128
PIXEL_DOOR = 192
PIXEL_FURNITURE = 255

CLASS_PIXEL_MAP: dict[int, int] = {
    PIXEL_BACKGROUND: CLASS_BACKGROUND,
    PIXEL_WALL: CLASS_WALL,
    PIXEL_WINDOW: CLASS_WINDOW,
    PIXEL_DOOR: CLASS_DOOR,
    PIXEL_FURNITURE: CLASS_FURNITURE,
}
PIXEL_VALUES: set[int] = set(CLASS_PIXEL_MAP.keys())

EraId = Literal["scan", "digital", "soviet"]
AugPreset = Literal["clean", "medium", "heavy"]


# ── Chapter 2 output (§2.7) ─────────────────────────────────────────────────
@dataclass
class Footprint:
    polygon: Polygon
    primitive_id: str
    n_cuts_applied: int = 0
    n_extrusions_applied: int = 0
    bevel_corners: list[int] = field(default_factory=list)
    aggression: float = 0.0
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    primitive_fallback: str | None = None


# ── Chapter 3 output (§3.9) ─────────────────────────────────────────────────
@dataclass
class Room:
    polygon: Polygon
    room_type: str
    area_px: float
    depth: int = 0
    idx: int = 0


# ── Chapter 4+5 output (§5.10) ──────────────────────────────────────────────
@dataclass
class WallSegment:
    centreline: LineString
    thickness_px: float
    style: Literal["solid", "hollow", "hatch"] = "solid"
    is_exterior: bool = False
    is_diagonal: bool = False
    angle_deg: float = 0.0
    no_annotate: bool = False
    band_polygon: Polygon | None = None  # filled band for mask + hollow-wall rule

    def ensure_band(self) -> Polygon:
        if self.band_polygon is None:
            half = max(self.thickness_px / 2.0, 0.5)
            self.band_polygon = self.centreline.buffer(half, cap_style=2, join_style=2)
        return self.band_polygon


@dataclass
class WallGraph:
    segments: list[WallSegment] = field(default_factory=list)

    def band_polygons(self) -> list[Polygon]:
        return [s.ensure_band() for s in self.segments]


# ── Chapter 6 output (§6.11) ────────────────────────────────────────────────
@dataclass
class Opening:
    kind: Literal["window", "door"]
    polygon: Polygon           # carves through the wall band; axis-aligned
    wall_index: int = -1       # index into WallGraph.segments
    swing_arc: Polygon | None = None  # doors only; part of class-3 footprint
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def class_polygon(self) -> Polygon:
        if self.kind == "door" and self.swing_arc is not None:
            from shapely.ops import unary_union
            return unary_union([self.polygon, self.swing_arc])
        return self.polygon


# ── Chapter 7 output (§7.12) ────────────────────────────────────────────────
@dataclass
class PlacedIcon:
    category: str
    stem: str
    rgba: np.ndarray                # (h, w, 4) uint8, pre-multiplied
    top_left: tuple[int, int]       # (x, y) on canvas
    footprint_polygon: Polygon      # for mask write (class 4)
    rotation_deg: int = 0


# ── Chapter 8 output (§8) ───────────────────────────────────────────────────
@dataclass
class Annotation:
    polygons: list[Polygon]    # arrow shafts, arrowheads, label bboxes
    text: str
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


# ── Top-level output (§1.3.2) ───────────────────────────────────────────────
@dataclass
class GeneratorOutput:
    image: np.ndarray
    mask: np.ndarray
    class_mask: np.ndarray
    yolo_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
