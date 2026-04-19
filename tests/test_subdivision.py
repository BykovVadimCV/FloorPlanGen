"""Chapter 3 acceptance tests."""
from __future__ import annotations

import numpy as np
import pytest

from floorplangen import GeneratorConfig
from floorplangen.footprint import generate_footprint
from floorplangen.rng import bundle_from_seed
from floorplangen.subdivision import subdivide


IMG = (512, 512)


@pytest.mark.parametrize("seed", [1, 7, 42, 100])
@pytest.mark.parametrize("aggression", [0.0, 0.5, 1.0])
def test_rooms_tile_footprint(seed: int, aggression: float) -> None:
    rng = bundle_from_seed(seed)
    fp = generate_footprint(IMG, aggression, rng["footprint"], margin=16)
    cfg = GeneratorConfig()
    rooms = subdivide(fp, cfg, rng["subdivision"])

    assert len(rooms) >= 1
    # Coverage ≥85% of the footprint
    room_area = sum(r.polygon.area for r in rooms)
    assert room_area / fp.polygon.area >= 0.85, (
        f"seed={seed} aggr={aggression} coverage={room_area / fp.polygon.area:.3f}"
    )


@pytest.mark.parametrize("seed", [1, 42, 100])
def test_rooms_do_not_overlap(seed: int) -> None:
    rng = bundle_from_seed(seed)
    fp = generate_footprint(IMG, 0.5, rng["footprint"], margin=16)
    cfg = GeneratorConfig()
    rooms = subdivide(fp, cfg, rng["subdivision"])

    total_overlap = 0.0
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            total_overlap += rooms[i].polygon.intersection(rooms[j].polygon).area
    # Allow tiny Shapely precision overlap (<1px² total)
    assert total_overlap < 1.0, f"overlap={total_overlap:.3f}"


def test_room_types_populated() -> None:
    rng = bundle_from_seed(42)
    fp = generate_footprint(IMG, 0.5, rng["footprint"], margin=16)
    cfg = GeneratorConfig()
    rooms = subdivide(fp, cfg, rng["subdivision"])
    assert all(r.room_type for r in rooms)
    assert all(r.area_px > 0 for r in rooms)
