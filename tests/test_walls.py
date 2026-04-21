"""Chapter 4+5 acceptance tests."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from floorplangen import GeneratorConfig
from floorplangen.footprint import generate_footprint
from floorplangen.rng import bundle_from_seed
from floorplangen.subdivision import subdivide
from floorplangen.walls import build_wall_graph


IMG = (512, 512)


def _build(seed: int, aggression: float = 0.5, era: str = "transitional"):
    rng = bundle_from_seed(seed)
    fp = generate_footprint(IMG, aggression, rng["footprint"], margin=16)
    cfg = GeneratorConfig()
    rooms = subdivide(fp, cfg, rng["subdivision"])
    walls = build_wall_graph(fp, rooms, cfg, rng["walls"], rng["diagonal"], era=era)
    return fp, rooms, walls


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_wall_graph_has_exterior_segments(seed: int) -> None:
    _, _, walls = _build(seed)
    assert len(walls.segments) > 0
    exterior = [s for s in walls.segments if s.is_exterior]
    assert len(exterior) >= 3, "need at least 3 exterior edges (triangle minimum)"


@pytest.mark.parametrize("seed", [1, 42])
def test_wall_bands_are_valid_polygons(seed: int) -> None:
    _, _, walls = _build(seed)
    for seg in walls.segments:
        band = seg.ensure_band()
        assert isinstance(band, Polygon)
        assert band.is_valid
        assert band.area > 0


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_all_walls_within_expanded_footprint(seed: int) -> None:
    fp, _, walls = _build(seed)
    # Guidelines §5.1 allows capital walls up to ~6 mm (1:100) which at
    # 512×512 is ~38 px; use a 25 px buffer to cover half-thickness + slack.
    padded = fp.polygon.buffer(25.0)
    for seg in walls.segments:
        band = seg.ensure_band()
        assert padded.contains(band.buffer(-0.1)) or padded.intersection(band).area / band.area > 0.90
