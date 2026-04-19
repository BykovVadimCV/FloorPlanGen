"""Chapter 2 acceptance tests (DESIGN §12.4)."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from floorplangen.footprint import generate_footprint
from floorplangen.footprint.primitives import PRIMITIVE_WEIGHTS
from floorplangen.footprint.validator import validate as _validate_footprint


IMG = (512, 512)
PRIMITIVES = list(PRIMITIVE_WEIGHTS.keys())


@pytest.mark.parametrize("aggression", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("seed", [1, 7, 42, 123])
def test_footprint_is_valid_polygon(seed: int, aggression: float) -> None:
    rng = np.random.default_rng(seed)
    fp = generate_footprint(IMG, aggression, rng, margin=16)
    assert isinstance(fp.polygon, Polygon)
    assert fp.polygon.is_valid
    assert not fp.polygon.is_empty
    assert fp.polygon.area > 0


@pytest.mark.parametrize("aggression", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("seed", range(20))
def test_footprint_passes_validator(seed: int, aggression: float) -> None:
    rng = np.random.default_rng(seed)
    fp = generate_footprint(IMG, aggression, rng, margin=16)
    assert _validate_footprint(fp.polygon, image_size=IMG), (
        f"seed={seed} aggr={aggression} primitive={fp.primitive_id}"
    )


def test_footprint_within_canvas_bounds() -> None:
    rng = np.random.default_rng(42)
    fp = generate_footprint(IMG, 1.0, rng, margin=16)
    minx, miny, maxx, maxy = fp.polygon.bounds
    assert minx >= 0 and miny >= 0
    assert maxx <= IMG[1] and maxy <= IMG[0]


def test_footprint_no_holes() -> None:
    for seed in range(10):
        rng = np.random.default_rng(seed)
        fp = generate_footprint(IMG, 1.0, rng, margin=16)
        assert len(list(fp.polygon.interiors)) == 0
