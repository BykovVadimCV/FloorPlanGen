"""Chapter 6 acceptance tests."""
from __future__ import annotations

import pytest

from floorplangen import GeneratorConfig
from floorplangen.footprint import generate_footprint
from floorplangen.openings import place_openings
from floorplangen.rng import bundle_from_seed
from floorplangen.subdivision import subdivide
from floorplangen.walls import build_wall_graph


IMG = (512, 512)


def _pipeline(seed: int):
    rng = bundle_from_seed(seed)
    fp = generate_footprint(IMG, 0.5, rng["footprint"], margin=16)
    cfg = GeneratorConfig()
    rooms = subdivide(fp, cfg, rng["subdivision"])
    walls = build_wall_graph(fp, rooms, cfg, rng["walls"], rng["diagonal"], era="scan")
    openings = place_openings(fp, rooms, walls, cfg, rng["openings"], era="scan")
    return fp, walls, openings


@pytest.mark.parametrize("seed", [1, 7, 42, 100])
def test_no_two_openings_on_same_wall(seed: int) -> None:
    """Placement must not stack two openings on a single wall segment.
    (Cross-wall overlap at T-junctions is expected; the mask priority stack
    resolves class ambiguity deterministically.)"""
    _, _, openings = _pipeline(seed)
    per_wall: dict[int, int] = {}
    for o in openings:
        if o.wall_index < 0:
            continue
        per_wall[o.wall_index] = per_wall.get(o.wall_index, 0) + 1
    offenders = {k: v for k, v in per_wall.items() if v > 1}
    assert not offenders, f"seed={seed}: multiple openings on walls {offenders}"


@pytest.mark.parametrize("seed", [1, 42])
def test_windows_on_exterior_only(seed: int) -> None:
    _, walls, openings = _pipeline(seed)
    for o in openings:
        if o.kind != "window":
            continue
        if o.wall_index < 0:
            continue
        seg = walls.segments[o.wall_index]
        assert seg.is_exterior, (
            f"window placed on interior wall_index={o.wall_index}"
        )


@pytest.mark.parametrize("seed", [1, 42])
def test_doors_have_swing_arc(seed: int) -> None:
    _, _, openings = _pipeline(seed)
    doors = [o for o in openings if o.kind == "door"]
    if not doors:
        pytest.skip("no doors placed for this seed")
    assert any(d.swing_arc is not None for d in doors)
