"""Statistical distribution test (DESIGN §12.3) — lightweight 60-sample batch."""
from __future__ import annotations

import numpy as np
import pytest

from floorplangen import generate_sample


@pytest.mark.slow
def test_era_mix(default_config) -> None:
    counts = {"soviet": 0, "transitional": 0, "modern": 0}
    N = 60
    for s in range(N):
        r = generate_sample(seed=s, image_size=(256, 256),
                            icon_pack_dir=str(default_config.icon_pack_dir))
        counts[str(r.metadata["era"])] += 1
    # Expected: soviet 0.30, transitional 0.40, modern 0.30. Wide bands for N=60.
    assert counts["transitional"] / N > 0.20
    assert counts["soviet"] / N > 0.10
    assert counts["modern"] / N > 0.10


@pytest.mark.slow
def test_monochrome_fraction(default_config) -> None:
    mono = 0
    N = 60
    for s in range(N):
        r = generate_sample(seed=s, image_size=(256, 256),
                            icon_pack_dir=str(default_config.icon_pack_dir))
        if r.metadata["monochrome"]:
            mono += 1
    # target 0.70 with wide band
    assert mono / N > 0.50
