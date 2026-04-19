"""Statistical distribution test (DESIGN §12.3) — lightweight 60-sample batch."""
from __future__ import annotations

import numpy as np
import pytest

from floorplangen import generate_sample


@pytest.mark.slow
def test_era_mix(default_config) -> None:
    counts = {"scan": 0, "digital": 0, "soviet": 0}
    N = 60
    for s in range(N):
        r = generate_sample(seed=s, image_size=(256, 256),
                            icon_pack_dir=str(default_config.icon_pack_dir))
        counts[str(r.metadata["era"])] += 1
    # Expected: scan 0.60, digital 0.25, soviet 0.15. With N=60 we use wide bands.
    assert counts["scan"] / N > 0.40
    assert counts["digital"] / N > 0.10
    assert counts["soviet"] / N > 0.03


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
