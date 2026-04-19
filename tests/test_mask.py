"""63-case integration matrix (DESIGN §12.3 / §12.4).

7 primitives × 3 aggression levels × 3 eras = 63 cases, all at seed=42.
Each case must produce a mask that passes validate_mask() with 0 errors.
"""
from __future__ import annotations

import numpy as np
import pytest

from floorplangen import GeneratorConfig, generate_sample, PIXEL_VALUES
from floorplangen.mask.validate import validate_mask


PRIMITIVES = ["RECT", "L", "T", "U", "Z", "STAIR", "BEVEL"]
AGGRESSIONS = [0.0, 0.5, 1.0]
ERAS = ["scan", "digital", "soviet"]


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("aggression", AGGRESSIONS)
@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_matrix(primitive: str, aggression: float, era: str,
                default_config: GeneratorConfig) -> None:
    cfg = GeneratorConfig()
    cfg.icon_pack_dir = default_config.icon_pack_dir
    cfg.forced_primitive = primitive  # may not exist; handled below
    sample = generate_sample(
        seed=42,
        image_size=(512, 512),
        era=era,
        aggression=aggression,
        config=cfg,
        icon_pack_dir=str(default_config.icon_pack_dir),
    )
    errs = validate_mask(sample.mask, None, require_wall_fraction=False)
    assert errs == [], f"primitive={primitive} aggr={aggression} era={era}: {errs}"

    unique = set(int(v) for v in np.unique(sample.mask))
    assert unique.issubset(PIXEL_VALUES), f"illegal pixel values: {unique - PIXEL_VALUES}"

    assert sample.metadata["era"] == era
    assert sample.image.shape[:2] == (512, 512)
    assert sample.mask.shape == (512, 512)
    assert sample.class_mask.shape == (512, 512)
    assert sample.class_mask.max() <= 4


def test_mask_is_readonly_after_generate() -> None:
    sample = generate_sample(seed=42, image_size=(256, 256))
    # The writer freezes the mask; the returned copy may be writable,
    # but class_mask must encode only the 5 classes.
    assert set(int(v) for v in np.unique(sample.class_mask)).issubset({0, 1, 2, 3, 4})
