"""Mask invariance under augmentation (DESIGN §12.4 Ch 10).

Same seed × {clean, medium, heavy} must produce bit-identical masks.
"""
from __future__ import annotations

import numpy as np
import pytest

from floorplangen import generate_sample


@pytest.mark.parametrize("seed", [1, 7, 42, 123, 999])
def test_mask_identical_across_presets(seed: int, default_config) -> None:
    kw = dict(seed=seed, image_size=(256, 256),
              icon_pack_dir=str(default_config.icon_pack_dir))
    a = generate_sample(augmentation_preset="clean", **kw)
    b = generate_sample(augmentation_preset="medium", **kw)
    c = generate_sample(augmentation_preset="heavy", **kw)

    assert np.array_equal(a.mask, b.mask), f"clean vs medium mask differ for seed={seed}"
    assert np.array_equal(b.mask, c.mask), f"medium vs heavy mask differ for seed={seed}"
    assert np.array_equal(a.class_mask, c.class_mask)


@pytest.mark.parametrize("seed", [1, 42, 999])
def test_image_differs_across_presets(seed: int, default_config) -> None:
    kw = dict(seed=seed, image_size=(256, 256),
              icon_pack_dir=str(default_config.icon_pack_dir))
    a = generate_sample(augmentation_preset="clean", **kw)
    c = generate_sample(augmentation_preset="heavy", **kw)
    # Clean and heavy must produce visibly different images
    assert not np.array_equal(a.image, c.image), (
        f"seed={seed}: heavy augmentation produced identical image to clean"
    )


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_determinism_same_seed(seed: int, default_config) -> None:
    kw = dict(seed=seed, image_size=(256, 256),
              icon_pack_dir=str(default_config.icon_pack_dir))
    a = generate_sample(**kw)
    b = generate_sample(**kw)
    assert np.array_equal(a.mask, b.mask)
    assert np.array_equal(a.image, b.image)
