"""Seed plumbing — one numpy Generator per sample, stable sub-streams per chapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


CHAPTER_STREAMS = (
    "footprint", "subdivision", "walls", "diagonal",
    "openings", "icons", "annotations", "theme", "augment",
)


@dataclass
class RNGBundle:
    """Per-chapter np.random.Generator instances derived from a single root seed.

    Derivation uses SeedSequence.spawn so that inserting a new chapter between
    existing ones does not shift the bit stream of unrelated chapters.
    """
    root: np.random.Generator
    streams: dict[str, np.random.Generator]

    def __getitem__(self, key: str) -> np.random.Generator:
        return self.streams[key]


def bundle_from_seed(seed: int) -> RNGBundle:
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(CHAPTER_STREAMS))
    streams = {
        name: np.random.default_rng(child)
        for name, child in zip(CHAPTER_STREAMS, children)
    }
    return RNGBundle(root=np.random.default_rng(ss), streams=streams)
