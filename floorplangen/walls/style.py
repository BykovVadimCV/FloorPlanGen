"""Wall draw-style mix (DESIGN §5.5)."""
from __future__ import annotations

import numpy as np

# Per-era style mixes (§5.5.1)
STYLE_MIX: dict[str, dict[str, float]] = {
    "digital": {"solid": 0.50, "hollow": 0.50, "hatch": 0.00},
    "scan":    {"solid": 0.45, "hollow": 0.50, "hatch": 0.05},
    "soviet":  {"solid": 0.25, "hollow": 0.50, "hatch": 0.25},
}


def sample_wall_style(rng: np.random.Generator, era: str) -> str:
    probs = STYLE_MIX.get(era, STYLE_MIX["scan"])
    names = list(probs.keys())
    p = np.array([probs[n] for n in names], dtype=float)
    p = p / p.sum()
    return str(rng.choice(names, p=p))
