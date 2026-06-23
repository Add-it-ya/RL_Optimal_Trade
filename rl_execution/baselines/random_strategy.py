"""Random execution.

Trades a uniformly random fraction of the remaining inventory each step (the environment
force-liquidates any remainder at the horizon).  Serves as a naive lower-bound benchmark
that any competent strategy should beat.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from rl_execution.baselines.base import BaseStrategy


class RandomStrategy(BaseStrategy):
    name = "Random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    def reset(self, env) -> None:
        super().reset(env)
        self.rng = np.random.default_rng(self.seed)

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        return float(self.rng.uniform(0.0, 1.0))
