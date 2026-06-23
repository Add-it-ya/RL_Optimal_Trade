"""Time-Weighted Average Price (TWAP) execution.

Trades an equal number of shares (``total_inventory / horizon``) in every window,
independent of price or volume.  The canonical, impact-minimising-under-no-information
benchmark.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rl_execution.baselines.base import BaseStrategy


class TWAP(BaseStrategy):
    name = "TWAP"

    def reset(self, env) -> None:
        super().reset(env)
        self.per_step = self.total / self.horizon

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        return self._shares_to_fraction(self.per_step, self.remaining)
