"""Volume-Weighted Average Price (VWAP) execution.

Allocates the parent order across the horizon in proportion to a *forecast* intraday
volume profile, so that more is traded when the market is expected to be liquid.  Here the
forecast is the (deterministic) U-shaped profile used by the market simulator; with real
data it would be an estimated historical profile.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rl_execution.baselines.base import BaseStrategy


class VWAP(BaseStrategy):
    name = "VWAP"

    def reset(self, env) -> None:
        super().reset(env)
        cfg = env.market_config
        h = self.horizon
        frac = np.arange(h) / max(h - 1, 1)
        profile = 1.0 + cfg.volume_u_shape * (2.0 * (frac - 0.5)) ** 2
        self.profile = profile / profile.sum()
        # planned cumulative shares to have executed by the *end* of each step
        self.plan = self.total * self.profile

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        t = min(self.step_index, self.horizon - 1)
        target_shares = self.plan[t]
        return self._shares_to_fraction(target_shares, self.remaining)
