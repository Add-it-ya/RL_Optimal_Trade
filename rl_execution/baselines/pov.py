"""Percentage-of-Volume (POV) execution.

Targets a fixed participation rate of the contemporaneous market volume each step, so the
schedule adapts to realised liquidity.  Any inventory still outstanding at the horizon is
force-liquidated by the environment.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rl_execution.baselines.base import BaseStrategy


class POV(BaseStrategy):
    name = "POV"

    def __init__(self, participation_rate: float = 0.20) -> None:
        self.participation_rate = float(participation_rate)

    def reset(self, env) -> None:
        super().reset(env)

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        market_volume = self.env.market.market_volume
        target_shares = self.participation_rate * market_volume
        return self._shares_to_fraction(target_shares, self.remaining)
