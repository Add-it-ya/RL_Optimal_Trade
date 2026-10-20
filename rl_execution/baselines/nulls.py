"""Null-hypothesis strategies: the degenerate policies an RL agent must be shown to beat.

* :class:`DumpAtStart` sells (or buys) the whole order at arrival: no price risk, maximum
  impact.
* :class:`WaitThenDump` does nothing and lets the final step liquidate everything: no
  impact management at all, and the full price risk of the horizon -- a directional bet.
* :class:`SignalRule` is the cheap way to use the order-book signal: the Almgren-Chriss
  pace, slowed down when the book says waiting pays and sped up when it does not, with one
  coefficient tuned on validation episodes.

An agent that cannot beat the first two on the objective has not learned execution; one
that cannot beat :class:`SignalRule` has learned nothing a two-line rule does not already
capture.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from rl_execution.baselines.almgren_chriss import AlmgrenChriss
from rl_execution.baselines.base import BaseStrategy

#: Index of the side-signed book imbalance in the environment's observation.
_IMBALANCE = 3


class DumpAtStart(BaseStrategy):
    name = "DumpAtStart"

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        return 1.0


class WaitThenDump(BaseStrategy):
    name = "WaitThenDump"

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        return 0.0  # the environment liquidates everything on the final step


class SignalRule(AlmgrenChriss):
    """Almgren-Chriss pace scaled by ``max(0, 1 - beta * imbalance)``.

    The imbalance feature is signed so that positive means waiting is favourable (buying
    pressure for a seller), so a positive ``beta`` slows down when the book says the price
    is about to move our way and speeds up when it says the opposite.  ``beta=None`` means
    "tune me": :meth:`fit` picks it from ``betas`` on validation episodes.
    """

    name = "SignalRule"
    BETAS: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)

    def __init__(self, beta: Optional[float] = None, risk_aversion: Optional[float] = None):
        super().__init__(risk_aversion=risk_aversion)
        self.beta = beta

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        t = self.step_index
        if t >= len(self.trades):
            return 1.0
        planned = float(self.holdings[t])
        pace = float(self.trades[t]) / planned if planned > 0 else 1.0
        tilt = max(0.0, 1.0 - float(self.beta or 0.0) * float(obs[_IMBALANCE]))
        return float(np.clip(pace * tilt, 0.0, 1.0))

    def fit(
        self,
        env_factory: Callable[[], Any],
        n_episodes: int,
        base_seed: int,
        betas: Optional[Sequence[float]] = None,
    ) -> None:
        """Pick ``beta`` minimising the mean objective on ``n_episodes`` validation episodes."""
        from rl_execution.backtest.engine import evaluate

        scores: Dict[float, float] = {}
        for beta in betas if betas is not None else self.BETAS:
            self.beta = float(beta)
            res = evaluate(env_factory, self, n_episodes=n_episodes, base_seed=base_seed)
            scores[self.beta] = res.aggregate["objective_bps_mean"]
        self.beta = min(scores, key=lambda b: scores[b])

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "beta": self.beta}
