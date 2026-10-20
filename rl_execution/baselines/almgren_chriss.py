"""Almgren-Chriss optimal execution.

Closed-form risk-averse optimal schedule under linear permanent + temporary market impact
and arithmetic price volatility (Almgren & Chriss, 2000, *Optimal execution of portfolio
transactions*).  The optimal holdings trajectory is

.. math::

    x_j = X \\frac{\\sinh\\big(\\kappa (T - t_j)\\big)}{\\sinh(\\kappa T)}, \\qquad
    \\kappa = \\frac{1}{\\tau}\\,\\operatorname{arccosh}\\!\\Big(\\tfrac{1}{2}\\tau^2\\tilde\\kappa^2 + 1\\Big),
    \\quad \\tilde\\kappa^2 = \\frac{\\lambda\\sigma^2}{\\tilde\\eta},
    \\quad \\tilde\\eta = \\eta - \\tfrac{1}{2}\\gamma\\tau .

As risk aversion :math:`\\lambda \\to 0` the schedule collapses to TWAP; larger
:math:`\\lambda` front-loads trading to reduce exposure to price risk.

By default the baseline uses the risk aversion of the environment's reward (converted by
:func:`~rl_execution.config.ac_risk_aversion`), so it optimises exactly the objective the
agents are rewarded on; pass ``risk_aversion`` explicitly to trace the efficient frontier.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from rl_execution.baselines.base import BaseStrategy
from rl_execution.config import MarketConfig, ac_risk_aversion


def absolute_volatility(market: MarketConfig, reference_price: Optional[float] = None) -> float:
    """Per-step price volatility in cash units, matched to the simulator's process.

    The closed form is derived for *arithmetic* Brownian motion, so ``sigma`` must be an
    absolute price volatility.  Under ``price_process='abm'`` the simulator uses exactly
    ``volatility * initial_price``, which makes the schedule genuinely optimal there.
    Under a geometric process the corresponding local volatility is ``volatility * S``,
    evaluated at ``reference_price`` (the arrival price) -- a first-order match.
    """
    sigma_rel = float(market.volatility) * np.sqrt(float(market.dt))
    if str(market.price_process) == "gbm" and reference_price:
        return sigma_rel * float(reference_price)
    return sigma_rel * float(market.initial_price)


def optimal_holdings(
    market: MarketConfig,
    horizon: int,
    total: float,
    risk_aversion: float,
    reference_price: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Almgren-Chriss holdings ``x_0 = total, ..., x_N = 0`` and the urgency ``kappa``.

    ``risk_aversion`` is the cash-unit ``lambda``.  ``kappa * horizon * dt`` measures how
    front-loaded the schedule is: 0 is TWAP, and it grows with risk aversion and volatility
    and falls with temporary impact.
    """
    n, tau, x = int(horizon), float(market.dt), float(total)
    eta_tilde = float(market.temporary_impact) - 0.5 * float(market.permanent_impact) * tau
    sigma = absolute_volatility(market, reference_price)
    twap = x * (1.0 - np.arange(n + 1) / n)
    if risk_aversion <= 0 or eta_tilde <= 0:
        return twap, 0.0

    kappa_tilde_sq = risk_aversion * sigma**2 / eta_tilde
    kappa = float(np.arccosh(0.5 * tau**2 * kappa_tilde_sq + 1.0) / tau)
    if not np.isfinite(kappa) or kappa < 1e-8:
        return twap, 0.0
    j = np.arange(n + 1)
    return x * np.sinh(kappa * (n - j) * tau) / np.sinh(kappa * n * tau), kappa


class AlmgrenChriss(BaseStrategy):
    name = "AlmgrenChriss"

    def __init__(self, risk_aversion: Optional[float] = None) -> None:
        # None -> match the reward's risk aversion at reset
        self.risk_aversion = None if risk_aversion is None else float(risk_aversion)

    def reset(self, env) -> None:
        super().reset(env)
        lam = self.risk_aversion
        if lam is None:
            lam = ac_risk_aversion(env.market_config, env.exec_config)
        self.holdings, self.kappa = optimal_holdings(
            env.market_config, self.horizon, self.total, lam, reference_price=env.arrival_price
        )
        self.trades = np.maximum(self.holdings[:-1] - self.holdings[1:], 0.0)

    def params(self) -> Dict[str, Any]:
        return {"risk_aversion": "matched" if self.risk_aversion is None else self.risk_aversion}

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        t = self.step_index
        target = float(self.trades[t]) if t < len(self.trades) else self.remaining
        return self._shares_to_fraction(target, self.remaining)
