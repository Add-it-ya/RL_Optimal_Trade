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
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rl_execution.baselines.base import BaseStrategy


class AlmgrenChriss(BaseStrategy):
    name = "AlmgrenChriss"

    def __init__(self, risk_aversion: float = 1.0e-7) -> None:
        self.risk_aversion = float(risk_aversion)

    def reset(self, env) -> None:
        super().reset(env)
        self.trades = self._optimal_trades()

    def _optimal_trades(self) -> np.ndarray:
        cfg = self.env.market_config
        N = self.horizon
        tau = 1.0
        X = self.total
        eta = cfg.temporary_impact
        gamma = cfg.permanent_impact
        lam = self.risk_aversion
        sigma = cfg.volatility * cfg.initial_price  # absolute per-step price volatility

        eta_tilde = eta - 0.5 * gamma * tau
        if lam <= 0 or eta_tilde <= 0:
            return np.full(N, X / N)

        kappa_tilde_sq = lam * sigma ** 2 / eta_tilde
        arg = 0.5 * tau ** 2 * kappa_tilde_sq + 1.0
        kappa = np.arccosh(arg) / tau
        if not np.isfinite(kappa) or kappa < 1e-8:
            return np.full(N, X / N)

        T = N * tau
        j = np.arange(N + 1)
        holdings = X * np.sinh(kappa * (T - j * tau)) / np.sinh(kappa * T)
        trades = holdings[:-1] - holdings[1:]
        return np.maximum(trades, 0.0)

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        t = self.step_index
        target = self.trades[t] if t < len(self.trades) else self.remaining
        return self._shares_to_fraction(target, self.remaining)
