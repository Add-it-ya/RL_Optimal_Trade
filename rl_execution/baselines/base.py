"""Base class and shared utilities for non-learning execution strategies.

All strategies (baselines *and* the wrappers around trained RL agents) implement the same
:class:`BaseStrategy` interface so the backtest engine can run any of them uniformly::

    strat.reset(env)
    action = strat.act(obs, info)   # returns an action in env.action_space

Baselines decide a *fraction of remaining inventory* to trade and convert it to the env's
action space; they may read the (already public) state of the bound environment, which is
appropriate for scheduling rules such as TWAP / VWAP / POV / Almgren-Chriss.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rl_execution.config import ActionType


class BaseStrategy:
    """Abstract execution strategy with a uniform ``reset`` / ``act`` interface."""

    name: str = "base"

    def reset(self, env) -> None:
        """Bind the strategy to an environment instance at the start of an episode."""
        self.env = env

    # -- convenient views onto the bound environment ----------------------------
    @property
    def total(self) -> float:
        return self.env.total_inventory

    @property
    def remaining(self) -> float:
        return self.env.remaining

    @property
    def horizon(self) -> int:
        return self.env.horizon

    @property
    def step_index(self) -> int:
        return self.env.t

    @property
    def steps_left(self) -> int:
        return max(self.env.horizon - self.env.t, 1)

    # -- action conversion ------------------------------------------------------
    def _fraction_to_action(self, fraction: float):
        """Convert a fraction in [0, 1] to a valid action for the bound env."""
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if self.env.exec_config.action_type is ActionType.DISCRETE:
            grid = self.env._action_grid
            return int(np.argmin(np.abs(grid - fraction)))
        return np.array([fraction], dtype=np.float32)

    def _decide_fraction(self, obs: np.ndarray, info: Dict[str, Any]) -> float:
        """Return the fraction of *remaining* inventory to execute this step."""
        raise NotImplementedError

    def act(self, obs: np.ndarray, info: Dict[str, Any]):
        return self._fraction_to_action(self._decide_fraction(obs, info))

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _shares_to_fraction(target_shares: float, remaining: float) -> float:
        if remaining <= 0:
            return 0.0
        return float(np.clip(target_shares / remaining, 0.0, 1.0))
