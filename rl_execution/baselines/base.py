"""Base class and shared utilities for non-learning execution strategies.

All strategies (baselines *and* the wrappers around trained RL agents) implement the same
:class:`BaseStrategy` interface so the backtest engine can run any of them uniformly::

    strat.reset(env)
    action = strat.act(obs, info)   # returns an action in env.action_space

Baselines decide a *fraction of remaining inventory* to trade and convert it with
:meth:`~rl_execution.envs.ExecutionEnv.fraction_to_action`, so they go through exactly the
same action interface as the agents.  They may read the (already public) state of the bound
environment, which is appropriate for scheduling rules such as TWAP / VWAP / Almgren-Chriss.

A strategy with free parameters can implement :meth:`BaseStrategy.fit`, which the
cross-regime runner calls on validation seeds before evaluating it on test seeds.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


class BaseStrategy:
    """Abstract execution strategy with a uniform ``reset`` / ``act`` interface."""

    name: str = "base"

    def reset(self, env) -> None:
        """Bind the strategy to an environment instance at the start of an episode."""
        self.env = env

    def fit(self, env_factory: Callable[[], Any], n_episodes: int, base_seed: int) -> None:
        """Tune free parameters on validation episodes (no-op for parameter-free rules)."""

    def params(self) -> Dict[str, Any]:
        """Fitted / configured parameters worth recording next to the results."""
        return {}

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
        """Convert a fraction of the remaining inventory to an action for the bound env."""
        return self.env.fraction_to_action(float(np.clip(fraction, 0.0, 1.0)))

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
