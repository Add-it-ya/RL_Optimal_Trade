"""Experiment runners: evaluate strategies across market regimes.

A strategy may require a particular action space (discrete value-based agents vs continuous
policy agents).  Because every regime is replayed with paired seeds and the market path is
drawn before anyone acts, strategies on different action spaces face identical price paths
and are directly comparable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from rl_execution.backtest import BacktestResult, evaluate
from rl_execution.baselines import (
    TWAP,
    VWAP,
    AlmgrenChriss,
    DumpAtStart,
    SignalRule,
    WaitThenDump,
)
from rl_execution.config import ActionType, ExecutionConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import (
    get_regime,
    list_regimes,
    randomized_market_config,
)


class DomainRandomizedEnv(ExecutionEnv):
    """Execution env that resamples the market regime on every ``reset``.

    Training against this distribution of regimes yields a single agent that is robust
    across volatility / liquidity / trend conditions rather than overfit to one.
    """

    def __init__(self, exec_config: Optional[ExecutionConfig] = None, seed: Optional[int] = None):
        self._dr_rng = np.random.default_rng(seed)
        super().__init__(randomized_market_config(self._dr_rng), exec_config)

    def reset(self, *, seed=None, options=None):
        self.market_config = randomized_market_config(self._dr_rng)
        return super().reset(seed=seed, options=options)


#: First seed of the validation block, used to fit strategy parameters (e.g. the
#: SignalRule coefficient).  Disjoint from the test block that ``rlx evaluate`` reports on.
VALIDATION_BASE_SEED = 1_000_000


def build_baselines() -> Dict[str, Any]:
    """The standard baselines plus the null-hypothesis policies.

    Almgren-Chriss uses the reward's own risk aversion, so it is the optimum of the agents'
    objective whenever its model assumptions hold.
    """
    return {
        "TWAP": TWAP(),
        "VWAP": VWAP(),
        "AlmgrenChriss": AlmgrenChriss(),
        "DumpAtStart": DumpAtStart(),
        "WaitThenDump": WaitThenDump(),
        "SignalRule": SignalRule(),
    }


def evaluate_across_regimes(
    strategies: Dict[str, Any],
    exec_config: Optional[ExecutionConfig] = None,
    regimes: Optional[List[str]] = None,
    action_types: Optional[Dict[str, ActionType]] = None,
    n_episodes: int = 50,
    base_seed: int = 0,
    progress: bool = True,
    antithetic: bool = False,
    fit_episodes: int = 200,
    fit_base_seed: int = VALIDATION_BASE_SEED,
) -> Dict[str, Dict[str, BacktestResult]]:
    """Evaluate each strategy in each regime on the same seeds (common random numbers).

    Strategies with free parameters are first fitted per regime on ``fit_episodes``
    validation episodes starting at ``fit_base_seed`` (see ``BaseStrategy.fit``); their
    parameters are kept on ``BacktestResult.params``.  ``antithetic`` evaluates in
    ``Z`` / ``-Z`` pairs (see :func:`evaluate`); downstream statistics must then use
    ``block=2``.  Returns ``{regime: {strategy_name: BacktestResult}}``.
    """
    exec_config = exec_config or ExecutionConfig()
    regimes = regimes or list_regimes()
    action_types = action_types or {}

    results: Dict[str, Dict[str, BacktestResult]] = {}
    for regime in regimes:
        market_config = get_regime(regime)
        results[regime] = {}
        for name, strat in strategies.items():
            at = action_types.get(name, ActionType.CONTINUOUS)
            ec = replace(exec_config, action_type=at)

            def factory(mc=market_config, ec=ec):
                return ExecutionEnv(mc, ec)

            if hasattr(strat, "fit"):
                strat.fit(factory, n_episodes=fit_episodes, base_seed=fit_base_seed)
            results[regime][name] = evaluate(
                factory,
                strat,
                n_episodes=n_episodes,
                base_seed=base_seed,
                name=name,
                progress=progress,
                antithetic=antithetic,
            )
            if hasattr(strat, "params"):
                results[regime][name].params = dict(strat.params())
    return results


def regime_results_frame(results: Dict[str, Dict[str, BacktestResult]]) -> pd.DataFrame:
    """Flatten nested regime results into a tidy DataFrame."""
    rows = []
    for regime, by_strat in results.items():
        for name, res in by_strat.items():
            row = res.summary_row()
            row["regime"] = regime
            rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["regime", "strategy"] + [c for c in df.columns if c not in ("regime", "strategy")]
    return df[cols].sort_values(["regime", "Objective_bps"]).reset_index(drop=True)
