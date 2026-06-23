"""Uniform backtesting engine for baselines and trained RL agents."""

from rl_execution.backtest.engine import (
    BacktestResult,
    compare_strategies,
    evaluate,
    paired_is_table,
    results_table,
    run_episode,
)

__all__ = [
    "BacktestResult",
    "run_episode",
    "evaluate",
    "compare_strategies",
    "results_table",
    "paired_is_table",
]
