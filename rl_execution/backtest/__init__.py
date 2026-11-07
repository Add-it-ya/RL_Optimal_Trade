"""Uniform backtesting engine for baselines and trained RL agents."""

from rl_execution.backtest.engine import (
    BacktestResult,
    compare_strategies,
    corrected_significance,
    episode_is,
    evaluate,
    normalised_schedule,
    paired_is_table,
    pool_results,
    results_table,
    run_episode,
    schedule_gap,
    schedule_gap_table,
)

__all__ = [
    "BacktestResult",
    "run_episode",
    "evaluate",
    "compare_strategies",
    "results_table",
    "paired_is_table",
    "pool_results",
    "corrected_significance",
    "episode_is",
    "normalised_schedule",
    "schedule_gap",
    "schedule_gap_table",
]
