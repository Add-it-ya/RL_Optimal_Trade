"""Market-regime presets and the cross-regime experiment runner."""

from rl_execution.experiments.regimes import (
    BROWNIAN_REGIMES,
    CLASSIC_REGIMES,
    HELD_OUT,
    IN_DISTRIBUTION,
    REGIME_GROUPS,
    REGIMES,
    SIGNAL_ALPHAS,
    SIGNAL_REGIMES,
    SIGNAL_TARGET_IC,
    SPLITS,
    get_regime,
    list_regimes,
    outside_randomized_support,
    randomized_market_config,
    resolve_regimes,
)
from rl_execution.experiments.runner import (
    DomainRandomizedEnv,
    build_baselines,
    evaluate_across_regimes,
    regime_results_frame,
)

__all__ = [
    "REGIMES",
    "REGIME_GROUPS",
    "CLASSIC_REGIMES",
    "BROWNIAN_REGIMES",
    "IN_DISTRIBUTION",
    "HELD_OUT",
    "SPLITS",
    "SIGNAL_REGIMES",
    "SIGNAL_ALPHAS",
    "SIGNAL_TARGET_IC",
    "outside_randomized_support",
    "resolve_regimes",
    "get_regime",
    "randomized_market_config",
    "list_regimes",
    "build_baselines",
    "evaluate_across_regimes",
    "regime_results_frame",
    "DomainRandomizedEnv",
]
