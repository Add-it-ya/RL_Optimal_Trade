"""Market-regime presets and the cross-regime experiment runner."""

from rl_execution.experiments.regimes import (
    BROWNIAN_REGIMES,
    CLASSIC_REGIMES,
    REGIME_GROUPS,
    REGIMES,
    get_regime,
    list_regimes,
    randomized_market_config,
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
    "get_regime",
    "randomized_market_config",
    "list_regimes",
    "build_baselines",
    "evaluate_across_regimes",
    "regime_results_frame",
    "DomainRandomizedEnv",
]
