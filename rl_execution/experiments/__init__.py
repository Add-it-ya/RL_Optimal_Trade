"""Market-regime presets and experiment runners."""

from rl_execution.experiments.regimes import (
    REGIMES,
    REGIME_GROUPS,
    get_regime,
    randomized_market_config,
    list_regimes,
)
from rl_execution.experiments.runner import (
    build_baselines,
    evaluate_across_regimes,
    regime_results_frame,
    DomainRandomizedEnv,
)

__all__ = [
    "REGIMES",
    "REGIME_GROUPS",
    "get_regime",
    "randomized_market_config",
    "list_regimes",
    "build_baselines",
    "evaluate_across_regimes",
    "regime_results_frame",
    "DomainRandomizedEnv",
]
