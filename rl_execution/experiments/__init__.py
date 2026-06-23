"""Market-regime presets and experiment runners."""

from rl_execution.experiments.regimes import (
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
    "get_regime",
    "randomized_market_config",
    "list_regimes",
    "build_baselines",
    "evaluate_across_regimes",
    "regime_results_frame",
    "DomainRandomizedEnv",
]
