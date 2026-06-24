"""Market-regime presets and experiment runners."""

# Statistical-rigor harnesses (Step 2). Imported after runner/regimes so their internal
# `from rl_execution.experiments.runner import ...` resolves against the loaded submodules.
from rl_execution.experiments.hpo import make_study, optimize, run_study, search_space
from rl_execution.experiments.multiseed import MultiSeedResult, run_multiseed
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
    "run_multiseed",
    "MultiSeedResult",
    "make_study",
    "run_study",
    "search_space",
    "optimize",
]
