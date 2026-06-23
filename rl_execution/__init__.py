"""RL-Optimal-Trade-Execution.

A production-quality Reinforcement Learning framework for optimal trade execution.

The package is organised into loosely-coupled sub-packages:

* :mod:`rl_execution.config`      -- typed configuration dataclasses.
* :mod:`rl_execution.envs`        -- LOB simulator, market process and the Gymnasium environment.
* :mod:`rl_execution.baselines`   -- classical execution strategies (TWAP, VWAP, POV, AC, Random).
* :mod:`rl_execution.agents`      -- model-free RL agents (DQN / DoubleDQN / PPO / A2C / SAC).
* :mod:`rl_execution.metrics`     -- execution quality metrics.
* :mod:`rl_execution.backtest`    -- a uniform backtesting engine for any policy.
* :mod:`rl_execution.experiments` -- market-regime presets and sweep runners.
* :mod:`rl_execution.viz`         -- plotting utilities.
* :mod:`rl_execution.data`        -- real / historical LOB data loaders.

The core simulation layers (everything except :mod:`rl_execution.agents`) depend only on
numpy/pandas/scipy/gymnasium so they can be used without a deep-learning backend installed.
"""

import os as _os

# Anaconda ships MKL's libiomp5 and PyTorch bundles its own OpenMP; on Windows both can be
# loaded into one process and abort with "OMP: Error #15". Allow duplicate runtimes (the
# standard, widely-used workaround) before torch is ever imported.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rl_execution.config import (
    ActionType,
    ExecutionConfig,
    MarketConfig,
    RewardConfig,
    Side,
)

__version__ = "0.1.0"

__all__ = [
    "MarketConfig",
    "ExecutionConfig",
    "RewardConfig",
    "Side",
    "ActionType",
    "__version__",
]
