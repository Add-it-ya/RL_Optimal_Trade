"""RL-Optimal-Trade-Execution.

A Reinforcement Learning framework for optimal trade execution, organised into
loosely-coupled sub-packages:

* :mod:`rl_execution.config`      -- typed configuration dataclasses.
* :mod:`rl_execution.envs`        -- Brownian market process, LOB simulator and the Gymnasium env.
* :mod:`rl_execution.baselines`   -- classical execution strategies (TWAP, VWAP, Almgren-Chriss).
* :mod:`rl_execution.agents`      -- RL agents (PPO via Stable-Baselines3; DQN / Double DQN).
* :mod:`rl_execution.metrics`     -- execution-quality metrics and statistics.
* :mod:`rl_execution.backtest`    -- a uniform backtesting engine for any policy.
* :mod:`rl_execution.experiments` -- market-regime presets and the cross-regime runner.
* :mod:`rl_execution.training`    -- training and run-folder persistence.
* :mod:`rl_execution.viz`         -- plotting utilities.

The simulation layers (everything except :mod:`rl_execution.agents`) depend only on
numpy/pandas/scipy/gymnasium so they can be used without a deep-learning backend installed.
"""

import os as _os

if _os.name == "nt":
    # Anaconda ships MKL's libiomp5 and PyTorch bundles its own OpenMP; on Windows both can
    # be loaded into one process and abort with "OMP: Error #15". Allow duplicate runtimes
    # (the standard, widely-used workaround) before torch is ever imported.
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rl_execution.config import (
    ActionType,
    ExecutionConfig,
    MarketConfig,
    RewardConfig,
    Side,
)

__version__ = "0.2.0"

__all__ = [
    "MarketConfig",
    "ExecutionConfig",
    "RewardConfig",
    "Side",
    "ActionType",
    "__version__",
]
