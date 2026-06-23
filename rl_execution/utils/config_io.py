"""Load / dump experiment configs from YAML.

A config file has two top-level sections, ``market`` and ``execution``, mapping onto
:class:`~rl_execution.config.MarketConfig` and
:class:`~rl_execution.config.ExecutionConfig`.  Unknown keys are ignored.
"""

from __future__ import annotations

from typing import Tuple

import yaml

from rl_execution.config import ExecutionConfig, MarketConfig


def load_config(path: str) -> Tuple[MarketConfig, ExecutionConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    market = MarketConfig.from_dict(raw.get("market", {}))
    execution = ExecutionConfig.from_dict(raw.get("execution", {}))
    return market, execution


def dump_config(market: MarketConfig, execution: ExecutionConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"market": market.to_dict(), "execution": execution.to_dict()},
            f,
            sort_keys=False,
        )
