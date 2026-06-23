"""Load / dump experiment configs from YAML.

A config file has two top-level sections, ``market`` and ``execution``, mapping onto
:class:`~rl_execution.config.MarketConfig` and
:class:`~rl_execution.config.ExecutionConfig`.  Unknown keys are ignored.

:func:`load_run_config` / :func:`dump_run_config` round-trip a full
:class:`~rl_execution.config.RunConfig` (which fully describes a reproducible run).
"""

from __future__ import annotations

from typing import Tuple

import yaml

from rl_execution.config import ExecutionConfig, MarketConfig, RunConfig


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


def load_run_config(path: str, *, validate: bool = True) -> RunConfig:
    """Load a full :class:`RunConfig` from YAML (optionally validating ranges)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    rc = RunConfig.from_dict(raw)
    if validate:
        rc.validate()
    return rc


def dump_run_config(run_config: RunConfig, path: str) -> None:
    """Serialise a :class:`RunConfig` to YAML so the run can be reproduced from the file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config.to_dict(), f, sort_keys=False)
