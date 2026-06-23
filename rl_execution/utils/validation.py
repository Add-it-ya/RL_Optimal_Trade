"""Pydantic validation of a serialised :class:`~rl_execution.config.RunConfig`.

Kept in its own module and importing :mod:`pydantic` lazily so the core package validates
configs *without* requiring the optional ``tracking`` extra -- :meth:`RunConfig.validate`
falls back to equivalent manual range checks when pydantic is absent.
"""

from __future__ import annotations

from typing import Any, Dict, Literal


def validate_run_config(d: Dict[str, Any]) -> None:
    """Validate a ``RunConfig.to_dict()`` payload; raise ``ValueError`` if out of range.

    Raises ``ImportError`` if pydantic is not installed (the caller falls back to manual
    checks).
    """
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

    class _Market(BaseModel):
        # extra="allow": only constrain the safety-critical params, pass the rest through.
        model_config = ConfigDict(extra="allow")
        initial_price: float = Field(gt=0)
        volatility: float = Field(ge=0)
        imbalance_alpha: float = Field(ge=0, le=1)
        imbalance_persistence: float = Field(ge=0, lt=1)

    class _Execution(BaseModel):
        model_config = ConfigDict(extra="allow")
        horizon: int = Field(gt=0)
        total_inventory: float = Field(gt=0)

    class _Run(BaseModel):
        model_config = ConfigDict(extra="allow")
        timesteps: int = Field(gt=0)
        seed: int = Field(ge=0)
        eval_episodes: int = Field(gt=0)
        tracker: Literal["auto", "mlflow", "wandb", "null"]
        market: _Market
        execution: _Execution

    try:
        _Run.model_validate(d)
    except ValidationError as exc:
        raise ValueError(f"Invalid RunConfig:\n{exc}") from exc
