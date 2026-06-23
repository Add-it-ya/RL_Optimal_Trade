"""Experiment tracking & model registry.

A single :class:`ExperimentTracker` interface with swappable backends so experiment code
never imports W&B / MLflow directly:

* :class:`NullTracker`   -- no-op (unit tests / offline CI).
* :class:`MLflowTracker` -- local ``mlruns/`` store **and** model registry (the default).
* :class:`WandbTracker`  -- opt-in free tier, lazily imported.

Use :func:`get_tracker` to pick one, and :func:`register_model` / :func:`load_registered`
for the versioned model registry.
"""

from rl_execution.tracking.base import (
    BaseTracker,
    ExperimentTracker,
    NullTracker,
    flatten_dict,
)
from rl_execution.tracking.factory import get_tracker
from rl_execution.tracking.mlflow_tracker import MLflowTracker
from rl_execution.tracking.registry import (
    load_registered,
    load_registered_agent,
    register_model,
    resolve_version,
)
from rl_execution.tracking.wandb_tracker import WandbTracker

__all__ = [
    "ExperimentTracker",
    "BaseTracker",
    "NullTracker",
    "MLflowTracker",
    "WandbTracker",
    "get_tracker",
    "flatten_dict",
    "register_model",
    "load_registered",
    "load_registered_agent",
    "resolve_version",
]
