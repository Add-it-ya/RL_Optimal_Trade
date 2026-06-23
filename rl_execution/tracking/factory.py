"""Backend selection for experiment tracking.

``get_tracker("auto")`` prefers W&B when a ``WANDB_API_KEY`` is present, otherwise the
offline MLflow-local store; if the chosen backend's library is missing it degrades to the
next option and ultimately to :class:`NullTracker`.  Tracking is a side-channel -- a tracker
failure must never abort training -- so construction errors are caught and downgraded.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from rl_execution.tracking.base import ExperimentTracker, NullTracker
from rl_execution.tracking.mlflow_tracker import MLflowTracker
from rl_execution.tracking.wandb_tracker import WandbTracker

log = logging.getLogger(__name__)


def get_tracker(
    backend: str = "auto",
    *,
    experiment: str = "rl-execution",
    tracking_uri: Optional[str] = None,
    entity: Optional[str] = None,
) -> ExperimentTracker:
    """Return a tracker for ``backend`` in {``auto``, ``mlflow``, ``wandb``, ``null``}."""
    backend = (backend or "auto").lower()
    if backend == "null":
        return NullTracker()
    if backend == "auto":
        backend = "wandb" if os.environ.get("WANDB_API_KEY") else "mlflow"

    if backend == "wandb":
        try:
            return WandbTracker(project=experiment, entity=entity)
        except Exception as exc:  # pragma: no cover - depends on optional wandb install
            log.warning("W&B tracker unavailable (%s); falling back to MLflow-local.", exc)
            backend = "mlflow"

    if backend == "mlflow":
        try:
            return MLflowTracker(experiment=experiment, tracking_uri=tracking_uri)
        except Exception as exc:
            log.warning("MLflow tracker unavailable (%s); falling back to NullTracker.", exc)
            return NullTracker()

    log.warning("Unknown tracker backend %r; using NullTracker.", backend)
    return NullTracker()
