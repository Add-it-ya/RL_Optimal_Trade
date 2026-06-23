"""MLflow-backed tracker -- the default, fully-offline backend (local ``mlruns/`` store)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rl_execution.tracking.base import BaseTracker, flatten_dict
from rl_execution.tracking.registry import ensure_experiment, register_model

_MAX_PARAM_LEN = 250  # MLflow caps param value length; truncate defensively.


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) <= _MAX_PARAM_LEN else s[: _MAX_PARAM_LEN - 3] + "..."


class MLflowTracker(BaseTracker):
    """Logs params/metrics/artifacts to a local MLflow store and registers model versions."""

    def __init__(self, experiment: str = "rl-execution", tracking_uri: Optional[str] = None):
        import mlflow

        self._mlflow = mlflow
        # Selects (and, if needed, creates with a local artifact root) the experiment, and
        # points MLflow at the local SQLite store by default.
        ensure_experiment(experiment, tracking_uri=tracking_uri)
        self._active = False

    def start_run(
        self, name: str, config: Dict[str, Any], tags: Optional[Dict[str, Any]] = None
    ) -> None:
        self._mlflow.start_run(run_name=name)
        self._active = True
        if tags:
            self._mlflow.set_tags({str(k): _short(v) for k, v in flatten_dict(tags).items()})
        if config:
            self.log_params(config)

    def log_params(self, params: Dict[str, Any]) -> None:
        flat = {k: _short(v) for k, v in flatten_dict(params).items()}
        if flat:
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        numeric = {str(k): float(v) for k, v in metrics.items() if _is_number(v)}
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: str, kind: str = "file") -> None:
        self._mlflow.log_artifact(path)

    def log_model(self, path: str, name: str, metadata: Dict[str, Any]) -> str:
        # Registers within the active run, so the version links to this run's lineage.
        return register_model(path, name, metadata)

    def finish(self, status: str = "ok") -> None:
        if self._active:
            self._mlflow.end_run(status="FINISHED" if status == "ok" else "FAILED")
            self._active = False
