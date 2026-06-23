"""The tracking abstraction: one interface, swappable backends.

Experiment code talks only to :class:`ExperimentTracker` so it never imports W&B or MLflow
directly.  Concrete backends (MLflow-local by default, W&B opt-in, Null for tests/CI) live in
sibling modules and are selected by :func:`rl_execution.tracking.get_tracker`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ExperimentTracker(Protocol):
    """Structural interface every tracking backend satisfies."""

    def start_run(
        self, name: str, config: Dict[str, Any], tags: Optional[Dict[str, Any]] = None
    ) -> None: ...

    def log_params(self, params: Dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None: ...

    def log_artifact(self, path: str, kind: str = "file") -> None: ...

    def log_model(self, path: str, name: str, metadata: Dict[str, Any]) -> str: ...

    def finish(self, status: str = "ok") -> None: ...


class BaseTracker:
    """No-op defaults + context-manager sugar; concrete trackers override the methods.

    Using a tracker as a context manager guarantees :meth:`finish` runs (with status
    ``"failed"`` if the body raised)::

        with get_tracker() as tracker:
            tracker.start_run(...)
            ...
    """

    def start_run(
        self, name: str, config: Dict[str, Any], tags: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str, kind: str = "file") -> None:
        pass

    def log_model(self, path: str, name: str, metadata: Dict[str, Any]) -> str:
        return ""

    def finish(self, status: str = "ok") -> None:
        pass

    def __enter__(self) -> "BaseTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # returning None (falsy) propagates any exception rather than suppressing it
        self.finish("failed" if exc_type is not None else "ok")


class NullTracker(BaseTracker):
    """A tracker that records nothing -- the default for unit tests and offline CI."""


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dicts into dotted keys (MLflow params/tags must be scalar).

    ``{"market": {"volatility": 0.02}}`` -> ``{"market.volatility": 0.02}``.
    """
    out: Dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, full, sep))
        else:
            out[full] = value
    return out
