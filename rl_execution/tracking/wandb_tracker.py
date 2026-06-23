"""Weights & Biases tracker -- opt-in, free-tier, lazily imported.

Honours ``WANDB_MODE=offline`` (no account/network needed for a local run that can be synced
later) and ``WANDB_API_KEY`` for the hosted free tier.  ``wandb`` is imported only when this
backend is actually constructed, so it stays an optional dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rl_execution.tracking.base import BaseTracker


class WandbTracker(BaseTracker):
    """Logs runs / metrics / model artifacts to Weights & Biases."""

    def __init__(self, project: str = "rl-execution", entity: Optional[str] = None):
        import wandb

        self._wandb = wandb
        self._project = project
        self._entity = entity
        self._run = None

    def start_run(
        self, name: str, config: Dict[str, Any], tags: Optional[Dict[str, Any]] = None
    ) -> None:
        # Tag *values* (git_sha, etc.) are most useful as config; W&B's own tags are a flat
        # list of strings, so we also surface the tag keys there for filtering.
        cfg = dict(config or {})
        if tags:
            cfg["provenance"] = tags
        self._run = self._wandb.init(
            project=self._project,
            entity=self._entity,
            name=name,
            config=cfg,
            tags=list(tags.keys()) if tags else None,
            reinit=True,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._run is not None:
            self._run.log(dict(metrics), step=step)

    def log_artifact(self, path: str, kind: str = "file") -> None:
        if self._run is None:
            return
        art = self._wandb.Artifact(name=Path(path).stem, type=kind)
        art.add_file(path)
        self._run.log_artifact(art)

    def log_model(self, path: str, name: str, metadata: Dict[str, Any]) -> str:
        if self._run is None:
            return ""
        art = self._wandb.Artifact(name=name, type="model", metadata=metadata)
        art.add_file(path)
        sidecar = Path(path).with_suffix(".json")
        if sidecar.exists() and sidecar != Path(path):
            art.add_file(str(sidecar))
        logged = self._run.log_artifact(art, aliases=["latest"])
        try:
            logged.wait()  # block until the version id is assigned
        except Exception:  # pragma: no cover - offline mode may defer versioning
            pass
        return getattr(logged, "version", "latest") or "latest"

    def finish(self, status: str = "ok") -> None:
        if self._run is not None:
            self._run.finish(exit_code=0 if status == "ok" else 1)
            self._run = None
