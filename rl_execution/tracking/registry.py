"""Versioned model registry backed by the **MLflow** Model Registry (local, offline).

Fully free and offline.  MLflow's plain-filesystem store is in maintenance mode and no longer
serves the Model Registry, so the local default uses a **SQLite** tracking database
(``mlflow.db``) with a local **file artifact root** (``mlruns/``) -- still no server, account or
network.  Registering a model creates an immutable, numbered version with provenance tags and
run lineage, so models become *versioned artifacts* instead of files that overwrite each other
in ``models/``.

The flat ``models/<tag>.pt`` file that :func:`rl_execution.training.save_agent` still writes is
just a convenient "latest working copy"; the registry is the authoritative versioned store (and
the resolution point for the Step 5 serving layer).
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

from rl_execution.tracking.base import flatten_dict
from rl_execution.utils.io import PROJECT_ROOT

# MLflow's import-time logging is chatty; quiet it to warnings.
logging.getLogger("mlflow").setLevel(logging.WARNING)

_DEFAULT_DB = PROJECT_ROOT / "mlflow.db"
_DEFAULT_ARTIFACTS = PROJECT_ROOT / "mlruns"
_DEFAULT_EXPERIMENT = "rl-execution"


def using_external_store() -> bool:
    """True when the user has pointed MLflow at their own tracking store via the env var."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def default_tracking_uri() -> str:
    """Local SQLite tracking URI (honours ``MLFLOW_TRACKING_URI`` if set)."""
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{_DEFAULT_DB.as_posix()}"


def default_artifact_location() -> Optional[str]:
    """Local file artifact root for our default store; ``None`` when using an external store."""
    if os.environ.get("MLFLOW_ARTIFACT_ROOT"):
        return os.environ["MLFLOW_ARTIFACT_ROOT"]
    if using_external_store():
        return None  # let the external store decide where artifacts live
    return _DEFAULT_ARTIFACTS.as_uri()


def configure_mlflow(tracking_uri: Optional[str] = None) -> None:
    """Point MLflow at the (local by default) tracking store."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or default_tracking_uri())


def ensure_experiment(
    name: str = _DEFAULT_EXPERIMENT, *, tracking_uri: Optional[str] = None
) -> str:
    """Select ``name`` (creating it with a local artifact root if needed); return its id."""
    import mlflow
    from mlflow.exceptions import MlflowException

    configure_mlflow(tracking_uri)
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        mlflow.set_experiment(name)
        return exp.experiment_id
    try:
        exp_id = mlflow.create_experiment(name, artifact_location=default_artifact_location())
    except MlflowException:  # pragma: no cover - race / already-exists
        mlflow.set_experiment(name)
        exp = mlflow.get_experiment_by_name(name)
        return exp.experiment_id if exp else ""
    mlflow.set_experiment(name)
    return exp_id


def _tagify(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Flatten + stringify metadata into MLflow-safe ``str -> str`` tags."""
    return {str(k): str(v) for k, v in flatten_dict(metadata).items()}


def register_model(
    local_path: str,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    tracking_uri: Optional[str] = None,
    experiment: str = _DEFAULT_EXPERIMENT,
) -> str:
    """Log ``local_path`` (+ its ``<stem>.json`` sidecar, if present) and register a version.

    Reuses the active MLflow run when one exists (so the version links to that run's params and
    metrics) and otherwise opens a short-lived run.  Each model is logged under its own artifact
    sub-path (``name``) so several models can be registered from a single run.  ``metadata``
    (provenance, config, metrics) is attached as version tags.  Returns the new version number.
    """
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    configure_mlflow(tracking_uri)
    model_file = Path(local_path)
    sidecar = model_file.with_suffix(".json")
    artifact_path = name  # distinct per model -> no collision when sharing a run

    active = mlflow.active_run()
    if active is None:
        ensure_experiment(experiment, tracking_uri=tracking_uri)
    run_ctx = (
        nullcontext(active) if active is not None else mlflow.start_run(run_name=f"register-{name}")
    )

    with run_ctx as ctx:
        run = active if active is not None else ctx
        run_id = run.info.run_id
        mlflow.log_artifact(str(model_file), artifact_path=artifact_path)
        if sidecar.exists() and sidecar != model_file:
            mlflow.log_artifact(str(sidecar), artifact_path=artifact_path)

        client = MlflowClient()
        try:
            client.create_registered_model(name)
        except MlflowException:
            pass  # already exists
        mv = client.create_model_version(
            name=name, source=f"runs:/{run_id}/{artifact_path}", run_id=run_id
        )
        for key, value in _tagify(metadata or {}).items():
            client.set_model_version_tag(name, mv.version, key, value)
    return str(mv.version)


def resolve_version(
    name: str, version: str = "latest", *, tracking_uri: Optional[str] = None
) -> str:
    """Resolve ``"latest"`` to the highest numeric version; pass other values through."""
    from mlflow.tracking import MlflowClient

    configure_mlflow(tracking_uri)
    if version != "latest":
        return str(version)
    versions = MlflowClient().search_model_versions(f"name='{name}'")
    if not versions:
        raise ValueError(f"No registered versions for model '{name}'")
    return str(max(int(v.version) for v in versions))


def load_registered(
    name: str, version: str = "latest", *, tracking_uri: Optional[str] = None
) -> str:
    """Resolve a registered model version to a **local model-file path**.

    Downloads the version's artifacts to a temp dir and returns the model file (the
    non-``.json`` artifact); the sidecar JSON is downloaded alongside it (same directory) so
    callers can rebuild the agent from it.
    """
    import mlflow

    resolved = resolve_version(name, version, tracking_uri=tracking_uri)
    dst = tempfile.mkdtemp(prefix=f"rlx-{name}-v{resolved}-")
    local_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{resolved}", dst_path=dst
    )

    model_files = [p for p in Path(local_dir).iterdir() if p.is_file() and p.suffix != ".json"]
    if not model_files:
        raise FileNotFoundError(f"No model file in registered artifacts for '{name}' v{resolved}")
    return str(model_files[0])


def load_registered_agent(
    name: str, env: Any, version: str = "latest", *, tracking_uri: Optional[str] = None
) -> Any:
    """Resolve + download a registered version and rebuild a usable agent bound to ``env``."""
    from rl_execution.training import load_agent

    model_file = load_registered(name, version, tracking_uri=tracking_uri)
    return load_agent(Path(model_file).stem, env, models_dir=Path(model_file).parent)
