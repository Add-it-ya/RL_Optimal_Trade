"""Provenance capture so every artifact answers "what produced this?".

A trained model is only trustworthy if you can trace it back to the exact code, config,
data and seed that created it.  :func:`capture_provenance` gathers that lineage into a
plain ``dict`` that gets stamped into each model's JSON sidecar (and logged as tracker
tags), making every artifact self-describing and auditable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Distribution names whose versions we record by default (importlib uses *distribution*
# names, e.g. "stable-baselines3", not the import name "stable_baselines3").
_DEFAULT_LIBS = (
    "numpy",
    "pandas",
    "scipy",
    "gymnasium",
    "torch",
    "stable-baselines3",
    "mlflow",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> Optional[str]:
    """Run a git command inside the repo; return stripped stdout or ``None`` on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def git_sha() -> str:
    """Current commit SHA, or ``"unknown"`` outside a git checkout."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_is_dirty() -> bool:
    """``True`` if the working tree has uncommitted changes (so the SHA under-describes it)."""
    status = _git("status", "--porcelain")
    return bool(status)  # non-empty porcelain output == dirty; None (no git) -> False


def canonical_json(obj: Any) -> str:
    """Deterministic JSON string: sorted keys, no insignificant whitespace.

    Used as the pre-image for :func:`config_hash`, so equal configs always hash equally
    regardless of key ordering.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "value"):  # Enum
        return o.value
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


def config_hash(config: Any) -> str:
    """SHA-256 of the canonical JSON of ``config`` (dataclass, dict or anything JSON-able)."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def lib_versions(packages: Iterable[str] = _DEFAULT_LIBS) -> Dict[str, str]:
    """Installed version of each distribution, or ``"not-installed"`` if absent."""
    out: Dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def capture_provenance(
    *,
    seed: Optional[int] = None,
    config: Any = None,
    data_hash: Optional[str] = None,
    extra_libs: Iterable[str] = (),
) -> Dict[str, Any]:
    """Assemble the full lineage record for an artifact.

    Parameters
    ----------
    seed:
        The seed the run was launched with.
    config:
        The run configuration; its :func:`config_hash` is recorded (omitted if ``None``).
    data_hash:
        Content hash of the dataset.  ``None`` for pure-synthetic runs (a real data hash
        arrives with the data pipeline in Step 3).
    extra_libs:
        Additional distribution names to version-stamp beyond the defaults.
    """
    record: Dict[str, Any] = {
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "config_hash": config_hash(config) if config is not None else None,
        "data_hash": data_hash,
        "seed": seed,
        "lib_versions": lib_versions((*_DEFAULT_LIBS, *extra_libs)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return record
