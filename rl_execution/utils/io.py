"""Filesystem paths and JSON helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Training runs: ``runs/<experiment>/<agent>/seed_<k>/`` (see :mod:`rl_execution.training`).
RUNS_DIR = PROJECT_ROOT / "runs"
#: Evaluation outputs and reports: ``results/<experiment>/``.
RESULTS_DIR = PROJECT_ROOT / "results"


def portable_path(path: os.PathLike | str) -> str:
    """``path`` as a POSIX string relative to the project root when it lies inside it.

    Paths recorded in ``meta.json`` stay valid on another machine or checkout (and carry
    no local user names); a path outside the project is kept absolute.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_run_dir(recorded: str, runs_dir: os.PathLike | str = RUNS_DIR) -> Path:
    """Find a run folder recorded in ``meta.json``, on this machine.

    A relative path is taken from the project root. If the recorded folder does not exist
    (e.g. an absolute path from another machine), its last three parts
    (``<experiment>/<Agent>/seed_<k>``) are looked up under ``runs_dir``.
    """
    p = Path(recorded)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.exists():
        return p
    parts = [s for s in recorded.replace("\\", "/").split("/") if s]
    return Path(runs_dir).joinpath(*parts[-3:])


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default(o: Any):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


def save_json(obj: Any, path: os.PathLike | str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
