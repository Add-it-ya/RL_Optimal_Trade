"""Smoke tests for the command-line entry points and the scoped-warnings helper."""

import hashlib
import importlib

import pytest

from rl_execution.utils.io import (
    FIGURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    resolve_artifact_dirs,
)
from rl_execution.utils.warnings import configure_cli_warnings

CLI_MODULES = [
    "rl_execution.cli.train",
    "rl_execution.cli.evaluate",
    "rl_execution.cli.experiments",
    "rl_execution.cli.report",
]


def test_configure_cli_warnings_runs():
    configure_cli_warnings()  # must not raise


@pytest.mark.parametrize("module", CLI_MODULES)
def test_cli_modules_import_and_expose_main(module):
    mod = importlib.import_module(module)
    assert callable(mod.main)


def test_experiments_parse_args(monkeypatch):
    from rl_execution.cli import experiments

    monkeypatch.setattr("sys.argv", ["rlx-experiments", "--quick", "--agents", "dqn"])
    args = experiments.parse_args()
    assert args.quick is True
    assert args.agents == ["dqn"]
    assert args.out_dir is None and args.models_dir is None


# ---------------------------------------------------------------- artefact sandboxing
def test_resolve_artifact_dirs_sandboxes_quick_runs():
    """--quick defaults to the sandbox; an explicit path always wins."""
    assert resolve_artifact_dirs() == (RESULTS_DIR, MODELS_DIR)
    assert resolve_artifact_dirs(quick=True) == (RESULTS_DIR / "quick", MODELS_DIR / "quick")
    out, models = resolve_artifact_dirs("/tmp/r", "/tmp/m", quick=True)
    assert (out.name, models.name) == ("r", "m")


def _snapshot(*dirs):
    """{path: sha1} for every file under ``dirs`` that exists."""
    snap = {}
    for d in dirs:
        for f in sorted(d.rglob("*")) if d.exists() else []:
            if f.is_file():
                snap[str(f)] = hashlib.sha1(f.read_bytes()).hexdigest()
    return snap


@pytest.mark.slow
def test_quick_run_writes_to_a_sandbox_dir(tmp_path, monkeypatch):
    """A quick pipeline run pointed at tmp_path leaves the published artefacts byte-identical."""
    pytest.importorskip("torch")
    from rl_execution.cli import experiments

    before = _snapshot(RESULTS_DIR, MODELS_DIR, FIGURES_DIR)
    out_dir, models_dir = tmp_path / "results", tmp_path / "models"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rlx-experiments",
            "--quick",
            "--agents",
            "dqn",
            "--regimes",
            "normal_liquidity",
            "--timesteps",
            "200",
            "--episodes",
            "2",
            "--tracker",
            "null",
            "--no-register",
            "--out-dir",
            str(out_dir),
            "--models-dir",
            str(models_dir),
        ],
    )
    experiments.main()

    assert (out_dir / "regime_results.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "figures" / "inventory_decay.png").exists()
    assert (models_dir / "dqn.json").exists()
    assert _snapshot(RESULTS_DIR, MODELS_DIR, FIGURES_DIR) == before
