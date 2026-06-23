"""Smoke tests for the command-line entry points and the scoped-warnings helper."""

import importlib

import pytest

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
