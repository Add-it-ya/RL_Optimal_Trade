"""Tests for the ``rlx`` command line: argument handling and a tiny end-to-end pipeline."""

import numpy as np
import pandas as pd
import pytest

from rl_execution.cli import build_parser, comparison_table, main, parse_seeds
from rl_execution.utils.warnings import configure_cli_warnings


def test_configure_cli_warnings_runs():
    configure_cli_warnings()  # must not raise


@pytest.mark.parametrize(
    "spec, expected",
    [("3", [3]), ("0-4", [0, 1, 2, 3, 4]), ("0,2,7", [0, 2, 7]), ("1-2,5", [1, 2, 5])],
)
def test_parse_seeds(spec, expected):
    assert parse_seeds(spec) == expected


def test_parser_has_the_three_commands():
    parser = build_parser()
    for argv in (["train"], ["evaluate", "--experiment", "x"], ["report", "--experiment", "x"]):
        assert callable(parser.parse_args(argv).func)


def test_comparison_table_pairs_episodes_and_aggregates_seeds():
    """Paired gaps are formed per episode; a multi-seed agent's CI is across seeds."""
    rng = np.random.default_rng(0)
    market = rng.normal(0.0, 100.0, size=50)  # shared price risk, cancels in the pairing
    rows = []
    for name, agent, seed, shift in [
        ("TWAP", "TWAP", np.nan, 0.0),
        ("PPO/s0", "PPO", 0.0, -5.0),
        ("PPO/s1", "PPO", 1.0, -7.0),
    ]:
        for i, m in enumerate(market):
            rows.append(
                {
                    "regime": "r",
                    "strategy": name,
                    "agent": agent,
                    "seed": seed,
                    "episode": i,
                    "implementation_shortfall_bps": m + shift,
                }
            )
    table = comparison_table(pd.DataFrame(rows)).set_index("strategy")
    assert table.loc["PPO", "n_seeds"] == 2
    assert table.loc["PPO", "vs_TWAP"] == pytest.approx(-6.0)
    assert table.loc["PPO", "ci_low"] <= -6.0 <= table.loc["PPO", "ci_high"]
    assert table.loc["TWAP", "n_seeds"] == "—"
    assert np.isnan(table.loc["TWAP", "p_value"])


@pytest.mark.slow
def test_train_evaluate_report_end_to_end(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    train = ["train", "--agent", "doubledqn", "--timesteps", "300", "--seeds", "0-1"]
    train += ["--regime", "abm_ac_optimal", "--experiment", "t", "--runs-dir", str(runs)]
    main(train)
    seed0 = runs / "t" / "DoubleDQN" / "seed_0"
    assert (seed0 / "provenance.json").exists()

    # re-running skips completed seeds instead of retraining them
    before = (seed0 / "model.pt").stat().st_mtime_ns
    main(train)
    assert (seed0 / "model.pt").stat().st_mtime_ns == before

    main(
        ["evaluate", "--experiment", "t", "--runs-dir", str(runs), "--out", str(results)]
        + ["--episodes", "6", "--antithetic"]
    )
    episodes = pd.read_csv(results / "episodes.csv")
    assert set(episodes["regime"]) == {"abm_ac_optimal"}  # evaluated where it was trained
    assert {"TWAP", "AlmgrenChriss", "DoubleDQN/s0", "DoubleDQN/s1"} <= set(episodes["strategy"])
    assert (results / "figures" / "abm_ac_optimal_schedule.png").exists()

    main(["report", "--experiment", "t", "--results", str(results)])
    report = (results / "REPORT.md").read_text(encoding="utf-8")
    assert "## abm_ac_optimal" in report and "DoubleDQN" in report
