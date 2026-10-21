"""Tests for the ``rlx`` command line: argument handling and a tiny end-to-end pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from rl_execution.cli import (
    build_parser,
    comparison_table,
    main,
    parse_seeds,
    schedule_l1,
    signal_sweep_table,
    split_degradation,
)
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
    table = comparison_table(
        pd.DataFrame(rows), benchmark="TWAP", metric="implementation_shortfall_bps"
    ).set_index("strategy")
    assert table.loc["PPO", "n_seeds"] == 2
    assert table.loc["PPO", "vs_TWAP"] == pytest.approx(-6.0)
    assert table.loc["PPO", "ci_low"] <= -6.0 <= table.loc["PPO", "ci_high"]
    assert table.loc["TWAP", "n_seeds"] == "—"
    assert table.loc["PPO", "n_degenerate"] == 0  # no effective_steps column -> not flagged
    assert np.isnan(table.loc["TWAP", "p_value"])
    assert np.isnan(table.loc["PPO", "L1_vs_AC"])  # no schedules given


def test_comparison_table_reports_float_noise_as_an_exact_tie():
    """SignalRule with a fitted beta of 0 trades like AC up to ~4e-7 bps of rounding."""
    rng = np.random.default_rng(1)
    market = rng.normal(30.0, 50.0, size=40)
    noise = rng.normal(0.0, 1e-7, size=40)
    rows = [
        {"regime": "r", "strategy": s, "agent": s, "seed": np.nan, "episode": i}
        | {"objective_bps": m + (e if s == "SignalRule" else 0.0)}
        for s in ("AlmgrenChriss", "SignalRule")
        for i, (m, e) in enumerate(zip(market, noise))
    ]
    row = comparison_table(pd.DataFrame(rows)).set_index("strategy").loc["SignalRule"]
    assert row["vs_AlmgrenChriss"] == 0.0 and row["ci_low"] == row["ci_high"] == 0.0
    assert row["p_value"] == 1.0 and row["p_holm"] == 1.0


def _schedule_rows(regime, strategy, agent, seed, fractions):
    return [
        {"regime": regime, "strategy": strategy, "agent": agent, "seed": seed, "step": i}
        | {"fraction": f}
        for i, f in enumerate(fractions)
    ]


def test_schedule_l1_is_zero_padded_and_the_table_takes_the_median_seed():
    ac = [0.4, 0.3, 0.2, 0.1]
    rows = _schedule_rows("r", "AlmgrenChriss", "AlmgrenChriss", np.nan, ac)
    rows += _schedule_rows("r", "TWAP", "TWAP", np.nan, [0.25] * 4)
    rows += _schedule_rows("r", "PPO/s0", "PPO", 0.0, ac)  # L1 0
    rows += _schedule_rows("r", "PPO/s1", "PPO", 1.0, [0.5, 0.3, 0.2])  # stops early: L1 0.2
    rows += _schedule_rows("r", "PPO/s2", "PPO", 2.0, [1.0])  # block trade: L1 1.2
    schedules = pd.DataFrame(rows)
    l1 = schedule_l1(schedules)
    assert l1[("r", "AlmgrenChriss")] == 0.0
    assert l1[("r", "TWAP")] == pytest.approx(0.4)
    assert l1[("r", "PPO/s1")] == pytest.approx(0.2)  # the missing last step counts
    assert l1[("r", "PPO/s2")] == pytest.approx(1.2)

    episodes = pd.DataFrame(
        [
            {"regime": "r", "strategy": s, "agent": a, "seed": k, "episode": i}
            | {"objective_bps": 30.0 + i}
            for s, a, k in [
                ("AlmgrenChriss", "AlmgrenChriss", np.nan),
                ("TWAP", "TWAP", np.nan),
                ("PPO/s0", "PPO", 0.0),
                ("PPO/s1", "PPO", 1.0),
                ("PPO/s2", "PPO", 2.0),
            ]
            for i in range(4)
        ]
    )
    table = comparison_table(episodes, schedules=schedules).set_index("strategy")
    assert table.loc["PPO", "L1_vs_AC"] == pytest.approx(0.2)  # median of 0, 0.2, 1.2
    assert table.loc["TWAP", "L1_vs_AC"] == pytest.approx(0.4)
    assert table.loc["AlmgrenChriss", "L1_vs_AC"] == 0.0


def test_split_degradation_compares_held_out_to_in_distribution_per_seed():
    """Seed 0 is 1 bps off AC in distribution and 3 bps off held out; seed 1: 2 and 6."""
    gaps = {
        ("in_a", 0): 1.0, ("in_b", 0): 1.0, ("out_a", 0): 2.0, ("out_b", 0): 4.0,
        ("in_a", 1): 2.0, ("in_b", 1): 2.0, ("out_a", 1): 6.0, ("out_b", 1): 6.0,
    }  # fmt: skip
    rows = []
    for regime in ("in_a", "in_b", "out_a", "out_b"):
        for i in range(6):
            base = 25.0 + 3.0 * i
            rows.append({"regime": regime, "strategy": "AlmgrenChriss", "agent": "AlmgrenChriss"}
                        | {"seed": np.nan, "episode": i, "objective_bps": base})  # fmt: skip
            rows.append({"regime": regime, "strategy": "TWAP", "agent": "TWAP", "seed": np.nan}
                        | {"episode": i, "objective_bps": base + 4.0})  # fmt: skip
            for seed in (0, 1):
                rows.append(
                    {"regime": regime, "strategy": f"PPO/s{seed}", "agent": "PPO", "seed": seed}
                    | {"episode": i, "objective_bps": base + gaps[(regime, seed)]}
                )
    splits = {"in_distribution": ["in_a", "in_b"], "held_out": ["out_a", "out_b"]}
    table = split_degradation(pd.DataFrame(rows), splits).set_index("strategy")
    assert "AlmgrenChriss" not in table.index
    assert table.loc["PPO", "n_seeds"] == 2
    assert table.loc["PPO", "vs_AlmgrenChriss_in"] == pytest.approx(1.5)
    assert table.loc["PPO", "vs_AlmgrenChriss_held_out"] == pytest.approx(4.5)
    assert table.loc["PPO", "degradation"] == pytest.approx(3.0)  # seeds: 2 and 4
    assert 2.0 <= table.loc["PPO", "ci_low"] <= 3.0 <= table.loc["PPO", "ci_high"] <= 4.0
    assert table.loc["TWAP", "degradation"] == pytest.approx(0.0)
    assert table.loc["TWAP", "n_seeds"] == "—" and np.isnan(table.loc["TWAP", "ci_low"])


def test_evaluate_in_parallel_writes_the_same_results(tmp_path):
    """``--jobs`` runs regimes in worker processes; the files must not depend on it."""
    common = ["evaluate", "--experiment", "none", "--runs-dir", str(tmp_path / "runs")]
    common += ["--episodes", "4", "--antithetic", "--regimes", "abm_ac_optimal", "low_vol"]
    main(common + ["--out", str(tmp_path / "serial")])
    main(common + ["--out", str(tmp_path / "parallel"), "--jobs", "2"])
    for name in ("episodes.csv", "schedules.csv", "summary.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "serial" / name), pd.read_csv(tmp_path / "parallel" / name)
        )
    serial = json.loads((tmp_path / "serial" / "meta.json").read_text())
    parallel = json.loads((tmp_path / "parallel" / "meta.json").read_text())
    assert serial["params"] == parallel["params"] and serial["regimes"] == [
        "abm_ac_optimal",
        "low_vol",
    ]
    assert (tmp_path / "parallel" / "figures" / "low_vol_schedule.png").exists()


@pytest.mark.slow
def test_train_evaluate_report_end_to_end(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    train = ["train", "--agent", "doubledqn", "--timesteps", "300", "--seeds", "0-1"]
    train += ["--regime", "abm_ac_optimal", "--experiment", "t", "--runs-dir", str(runs)]
    train += ["--eval-every", "150", "--val-episodes", "10"]
    main(train)
    seed0 = runs / "t" / "DoubleDQN" / "seed_0"
    assert (seed0 / "provenance.json").exists()
    val_curve = pd.read_csv(seed0 / "val_curve.csv")
    assert list(val_curve["step"]) == [0, 150, 300]

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
    assert {"DumpAtStart", "WaitThenDump", "SignalRule"} <= set(episodes["strategy"])
    assert {"objective_bps", "effective_steps"} <= set(episodes.columns)
    assert (results / "figures" / "abm_ac_optimal_schedule.png").exists()
    schedules = pd.read_csv(results / "schedules.csv")
    full = schedules.groupby("strategy")["fraction"].sum()
    assert full.to_numpy() == pytest.approx(1.0)  # every strategy completes the order

    main(["report", "--experiment", "t", "--results", str(results)])
    report = (results / "REPORT.md").read_text(encoding="utf-8")
    assert "## abm_ac_optimal" in report and "DoubleDQN" in report
    assert "vs_AlmgrenChriss" in report and "J_bps" in report and "L1_vs_AC" in report
    comparison = pd.read_csv(results / "comparison.csv").set_index("strategy")
    assert comparison.loc["AlmgrenChriss", "L1_vs_AC"] == 0.0
    assert comparison.loc["DoubleDQN", "L1_vs_AC"] >= 0.0
    assert "SignalRule coefficient fitted on validation seeds" in report
    assert "## Checkpoint selection" in report and "n_degenerate" in report
    assert "code version(s): " in report
    assert (results / "figures" / "validation_curves.png").exists()


def _sweep_episodes(regime, ppo_shifts, signal_rule_shift, n=30, seed=0):
    """AC, SignalRule and a multi-seed PPO on shared market noise, offset by fixed shifts."""
    market = np.random.default_rng(seed).normal(30.0, 60.0, size=n)
    runs = [("AlmgrenChriss", "AlmgrenChriss", np.nan, 0.0)]
    runs += [("SignalRule", "SignalRule", np.nan, signal_rule_shift)]
    runs += [(f"PPO/s{k}", "PPO", float(k), shift) for k, shift in enumerate(ppo_shifts)]
    return pd.DataFrame(
        [
            {"regime": regime, "strategy": s, "agent": a, "seed": k, "episode": i}
            | {"objective_bps": m + shift}
            for s, a, k, shift in runs
            for i, m in enumerate(market)
        ]
    )


def test_signal_sweep_table_flips_gaps_into_improvements():
    inputs = {
        "e2_ic000": (_sweep_episodes("signal_ic000", [0.1, 0.3], 0.0), 1),
        "e2_ic100": (_sweep_episodes("signal_ic100", [-2.0, -3.0], -1.0, seed=1), 1),
    }
    sweep = signal_sweep_table(inputs, n_boot=2_000)
    ppo = sweep[(sweep.strategy == "PPO") & (sweep.benchmark == "AlmgrenChriss")]
    assert list(ppo["ic"]) == [0.0, 0.1]
    assert list(ppo["improvement"]) == pytest.approx([-0.2, 2.5])  # positive = cheaper
    assert (ppo["ci_low"] <= ppo["improvement"]).all()
    assert (ppo["improvement"] <= ppo["ci_high"]).all()
    vs_rule = sweep[(sweep.strategy == "PPO") & (sweep.benchmark == "SignalRule")]
    assert list(vs_rule["improvement"]) == pytest.approx([-0.2, 1.5])
    rule = sweep[(sweep.strategy == "SignalRule") & (sweep.benchmark == "AlmgrenChriss")]
    assert list(rule["improvement"]) == pytest.approx([0.0, 1.0])
    assert not ((sweep.strategy == "SignalRule") & (sweep.benchmark == "SignalRule")).any()
    tested = sweep["p_value"].notna()
    assert (sweep.loc[tested, "p_holm"] >= sweep.loc[tested, "p_value"]).all()


def test_sweep_command_writes_table_report_and_figure(tmp_path):
    for name, regime, shifts in [
        ("e2_ic000", "signal_ic000", [0.2, 0.1]),
        ("e2_ic050", "signal_ic050", [-1.0, -1.5]),
    ]:
        res = tmp_path / name
        res.mkdir()
        _sweep_episodes(regime, shifts, -0.5).to_csv(res / "episodes.csv", index=False)
        (res / "meta.json").write_text(json.dumps({"antithetic": False}))
    out = tmp_path / "e2"
    main(["sweep", "--experiments", "e2_ic000", "e2_ic050"]
         + ["--results-root", str(tmp_path), "--out", str(out)])  # fmt: skip
    sweep = pd.read_csv(out / "sweep.csv")
    assert set(sweep["ic"]) == {0.0, 0.05}
    report = (out / "SWEEP.md").read_text(encoding="utf-8")
    assert "## Improvement over AlmgrenChriss" in report
    assert "## Improvement over SignalRule" in report
    assert (out / "figures" / "improvement_vs_ic.png").exists()
