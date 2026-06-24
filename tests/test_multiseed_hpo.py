"""Tests for the Step-2 research harnesses: paired CIs + multiple-testing correction in the
backtest engine, registry-cached multi-seed evaluation, and resumable Optuna HPO.

The multi-seed test runs fully offline (an isolated SQLite MLflow store under ``tmp_path``) and
trains a tiny DQN twice to prove the second run loads from the registry instead of retraining.
The Optuna tests are skipped where optuna is not installed.
"""

import numpy as np
import pytest

from rl_execution.backtest import corrected_significance, episode_is, paired_is_table
from rl_execution.backtest.engine import BacktestResult


def _result(name, is_values):
    """A minimal BacktestResult carrying the given per-episode implementation-shortfall values."""
    res = BacktestResult(name=name)
    res.episode_metrics = [{"implementation_shortfall_bps": float(v)} for v in is_values]
    return res


# --------------------------------------------------------------------------- engine: paired CIs
def test_paired_is_table_reports_ci_and_pvalue():
    rng = np.random.default_rng(0)
    bench = rng.normal(50, 20, size=120)
    strat = bench - 8.0 + rng.normal(0, 1.0, size=120)  # consistently ~8 bps better than TWAP
    table = paired_is_table(
        {"TWAP": _result("TWAP", bench), "DQN": _result("DQN", strat)},
        benchmark="TWAP",
        n_boot=1000,
        rng=0,
    )
    for col in ("vs_TWAP", "vs_TWAP_ci_low", "vs_TWAP_ci_high", "p_value"):
        assert col in table.columns
    dqn = table.loc["DQN"]
    assert dqn["vs_TWAP"] < 0
    assert dqn["vs_TWAP_ci_high"] < 0  # whole interval below zero -> robust improvement
    assert dqn["p_value"] < 1e-3
    assert np.isnan(table.loc["TWAP"]["p_value"])  # benchmark vs itself -> undefined


def test_paired_is_table_ci_false_skips_interval_columns():
    table = paired_is_table(
        {"TWAP": _result("TWAP", [1, 2, 3]), "X": _result("X", [2, 3, 4])},
        benchmark="TWAP",
        ci=False,
    )
    assert "p_value" not in table.columns and "vs_TWAP_ci_low" not in table.columns


def test_episode_is_extracts_shortfall():
    assert np.allclose(episode_is(_result("X", [1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])


# ------------------------------------------------------------------ engine: corrected grid
def test_corrected_significance_grid_and_holm():
    rng = np.random.default_rng(1)
    by_regime = {}
    for regime in ("calm", "volatile"):
        bench = rng.normal(40, 15, size=100)
        good = bench - 10.0 + rng.normal(0, 1.0, size=100)  # clearly beats TWAP
        tie = bench + rng.normal(0, 15, size=100)  # no real edge
        by_regime[regime] = {
            "TWAP": _result("TWAP", bench),
            "DQN": _result("DQN", good),
            "Random": _result("Random", tie),
        }
    df = corrected_significance(by_regime, benchmark="TWAP", method="holm")
    assert set(df.columns) >= {
        "regime",
        "strategy",
        "vs_TWAP",
        "p_value",
        "p_adjusted",
        "reject_H0",
    }
    assert len(df) == 4  # 2 regimes x 2 non-benchmark strategies
    assert np.all(df["p_adjusted"] >= df["p_value"] - 1e-9)  # correction never lowers a p-value
    assert df[df["strategy"] == "DQN"]["reject_H0"].all()  # strong effect survives Holm


# --------------------------------------------------------------- multiseed registry caching
@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Isolate MLflow's tracking DB + artifact root under a temp dir (fully offline)."""
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", (tmp_path / "mlruns").as_uri())
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return tmp_path


def test_multiseed_uses_registry_cache(mlflow_store, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import rl_execution.experiments.multiseed as ms
    from rl_execution.config import ExecutionConfig, RunConfig

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr("rl_execution.training.MODELS_DIR", models_dir)

    calls = {"n": 0}
    real_train = ms.train_agent

    def counting_train(*a, **k):
        calls["n"] += 1
        return real_train(*a, **k)

    monkeypatch.setattr(ms, "train_agent", counting_train)

    rc = RunConfig(
        agent="dqn",
        timesteps=80,
        randomized=False,
        regime="normal_liquidity",
        eval_episodes=3,
        eval_base_seed=10_000,
        execution=ExecutionConfig(),
        agent_kwargs=dict(learning_starts=16, batch_size=8, buffer_size=200),
    )
    kw = dict(regimes=["normal_liquidity"], benchmark="TWAP", episodes=3)

    first = ms.run_multiseed("dqn", rc, seeds=[0], **kw)
    assert calls["n"] == 1
    assert first.per_seed_vs_bench["normal_liquidity"].shape == (1,)

    # Identical {config, seed}: must resolve from the registry rather than retrain.
    second = ms.run_multiseed("dqn", rc, seeds=[0], **kw)
    assert calls["n"] == 1

    across = second.across_seed_frame()
    assert {"regime", "across_seed_ci_low", "across_seed_ci_high", "n_seeds"} <= set(across.columns)


# ----------------------------------------------------------------------- HPO study plumbing
def test_search_space_per_algo():
    optuna = pytest.importorskip("optuna")
    from rl_execution.experiments.hpo import search_space

    # A fresh study per algorithm, as in real use: one study's parameter names must keep a
    # consistent categorical space, but different algos legitimately reuse names (e.g. batch_size).
    for algo, expected_key in (("dqn", "lr"), ("ppo", "learning_rate"), ("sac", "buffer_size")):
        study = optuna.create_study(direction="minimize")
        params = search_space(study.ask(), algo)
        assert expected_key in params


def test_hpo_study_resumes(tmp_path):
    pytest.importorskip("optuna")
    from rl_execution.experiments.hpo import make_study, run_study

    storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"

    def objective(trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        return x * x

    s1 = make_study("resume-test", storage=storage)
    run_study(s1, objective, n_trials=3)
    assert len(s1.trials) == 3

    s2 = make_study("resume-test", storage=storage)  # reopen the persisted study
    assert len(s2.trials) == 3  # resumed, not restarted
    run_study(s2, objective, n_trials=2)
    assert len(s2.trials) == 5
