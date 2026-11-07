"""scripts/reproduce.py: re-scoring committed runs and checking the published tables."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_execution.cli import main

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce.py"


@pytest.fixture()
def reproduce():
    spec = importlib.util.spec_from_file_location("reproduce", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(j, ci_low=-0.1):
    return pd.DataFrame(
        {
            "regime": ["r", "r"],
            "strategy": ["AlmgrenChriss", "PPO"],
            "n_seeds": ["—", "10"],
            "J_bps": [28.87, j],
            "ci_low": [np.nan, ci_low],
        }
    )


def test_compare_tables_accepts_noise_and_flags_real_differences(reproduce):
    cols = {"J_bps": 1e-3, "ci_low": 1e-3}
    keys = ["regime", "strategy"]
    ok, worst, problems = reproduce.compare_tables(_table(29.0 + 4e-6), _table(29.0), keys, cols)
    assert ok and worst == pytest.approx(4e-6) and not problems  # NaN == NaN for AC's CI

    ok, worst, problems = reproduce.compare_tables(_table(29.01), _table(29.0), keys, cols)
    assert not ok and worst == pytest.approx(0.01) and "J_bps" in problems[0]

    ok, _, problems = reproduce.compare_tables(_table(29.0).iloc[:1], _table(29.0), keys, cols)
    assert not ok and "missing" in problems[0]

    ok, _, problems = reproduce.compare_tables(
        _table(29.0, ci_low=np.nan), _table(29.0), keys, cols
    )
    assert not ok and "NaN in one table only" in problems[0]


def test_evaluate_command_repeats_the_recorded_evaluation(reproduce, tmp_path):
    meta = {
        "runs_dir": "runs/e1b_5m",
        "regimes": ["signal_ic000"],
        "episodes": 5000,
        "base_seed": 2_000_000,
        "antithetic": True,
    }
    cmd = reproduce.evaluate_command("e2_ic000", meta, tmp_path, jobs=2)
    joined = " ".join(cmd)
    assert "--experiment e1b_5m" in joined and "--regimes signal_ic000" in joined
    assert "--episodes 5000 --base-seed 2000000" in joined and cmd[-1] == "--antithetic"
    assert cmd[cmd.index("--runs-dir") + 1] == str(reproduce.ROOT / "runs")


def test_a_runs_folder_that_changed_since_the_evaluation_is_refused(
    reproduce, tmp_path, monkeypatch
):
    for seed, sha in (("seed_0", "a" * 64), ("seed_1", "b" * 64)):
        folder = tmp_path / "runs" / "t" / "PPO" / seed
        folder.mkdir(parents=True)
        (folder / "provenance.json").write_text(json.dumps({"weights_sha256": sha}))
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    both = {
        "runs_dir": "runs/t",
        "runs": [{"weights_sha256": "a" * 64}, {"weights_sha256": "b" * 64}],
    }
    assert reproduce.runs_mismatch(both) == ""
    one = {"runs_dir": "runs/t", "runs": [{"weights_sha256": "a" * 64}]}  # a seed was added later
    assert "holds 2 run(s), 1 of them among the 1" in reproduce.runs_mismatch(one)


def test_reproduce_passes_on_its_own_results_and_fails_on_a_tampered_table(
    reproduce, tmp_path, monkeypatch
):
    runs, results = tmp_path / "runs", tmp_path / "results"
    train = ["train", "--agent", "doubledqn", "--timesteps", "200", "--seeds", "0-1"]
    train += ["--regime", "abm_ac_optimal", "--experiment", "t", "--runs-dir", str(runs)]
    main(train + ["--eval-every", "100", "--val-episodes", "4"])
    main(
        ["evaluate", "--experiment", "t", "--runs-dir", str(runs), "--out", str(results / "t")]
        + ["--episodes", "4", "--antithetic"]
    )
    main(["report", "--experiment", "t", "--results", str(results / "t")])

    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    monkeypatch.setattr(reproduce, "RESULTS", results)
    monkeypatch.setattr(reproduce, "OUT", results / "_reproduce")
    assert reproduce.committed() == ["t"]
    name, ok, note = reproduce.reproduce_one("t", jobs=1, tol_bps=1e-3)
    assert ok, note
    assert "worst deviation 0," in note  # same machine: bit-identical
    assert (results / "_reproduce" / "t" / "per_seed.csv").exists()

    for file in ("comparison.csv", "per_seed.csv"):  # a changed committed number is caught
        original = (results / "t" / file).read_bytes()
        table = pd.read_csv(results / "t" / file)
        table.loc[
            table["agent" if "agent" in table else "strategy"] == "DoubleDQN", "J_bps"
        ] += 0.01
        table.to_csv(results / "t" / file, index=False)
        ok, note = reproduce.check("t", tol_bps=1e-3)
        assert not ok and "J_bps" in note
        (results / "t" / file).write_bytes(original)
    assert reproduce.check("t", tol_bps=1e-3)[0]

    meta = json.loads((results / "t" / "meta.json").read_text(encoding="utf-8"))
    assert meta["runs_dir"].endswith("runs/t")  # POSIX, wherever it was written
