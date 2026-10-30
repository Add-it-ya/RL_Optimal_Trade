"""The lab tuning study: its configs and the rule that picks the winner."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from rl_execution.utils.config_io import load_run_config
from rl_execution.vec.train import vec_config

LAB = Path(__file__).resolve().parents[1] / "scripts" / "lab"
CONFIGS = sorted((LAB / "configs").glob("*.yaml"))


def _pick_config():
    spec = importlib.util.spec_from_file_location("pick_config", LAB / "pick_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_there_are_four_configs_a_to_d():
    assert [p.name[0] for p in CONFIGS] == ["A", "B", "C", "D"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_tuning_configs_load_and_keep_selection_pressure_equal(path):
    rc = load_run_config(str(path))  # validates
    cfg = vec_config(rc)  # rejects unknown agent_kwargs
    assert rc.agent == "vecppo" and rc.regime == "abm_ac_optimal" and not rc.randomized
    assert rc.timesteps // rc.eval_every == 50  # same number of checkpoints as config A
    assert cfg.minibatch_size <= cfg.n_envs * rc.execution.horizon  # fits in one rollout


def test_config_a_is_the_phase4_default():
    rc = load_run_config(str(LAB / "configs" / "A_n256_5m.yaml"))
    assert (rc.timesteps, rc.n_envs, rc.eval_every, rc.agent_kwargs) == (
        5_000_000,
        256,
        100_000,
        {},
    )


def _row(label, gap, degenerate=0):
    return {"label": label, "gap_vs_AC": gap, "n_degenerate": degenerate}


def test_choose_takes_the_lowest_gap():
    pick = _pick_config()
    assert pick.choose([_row("A", 0.23), _row("C", 0.10), _row("D", 0.02), _row("B", 0.20)]) == "D"


def test_choose_prefers_a_cheaper_config_within_the_tie_margin():
    pick = _pick_config()
    rows = [_row("A", 0.23), _row("C", 0.06), _row("D", 0.02), _row("B", 0.01)]
    assert pick.choose(rows, tie_bps=0.05) == "C"  # 0.06 <= 0.01 + 0.05; C is cheaper
    assert pick.choose(rows, tie_bps=0.0) == "B"


def test_choose_skips_configs_with_a_degenerate_seed():
    pick = _pick_config()
    rows = [_row("A", 0.23), _row("C", -0.50, degenerate=1), _row("D", 0.10)]
    assert pick.choose(rows) == "D"
    with pytest.raises(ValueError):
        pick.choose([_row("A", 0.1, degenerate=2)])


def test_candidate_row_reads_the_vecppo_line(tmp_path):
    pick = _pick_config()
    pd.DataFrame(
        [
            {"strategy": "AlmgrenChriss", "n_seeds": "—", "n_degenerate": "—", "J_bps": 28.87,
             "vs_AlmgrenChriss": 0.0, "ci_low": float("nan"), "ci_high": float("nan"),
             "L1_vs_AC": 0.0},
            {"strategy": "VecPPO", "n_seeds": 10, "n_degenerate": 0, "J_bps": 29.10,
             "vs_AlmgrenChriss": 0.23, "ci_low": 0.10, "ci_high": 0.38, "L1_vs_AC": 0.074},
        ]
    ).to_csv(tmp_path / "comparison.csv", index=False)  # fmt: skip
    row = pick.candidate_row("A", tmp_path)
    assert row["gap_vs_AC"] == pytest.approx(0.23) and row["n_seeds"] == 10
    assert pick.config_file("C") == "scripts/lab/configs/C_n4096_50m.yaml"
