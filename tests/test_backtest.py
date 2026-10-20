"""Backtest-engine bookkeeping."""

import numpy as np
import pytest

from rl_execution.backtest import evaluate, pool_results, schedule_gap_table
from rl_execution.baselines import TWAP, AlmgrenChriss, DumpAtStart
from rl_execution.envs import ExecutionEnv


def _factory():
    return ExecutionEnv()


def test_mean_schedule_pads_short_episodes_instead_of_truncating():
    """A one-step episode must not cut every other schedule down to one step."""
    mixed = pool_results(
        [evaluate(_factory, TWAP(), 3), evaluate(_factory, DumpAtStart(), 3)], "mixed"
    )
    schedule = mixed.mean_schedule()
    assert schedule.size == 20
    assert schedule.sum() == pytest.approx(10_000.0)
    assert schedule[0] == pytest.approx(0.5 * 500.0 + 0.5 * 10_000.0)


def test_schedule_gap_is_not_biased_against_front_loaded_policies():
    results = {
        "AlmgrenChriss": evaluate(_factory, AlmgrenChriss(), 4),
        "DumpAtStart": evaluate(_factory, DumpAtStart(), 4),
        "TWAP": evaluate(_factory, TWAP(), 4),
    }
    gaps = schedule_gap_table(results, reference="TWAP")["schedule_L1"]
    # one block trade vs TWAP: |1 - 1/20| + 19 * 1/20 = 1.9 (truncation used to give 0.95)
    assert gaps["DumpAtStart"] == pytest.approx(1.9, abs=1e-6)
    assert 0.2 < gaps["AlmgrenChriss"] < 1.0


def test_pool_results_concatenates_episodes():
    a, b = evaluate(_factory, TWAP(), 2, base_seed=0), evaluate(_factory, TWAP(), 3, base_seed=9)
    pooled = pool_results([a, b], "TWAP")
    assert pooled.aggregate["n_episodes"] == 5
    assert np.isfinite(pooled.aggregate["objective_bps_mean"])
