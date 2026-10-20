"""Pin the market calibration: the default market must pose a real execution problem.

If a change to the simulator or its defaults moves any of these quantities out of its
band, the agents would be training on a different problem -- re-run
``scripts/calibrate.py`` and recalibrate deliberately rather than let it drift.
"""

from dataclasses import replace

import pytest

from rl_execution.config import ExecutionConfig, MarketConfig, RewardConfig
from rl_execution.experiments.calibration import (
    TARGETS,
    ac_schedule,
    calibration_report,
    measure_signal,
    solve_risk_aversion,
)


@pytest.fixture(scope="module")
def report():
    return calibration_report(n_episodes=200)


@pytest.mark.parametrize("key", sorted(TARGETS))
def test_default_market_meets_calibration_target(report, key):
    low, high = TARGETS[key]
    assert low <= report[key] <= high, f"{key} = {report[key]:.3f}, target [{low}, {high}]"


def test_reference_schedules_rank_sensibly_on_the_objective(report):
    """AC < TWAP < dumping at once < waiting then dumping."""
    j = {k.removeprefix("objective_"): v for k, v in report.items() if k.startswith("objective_")}
    assert j["AlmgrenChriss"] < j["TWAP"] < j["DumpAtStart"] < j["WaitThenDump"]


def test_signal_strength_scales_with_alpha():
    market = MarketConfig()
    none = measure_signal(replace(market, imbalance_alpha=0.0), n_episodes=1000)
    strong = measure_signal(replace(market, imbalance_alpha=4 * market.imbalance_alpha), 1000)
    assert abs(none["ic_observed"]) < 0.03
    assert strong["ic_observed"] > 0.2


def test_solve_risk_aversion_hits_the_target_urgency():
    market, execution = MarketConfig(), ExecutionConfig()
    lam = solve_risk_aversion(market, execution, target_kappa_t=2.0)
    tuned = replace(execution, reward=RewardConfig(risk_aversion=lam))
    assert ac_schedule(market, tuned)["kappa_T"] == pytest.approx(2.0, rel=1e-3)
