"""The regime families the experiments are reported on: the E3 split and the E2 signal sweep."""

from dataclasses import asdict

import numpy as np
import pytest

from rl_execution.experiments import (
    HELD_OUT,
    IN_DISTRIBUTION,
    REGIMES,
    SIGNAL_REGIMES,
    SIGNAL_TARGET_IC,
    get_regime,
    list_regimes,
    outside_randomized_support,
    randomized_market_config,
    resolve_regimes,
)
from rl_execution.experiments.calibration import CALIBRATION_BASE_SEED, measure_signal


# --------------------------------------------------------------------------- E3 split
def test_the_randomised_sampler_stays_inside_its_own_support():
    rng = np.random.default_rng(0)
    for _ in range(500):
        assert outside_randomized_support(randomized_market_config(rng)) == []


@pytest.mark.parametrize("name", IN_DISTRIBUTION)
def test_in_distribution_regimes_lie_inside_the_training_family(name):
    assert outside_randomized_support(get_regime(name)) == []


@pytest.mark.parametrize("name", HELD_OUT)
def test_held_out_regimes_lie_outside_the_training_family(name):
    assert outside_randomized_support(get_regime(name)), f"{name} is not out of distribution"


def test_held_out_regimes_move_the_axes_they_are_named_for():
    def axes(name):
        return set(outside_randomized_support(get_regime(name)))

    assert axes("low_vol") == {"volatility"}
    assert {"volatility", "vol_of_vol"} <= axes("high_vol")
    assert {"base_depth", "temporary_impact"} <= axes("thin") & axes("deep")
    assert axes("bull") == axes("bear") == {"drift"}
    assert axes("gbm_medium") == {"price_process"}
    assert "leverage_corr" in axes("leverage")


def test_the_split_is_disjoint_and_covers_the_grid():
    assert not set(IN_DISTRIBUTION) & set(HELD_OUT)
    assert set(list_regimes(include_brownian=True)) - {"abm_ac_optimal"} == set(
        IN_DISTRIBUTION + HELD_OUT
    )


def test_resolve_regimes_expands_groups_in_order_without_duplicates():
    assert resolve_regimes(["in_distribution"]) == IN_DISTRIBUTION
    assert resolve_regimes(["low_vol", "volatility"]) == ["low_vol", "medium_vol", "high_vol"]
    assert resolve_regimes(["held_out", "normal_liquidity"]) == HELD_OUT + ["normal_liquidity"]
    with pytest.raises(KeyError):
        resolve_regimes(["no_such_regime"])


def test_every_regime_resolves_to_itself():
    for name in REGIMES:
        assert resolve_regimes([name]) == [name]


# --------------------------------------------------------------------------- E2 sweep
def test_signal_regimes_differ_from_the_ground_truth_only_in_the_signal():
    truth = asdict(get_regime("abm_ac_optimal"))
    assert asdict(get_regime("signal_ic000")) == truth
    for name in SIGNAL_REGIMES:
        cfg = asdict(get_regime(name))
        assert {k for k in cfg if cfg[k] != truth[k]} <= {"imbalance_alpha"}
    alphas = [get_regime(name).imbalance_alpha for name in SIGNAL_REGIMES]
    assert alphas == sorted(alphas) and alphas[0] == 0.0


def test_signal_regimes_stay_out_of_the_default_grid():
    assert not set(SIGNAL_REGIMES) & set(list_regimes(include_brownian=True))
    assert resolve_regimes(["signal"]) == SIGNAL_REGIMES


@pytest.mark.parametrize("name", SIGNAL_REGIMES)
def test_signal_regimes_hit_their_target_ic_on_fresh_seeds(name):
    """Solved on the calibration block; re-measured here on a disjoint one."""
    ic = measure_signal(get_regime(name), n_episodes=2000, seed=CALIBRATION_BASE_SEED + 500_000)
    assert ic["ic_observed"] == pytest.approx(SIGNAL_TARGET_IC[name], abs=0.01)
