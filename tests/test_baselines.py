import numpy as np
import pytest

from rl_execution.backtest import compare_strategies, evaluate, run_episode
from rl_execution.baselines import (
    TWAP,
    VWAP,
    AlmgrenChriss,
    DumpAtStart,
    SignalRule,
    WaitThenDump,
    make_baseline,
)
from rl_execution.config import (
    ActionType,
    ExecutionConfig,
    MarketConfig,
    RewardConfig,
    ac_risk_aversion,
)
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import get_regime
from rl_execution.metrics.stats import paired_bootstrap_ci

ALL = [TWAP, VWAP, AlmgrenChriss, DumpAtStart, WaitThenDump, SignalRule]


def make_env(action_type=ActionType.CONTINUOUS, market=None):
    return ExecutionEnv(market or MarketConfig(), ExecutionConfig(action_type=action_type))


def _schedule(strategy, seed=0, **kwargs):
    _, hist, _, _ = run_episode(make_env(**kwargs), strategy, seed=seed)
    return hist["shares"].to_numpy() / 10_000.0


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("action_type", [ActionType.CONTINUOUS, ActionType.DISCRETE])
def test_baseline_executes_full_inventory(cls, action_type):
    summary, _, _, _ = run_episode(make_env(action_type), cls(), seed=0)
    assert summary["unexecuted_shares"] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(summary["objective_bps"])


def test_twap_is_uniform():
    assert np.allclose(_schedule(TWAP()), 1.0 / 20, rtol=1e-6)


def test_dump_at_start_is_one_block():
    assert _schedule(DumpAtStart()) == pytest.approx([1.0])


def test_wait_then_dump_trades_only_on_the_last_step():
    sched = _schedule(WaitThenDump())
    assert len(sched) == 20
    assert np.allclose(sched[:-1], 0.0) and sched[-1] == pytest.approx(1.0)


# =============================================================== Almgren-Chriss
def test_almgren_chriss_matches_the_reward_risk_aversion():
    env = make_env()
    env.reset(seed=0)
    matched, explicit = AlmgrenChriss(), AlmgrenChriss(
        ac_risk_aversion(env.market_config, env.exec_config)
    )
    matched.reset(env)
    explicit.reset(env)
    assert np.allclose(matched.trades, explicit.trades)
    assert matched.trades.sum() == pytest.approx(env.total_inventory, rel=1e-9)
    assert np.all(np.diff(matched.trades) <= 1e-9)  # risk-averse => front-loaded
    assert matched.trades[0] > 2.0 * env.total_inventory / env.horizon


def test_almgren_chriss_schedule_is_followed_exactly():
    env = make_env()
    ac = AlmgrenChriss()
    _, hist, _, _ = run_episode(env, ac, seed=1)
    assert np.allclose(hist["shares"].to_numpy(), ac.trades, rtol=1e-4)


def test_almgren_chriss_reduces_to_twap_without_risk():
    ac = AlmgrenChriss(risk_aversion=0.0)
    env = make_env()
    env.reset(seed=0)
    ac.reset(env)
    assert np.allclose(ac.trades, ac.trades[0], rtol=1e-6)


def test_almgren_chriss_is_best_on_the_objective_in_its_own_world():
    """E1 sanity check: with no signal and linear impact, AC beats every null on J.

    Antithetic pairs make the paired differences nearly deterministic for fixed schedules,
    so a modest number of episodes gives intervals well clear of zero.
    """
    market = get_regime("abm_ac_optimal")
    strategies = {
        "AlmgrenChriss": AlmgrenChriss(),
        "TWAP": TWAP(),
        "DumpAtStart": DumpAtStart(),
        "WaitThenDump": WaitThenDump(),
    }
    results = compare_strategies(
        lambda: ExecutionEnv(market), strategies, n_episodes=100, progress=False, antithetic=True
    )

    def objective(name):
        return np.array([m["objective_bps"] for m in results[name].episode_metrics])

    ac = objective("AlmgrenChriss")
    for rival in ("TWAP", "DumpAtStart", "WaitThenDump"):
        lo, hi = paired_bootstrap_ci(objective(rival), ac, n_boot=2_000, rng=0, block=2)
        assert lo > 0, f"AC should beat {rival} on the objective, CI ({lo:.2f}, {hi:.2f})"


# =============================================================== SignalRule
def test_signal_rule_slows_down_when_waiting_pays():
    env = make_env()
    obs, _ = env.reset(seed=0)
    rule = SignalRule(beta=2.0)
    rule.reset(env)
    calm, favourable, adverse = obs.copy(), obs.copy(), obs.copy()
    calm[3], favourable[3], adverse[3] = 0.0, 0.3, -0.3
    pace = rule._decide_fraction(calm, {})
    assert rule._decide_fraction(favourable, {}) < pace < rule._decide_fraction(adverse, {})


def test_signal_rule_with_zero_beta_is_almgren_chriss():
    assert np.allclose(_schedule(SignalRule(beta=0.0)), _schedule(AlmgrenChriss()), rtol=1e-4)


def test_signal_rule_fit_picks_the_best_beta_on_validation_seeds():
    factory = lambda: ExecutionEnv(MarketConfig())  # noqa: E731
    rule = SignalRule()
    rule.fit(factory, n_episodes=40, base_seed=500, betas=(0.0, 2.0))
    assert rule.beta in (0.0, 2.0)
    scores = {
        b: evaluate(factory, SignalRule(beta=b), 40, 500).aggregate["objective_bps_mean"]
        for b in (0.0, 2.0)
    }
    assert rule.beta == min(scores, key=scores.get)
    assert rule.params()["beta"] == rule.beta


def test_signal_rule_is_ac_without_a_signal():
    """With no alpha the fitted rule cannot do better than AC by timing; beta=0 must win
    or tie on the validation objective (up to noise from the book's imbalance)."""
    market = MarketConfig(imbalance_alpha=0.0)
    factory = lambda: ExecutionEnv(market)  # noqa: E731
    rule = SignalRule()
    rule.fit(factory, n_episodes=60, base_seed=700, betas=(0.0, 5.0))
    assert rule.beta == 0.0


# =============================================================== factory
def test_factory():
    assert isinstance(make_baseline("twap"), TWAP)
    assert isinstance(make_baseline("AC", risk_aversion=1e-7), AlmgrenChriss)
    assert isinstance(make_baseline("wait-then-dump"), WaitThenDump)
    assert isinstance(make_baseline("DumpAtStart"), DumpAtStart)
    with pytest.raises(KeyError):
        make_baseline("nonexistent")


def test_reward_config_round_trip_ignores_retired_fields():
    """Old configs carried temp_impact_penalty / unexecuted_penalty; they are ignored."""
    rc = RewardConfig.from_dict({"risk_aversion": 2e-3, "temp_impact_penalty": 1.0})
    assert rc.risk_aversion == 2e-3
