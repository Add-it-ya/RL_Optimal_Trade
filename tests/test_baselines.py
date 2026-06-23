import numpy as np
import pytest

from rl_execution.baselines import (
    AlmgrenChriss, POV, RandomStrategy, TWAP, VWAP, make_baseline,
)
from rl_execution.config import ExecutionConfig, MarketConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.backtest import run_episode


def make_env():
    return ExecutionEnv(MarketConfig(), ExecutionConfig(horizon=20))


@pytest.mark.parametrize("strat", [TWAP(), VWAP(), POV(0.2), RandomStrategy(0),
                                   AlmgrenChriss(1e-7)])
def test_baseline_executes_full_inventory(strat):
    env = make_env()
    summary, hist, rewards, _ = run_episode(env, strat, seed=0)
    assert summary["unexecuted_shares"] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(summary["implementation_shortfall_bps"])


def test_twap_is_uniform():
    env = make_env()
    strat = TWAP()
    _, hist, _, _ = run_episode(env, strat, seed=0)
    # every non-final step trades the same nominal amount
    shares = hist["shares"].to_numpy()[:-1]
    assert np.allclose(shares, shares[0], rtol=1e-6)


def test_almgren_chriss_front_loads():
    ac = AlmgrenChriss(risk_aversion=1e-6)
    env = make_env()
    env.reset(seed=0)
    ac.reset(env)
    trades = ac.trades
    assert trades.sum() == pytest.approx(env.total_inventory, rel=1e-6)
    # risk-averse schedule is monotonically decreasing
    assert np.all(np.diff(trades) <= 1e-6)


def test_almgren_chriss_reduces_to_twap_without_risk():
    ac = AlmgrenChriss(risk_aversion=0.0)
    env = make_env()
    env.reset(seed=0)
    ac.reset(env)
    assert np.allclose(ac.trades, ac.trades[0], rtol=1e-6)


def test_factory():
    assert isinstance(make_baseline("twap"), TWAP)
    assert isinstance(make_baseline("AC", risk_aversion=1e-7), AlmgrenChriss)
    with pytest.raises(KeyError):
        make_baseline("nonexistent")
