import numpy as np
import pytest

from rl_execution.config import ExecutionConfig, MarketConfig, Side
from rl_execution.data import HistoricalMarketSource, synthetic_lob_dataframe
from rl_execution.envs import ExecutionEnv, MultiAgentSimulator, Participant
from rl_execution.baselines import TWAP, AlmgrenChriss
from rl_execution.backtest import run_episode
from rl_execution.experiments import get_regime, list_regimes, randomized_market_config


def test_historical_replay_runs():
    df = synthetic_lob_dataframe(500, MarketConfig(), seed=0)
    assert {"mid", "spread", "volume", "bid_px_1", "ask_sz_1"}.issubset(df.columns)
    src = HistoricalMarketSource(df, MarketConfig())
    env = ExecutionEnv(MarketConfig(), ExecutionConfig(horizon=20), market_source=src)
    summary, hist, _, _ = run_episode(env, TWAP(), seed=1)
    assert summary["unexecuted_shares"] == pytest.approx(0.0, abs=1e-6)
    assert len(hist) == 20


def test_multi_agent_simulation():
    sim = MultiAgentSimulator(MarketConfig(), horizon=15)
    sim.add_participant(Participant("A", TWAP(), Side.SELL, 12_000))
    sim.add_participant(Participant("B", AlmgrenChriss(1e-7), Side.SELL, 8_000))
    out = sim.run(seed=0)
    tbl = out["table"]
    assert len(tbl) == 2
    assert np.isfinite(tbl["implementation_shortfall_bps"]).all()
    # all participants fully execute
    for p in sim.participants:
        assert p.remaining == pytest.approx(0.0, abs=1e-6)


def test_regimes_available():
    names = list_regimes()
    assert {"low_vol", "high_vol", "thin", "deep", "bull", "bear", "sideways"} <= set(names)
    cfg = get_regime("bull")
    assert cfg.drift > 0


def test_randomized_config_in_bounds():
    cfg = randomized_market_config(np.random.default_rng(0))
    assert 0.008 <= cfg.volatility <= 0.05
    assert 200.0 <= cfg.base_depth <= 1500.0
