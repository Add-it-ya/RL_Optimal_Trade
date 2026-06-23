import numpy as np

from rl_execution.config import MarketConfig, Side
from rl_execution.envs.lob import build_book, walk_book
from rl_execution.envs.market import MarketSimulator


def test_book_is_sorted_and_two_sided():
    rng = np.random.default_rng(0)
    snap = build_book(100.0, 0.02, MarketConfig(), 0.0, rng)
    assert snap.best_ask > snap.best_bid
    assert np.all(np.diff(snap.bid_prices) < 0)   # bids decrease away from touch
    assert np.all(np.diff(snap.ask_prices) > 0)   # asks increase away from touch
    assert snap.spread > 0


def test_walk_book_fills_requested_quantity():
    rng = np.random.default_rng(1)
    snap = build_book(100.0, 0.02, MarketConfig(), 0.0, rng)
    fill = walk_book(snap, Side.SELL, 1500.0)
    assert fill.filled_shares == 1500.0
    # selling consumes bids: average price cannot exceed the best bid
    assert fill.avg_price <= snap.best_bid + 1e-9


def test_large_order_sweeps_past_depth():
    rng = np.random.default_rng(2)
    cfg = MarketConfig(base_depth=100.0, n_levels=5)
    snap = build_book(100.0, 0.02, cfg, 0.0, rng)
    huge = snap.bid_sizes.sum() * 5
    fill = walk_book(snap, Side.SELL, huge)
    assert fill.filled_shares == huge                 # everything fills
    assert fill.worst_price < snap.bid_prices[-1]     # swept beyond displayed depth


def test_imbalance_skews_depth():
    rng = np.random.default_rng(3)
    bid_heavy = build_book(100.0, 0.02, MarketConfig(), imbalance_factor=0.8, rng=rng)
    assert bid_heavy.imbalance > 0


def test_permanent_impact_moves_mid_adversely():
    sim = MarketSimulator(MarketConfig(permanent_impact=1e-4), rng=np.random.default_rng(0))
    sim.reset(horizon=10)
    mid0 = sim.mid
    res = sim.execute(Side.SELL, 1000.0)
    assert res.mid_after < mid0                        # selling pushes the mid down
    assert res.perm_impact < 0
