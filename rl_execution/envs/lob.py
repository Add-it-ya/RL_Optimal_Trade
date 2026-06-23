"""Limit-order-book snapshot construction and market-order execution (book walking).

The book is represented as a static :class:`LOBSnapshot` (a depth ladder per side).
Each simulation step a fresh snapshot is generated around the prevailing mid-price; an
incoming market order is filled by *walking* the relevant side of the ladder, which
produces a convex (size-dependent) execution cost — the microstructural source of
slippage and temporary market impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_execution.config import MarketConfig, Side


@dataclass
class LOBSnapshot:
    """A point-in-time limit order book.

    Prices are sorted from the touch outward, i.e. ``bid_prices[0]`` is the best (highest)
    bid and ``ask_prices[0]`` is the best (lowest) ask.
    """

    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray
    mid: float

    @property
    def best_bid(self) -> float:
        return float(self.bid_prices[0])

    @property
    def best_ask(self) -> float:
        return float(self.ask_prices[0])

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def microprice(self) -> float:
        """Size-weighted touch price (a better fair-value estimate than the mid)."""
        bq, aq = self.bid_sizes[0], self.ask_sizes[0]
        denom = bq + aq
        if denom <= 0:
            return self.mid
        return (self.best_bid * aq + self.best_ask * bq) / denom

    @property
    def imbalance(self) -> float:
        """Order-book imbalance in [-1, 1]: +1 = bid-heavy (buy pressure)."""
        bid = float(self.bid_sizes.sum())
        ask = float(self.ask_sizes.sum())
        denom = bid + ask
        if denom <= 0:
            return 0.0
        return (bid - ask) / denom

    def depth(self, levels: int | None = None) -> float:
        """Total resting shares within the top ``levels`` of both sides."""
        if levels is None:
            return float(self.bid_sizes.sum() + self.ask_sizes.sum())
        return float(self.bid_sizes[:levels].sum() + self.ask_sizes[:levels].sum())


@dataclass
class Fill:
    """Aggregate result of walking the book with a market order."""

    filled_shares: float
    avg_price: float
    worst_price: float
    notional: float
    levels_consumed: int


def build_book(
    mid: float,
    spread: float,
    config: MarketConfig,
    imbalance_factor: float,
    rng: np.random.Generator,
) -> LOBSnapshot:
    """Generate a fresh order-book snapshot around ``mid``.

    Parameters
    ----------
    mid:
        Current mid-price.
    spread:
        Current absolute bid-ask spread (price units).
    imbalance_factor:
        Latent factor in roughly [-1, 1]; positive skews resting depth toward the bid
        (buy pressure), negative toward the ask.  Drives the observable imbalance feature.
    """
    tick = config.tick_size
    n = config.n_levels
    half = max(spread / 2.0, tick)

    best_bid = mid - half
    best_ask = mid + half

    # price ladders stepping away from the touch by one tick per level
    level_idx = np.arange(n)
    bid_prices = best_bid - level_idx * tick
    ask_prices = best_ask + level_idx * tick

    # depth grows away from the touch; multiplicative log-noise per level
    growth = (1.0 + config.level_growth) ** level_idx
    bid_noise = np.exp(rng.normal(0.0, config.depth_vol, size=n))
    ask_noise = np.exp(rng.normal(0.0, config.depth_vol, size=n))

    # imbalance skews the two sides in opposite directions
    skew = config.imbalance_strength * float(np.clip(imbalance_factor, -1.0, 1.0))
    bid_scale = 1.0 + skew
    ask_scale = 1.0 - skew

    bid_sizes = config.base_depth * growth * bid_noise * bid_scale
    ask_sizes = config.base_depth * growth * ask_noise * ask_scale

    bid_sizes = np.maximum(bid_sizes, 1.0)
    ask_sizes = np.maximum(ask_sizes, 1.0)

    return LOBSnapshot(
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        mid=mid,
    )


def walk_book(snapshot: LOBSnapshot, side: Side, shares: float) -> Fill:
    """Execute a market order of ``shares`` against ``snapshot``.

    A ``SELL`` parent order consumes the *bid* ladder (we sell into resting buy orders);
    a ``BUY`` consumes the *ask* ladder.  Liquidity beyond the displayed depth is filled
    at the worst displayed level with a widening concession of one tick per exhausted
    book, modelling the cost of sweeping past visible size.
    """
    if shares <= 0:
        return Fill(0.0, snapshot.mid, snapshot.mid, 0.0, 0)

    if side is Side.SELL:
        prices = snapshot.bid_prices
        sizes = snapshot.bid_sizes
        direction = -1.0  # extra liquidity gets progressively cheaper (worse for a seller)
    else:
        prices = snapshot.ask_prices
        sizes = snapshot.ask_sizes
        direction = +1.0  # extra liquidity gets progressively more expensive

    remaining = float(shares)
    notional = 0.0
    filled = 0.0
    levels_consumed = 0
    worst_price = float(prices[0])
    tick = float(prices[1] - prices[0]) if len(prices) > 1 else 0.01
    tick = abs(tick)

    for price, size in zip(prices, sizes):
        if remaining <= 0:
            break
        take = min(remaining, float(size))
        notional += take * float(price)
        filled += take
        remaining -= take
        worst_price = float(price)
        levels_consumed += 1

    # sweep past displayed depth: fill the remainder beyond the deepest level with a
    # one-tick-per-book concession so very large orders are penalised super-linearly.
    if remaining > 0:
        sweep_price = float(prices[-1]) + direction * tick
        notional += remaining * sweep_price
        filled += remaining
        worst_price = sweep_price
        remaining = 0.0
        levels_consumed = len(prices)

    avg_price = notional / filled if filled > 0 else snapshot.mid
    return Fill(
        filled_shares=filled,
        avg_price=avg_price,
        worst_price=worst_price,
        notional=notional,
        levels_consumed=levels_consumed,
    )
