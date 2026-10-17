"""Stochastic market simulator: price dynamics, depth/imbalance evolution and impact.

The simulator owns the mid-price process and re-quotes a :class:`LOBSnapshot` each step.
It exposes two operations to the environment:

* :meth:`execute` -- fill a market order against the current book, applying temporary
  (transient) and permanent (persistent) market impact.
* :meth:`advance` -- move the market one step along its exogenous path.

The price follows an explicit Brownian process from :mod:`rl_execution.envs.processes`.
The entire exogenous market (price, volatility, imbalance, spread, ambient volume,
per-level depth noise) is drawn **up front** at :meth:`reset`, and market impact is applied
as an additive ``shift`` on top of it:

.. math::

    \\text{mid}_t = \\text{exogenous}_t + \\text{intra-step deviation} + \\text{cumulative impact}

Because the exogenous path is fixed before the first action, no trading decision can consume
randomness that alters the path -- two strategies run on the same seed face *exactly* the
same market.  That is what makes the paired (common-random-number) comparisons in
:func:`~rl_execution.backtest.engine.paired_is_table` a genuine variance reduction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rl_execution.config import MarketConfig, Side
from rl_execution.envs.brownian import NoiseStreams, Stream, make_streams
from rl_execution.envs.lob import LOBSnapshot, build_book, walk_book
from rl_execution.envs.processes import MarketPaths, generate_market_paths, make_price_process


@dataclass
class ExecutionResult:
    """Outcome of executing a market order, with cost decomposition."""

    filled_shares: float
    avg_price: float  # effective price actually paid/received (incl. temp impact)
    raw_avg_price: float  # price from walking the book only (excl. temp impact)
    notional: float  # effective cash exchanged (avg_price * filled_shares)
    mid_before: float  # mid at execution time (pre permanent impact)
    mid_after: float  # mid after permanent impact has been applied
    slippage: float  # signed cost vs mid_before, as a fraction (+ = adverse)
    temp_impact_cost: float  # cash cost of transient impact
    perm_impact: float  # signed mid shift caused by this order (price units)


class MarketSimulator:
    """Synthetic market with a re-quoting limit order book.

    Parameters
    ----------
    config:
        :class:`~rl_execution.config.MarketConfig`.
    rng:
        Optional numpy ``Generator`` for reproducibility.
    vol_window:
        Number of recent returns used to estimate realised volatility.
    streams:
        Optional pre-built :class:`~rl_execution.envs.brownian.NoiseStreams`.  When omitted
        one is derived from ``rng``.
    """

    def __init__(
        self,
        config: MarketConfig,
        rng: np.random.Generator | None = None,
        vol_window: int = 10,
        streams: Optional[NoiseStreams] = None,
    ) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.streams = (
            streams if streams is not None else make_streams(self.rng, method=config.normal_method)
        )
        self.process = make_price_process(config)
        self.vol_window = vol_window
        self.horizon = 1
        self.paths: Optional[MarketPaths] = None
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, horizon: int = 1) -> LOBSnapshot:
        """Reset the market to its initial state and return the first snapshot."""
        cfg = self.config
        self.horizon = max(int(horizon), 1)
        self.t = 0
        self.shift = 0.0  # cumulative permanent impact (price units)
        self._intrastep_dev = 0.0  # transient deviation inside the current step
        self._returns: deque[float] = deque(
            [cfg.volatility] * self.vol_window, maxlen=self.vol_window
        )

        # draw the whole exogenous episode before the agent acts (+2 points of headroom so
        # a caller that advances past the horizon still indexes a real value)
        self.paths = generate_market_paths(cfg, self.horizon + 2, self.streams, self.process)
        self._sync()
        return self.snapshot

    # ------------------------------------------------------------------ helpers
    def _index(self) -> int:
        assert self.paths is not None
        return min(self.t, self.paths.n_points - 1)

    def _sync(self) -> None:
        """Refresh every observable from the pre-drawn paths at the current step."""
        assert self.paths is not None
        p, i = self.paths, self._index()
        self.spread = float(p.spread[i])
        self.imbalance_factor = float(p.imbalance[i])
        self.sigma = p.sigma_at(i)
        self.market_volume = float(p.volume[i])
        self._base_mid = float(p.mid[i])
        self._sync_mid()

    def _sync_mid(self) -> None:
        """Recompute the mid from the exogenous base plus deviations, and re-quote the book."""
        self.mid = max(
            self._base_mid + self._intrastep_dev + self.shift, float(self.config.tick_size)
        )
        self.snapshot = self._make_snapshot()

    def _make_snapshot(self) -> LOBSnapshot:
        assert self.paths is not None
        i = self._index()
        return build_book(
            mid=self.mid,
            spread=self.spread,
            config=self.config,
            imbalance_factor=self.imbalance_factor,
            noise=(self.paths.depth_bid[i], self.paths.depth_ask[i]),
        )

    def recent_volatility(self) -> float:
        """Realised volatility estimate from the recent return window."""
        if len(self._returns) < 2:
            return float(self.config.volatility)
        return float(np.std(self._returns))

    # ------------------------------------------------------------------ execution
    def execute(self, side: Side, shares: float) -> ExecutionResult:
        """Fill ``shares`` against the current book and apply market impact."""
        cfg = self.config
        mid_before = self.mid
        fill = walk_book(self.snapshot, side, shares)
        filled = fill.filled_shares

        if filled <= 0:
            return ExecutionResult(
                filled_shares=0.0,
                avg_price=mid_before,
                raw_avg_price=mid_before,
                notional=0.0,
                mid_before=mid_before,
                mid_after=mid_before,
                slippage=0.0,
                temp_impact_cost=0.0,
                perm_impact=0.0,
            )

        # temporary impact: a transient per-share concession linear in size (=> convex cost)
        temp_per_share = cfg.temporary_impact * filled
        eff_price = fill.avg_price - side.sign * temp_per_share
        temp_impact_cost = abs(temp_per_share) * filled

        # permanent impact: the mid moves adversely and persists for future steps
        # impact is an additive overlay on the exogenous path, so it persists for the rest of
        # the episode without consuming any randomness
        self.shift -= side.sign * cfg.permanent_impact * filled
        self._sync_mid()
        mid_after = self.mid

        # signed slippage vs the pre-trade mid (positive = adverse to us)
        slippage = side.sign * (mid_before - eff_price) / mid_before

        return ExecutionResult(
            filled_shares=filled,
            avg_price=eff_price,
            raw_avg_price=fill.avg_price,
            notional=eff_price * filled,
            mid_before=mid_before,
            mid_after=mid_after,
            slippage=slippage,
            temp_impact_cost=temp_impact_cost,
            perm_impact=mid_after - mid_before,  # signed mid shift (adverse direction)
        )

    # ------------------------------------------------------------------ dynamics
    def advance(self) -> LOBSnapshot:
        """Move the market one step along its pre-drawn exogenous path (no agent impact)."""
        self.t += 1
        prev_mid = self.mid
        self._intrastep_dev = 0.0  # an intra-step latency deviation does not persist
        self._sync()
        if prev_mid > 0:
            self._returns.append((self.mid - prev_mid) / prev_mid)
        return self.snapshot

    def apply_latency(self, latency_steps: float) -> float:
        """Move the mid to where it actually is when a delayed order reaches the book.

        The mid at the fractional arrival time is drawn from the **Brownian bridge** between
        the current and the next exogenous price -- the correct conditional law of a Brownian
        path at an interior time given both endpoints.  It is unbiased: latency hurts through
        *variance* and adverse selection on the fill, not through an assumed drift.  The
        deviation is transient and is cleared by the next :meth:`advance`.

        Returns the (signed) price shift applied.
        """
        if latency_steps <= 0:
            return 0.0
        assert self.paths is not None
        i = self._index()
        nxt = min(i + 1, self.paths.n_points - 1)
        delta = float(np.clip(latency_steps, 0.0, 1.0))
        z = float(self.streams.normals(Stream.LATENCY))
        bridged = self.process.bridge(
            float(self.paths.mid[i]),
            float(self.paths.mid[nxt]),
            delta,
            self.paths.sigma_at(i),
            z,
        )
        before = self.mid
        self._intrastep_dev = float(bridged - self._base_mid)
        self._sync_mid()
        return float(self.mid - before)
