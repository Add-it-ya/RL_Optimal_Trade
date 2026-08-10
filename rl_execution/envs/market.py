"""Stochastic market simulator: price dynamics, depth/imbalance evolution and impact.

The simulator owns the mid-price process and re-generates a :class:`LOBSnapshot` each
step.  It exposes two operations to the environment:

* :meth:`execute` -- fill a market order against the current book, applying temporary
  (transient) and permanent (persistent) market impact.
* :meth:`advance` -- evolve the market one step (drift / volatility / mean reversion /
  imbalance), refreshing the book and the ambient traded volume.

Two dynamics modes are supported, selected by ``MarketConfig.price_process``.

``'legacy'`` (the default) keeps the original inline update -- a multiplicative random walk
whose noise is drawn step by step from a single shared generator -- so every previously
published result reproduces bit-for-bit.

Any other value delegates to an explicit Brownian process from
:mod:`rl_execution.envs.processes`.  The entire exogenous market (price, volatility,
imbalance, spread, ambient volume, per-level depth noise) is then drawn **up front** at
:meth:`reset`, and market impact is applied as an additive ``shift`` on top of it:

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
    """Synthetic market with a regenerating limit order book.

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
        one is derived from ``rng`` according to ``config.rng_mode``.
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
        self.legacy = str(config.price_process) == "legacy"
        self.streams = (
            streams
            if streams is not None
            else make_streams(
                self.rng, config.rng_mode, method=getattr(config, "normal_method", "ziggurat")
            )
        )
        self.process = None if self.legacy else make_price_process(config)
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

        if self.legacy:
            self.mid = float(cfg.initial_price)
            self.spread = float(cfg.base_spread)
            self.imbalance_factor = 0.0
            self.sigma = float(cfg.volatility)
            self.market_volume = self._sample_volume()
            self.snapshot = self._make_snapshot()
            return self.snapshot

        # draw the whole exogenous episode before the agent acts (+2 points of headroom so
        # a caller that advances past the horizon still indexes a real value)
        self.paths = generate_market_paths(cfg, self.horizon + 2, self.streams, self.process)
        self._sync()
        return self.snapshot

    # ------------------------------------------------------------------ helpers
    def _sample_volume(self) -> float:
        """Ambient market volume for the current step (U-shaped intraday profile)."""
        cfg = self.config
        frac = self.t / max(self.horizon - 1, 1)
        # U-shape: high near the open (0) and close (1), lower mid-session.
        u = 1.0 + cfg.volume_u_shape * (2.0 * (frac - 0.5)) ** 2
        noise = np.exp(self.rng.normal(0.0, cfg.volume_vol))
        return float(max(cfg.base_market_volume * u * noise, 1.0))

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
        noise = None
        if not self.legacy:
            assert self.paths is not None
            i = self._index()
            noise = (self.paths.depth_bid[i], self.paths.depth_ask[i])
        return build_book(
            mid=self.mid,
            spread=self.spread,
            config=self.config,
            imbalance_factor=self.imbalance_factor,
            rng=self.rng,
            noise=noise,
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
        perm_amount = cfg.permanent_impact * filled
        if self.legacy:
            mid_after = mid_before - side.sign * perm_amount
            mid_after = max(mid_after, self.config.tick_size)
            self.mid = mid_after
            self.snapshot = self._make_snapshot()  # book re-centres on the new mid
        else:
            # impact is an additive overlay on the exogenous path, so it persists for the
            # rest of the episode without consuming any randomness
            self.shift -= side.sign * perm_amount
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
        """Evolve the market by one step (exogenous dynamics, no agent impact)."""
        cfg = self.config
        self.t += 1

        if not self.legacy:
            prev_mid = self.mid
            self._intrastep_dev = 0.0  # an intra-step latency deviation does not persist
            self._sync()
            if prev_mid > 0:
                self._returns.append((self.mid - prev_mid) / prev_mid)
            return self.snapshot

        # stochastic volatility (log-AR(0) shock around the base level)
        if cfg.vol_of_vol > 0:
            self.sigma = float(cfg.volatility * np.exp(self.rng.normal(0.0, cfg.vol_of_vol)))
        else:
            self.sigma = float(cfg.volatility)

        # mean-reversion pull toward the initial price (sideways regimes)
        reversion = cfg.mean_reversion * (cfg.initial_price - self.mid) / cfg.initial_price

        # imbalance has a weak predictive (alpha) effect on the next return
        alpha = cfg.imbalance_alpha * self.imbalance_factor

        shock = self.rng.normal(0.0, self.sigma)
        ret = cfg.drift + reversion + alpha + shock
        new_mid = max(self.mid * (1.0 + ret), cfg.tick_size)
        realised_ret = (new_mid - self.mid) / self.mid
        self.mid = new_mid
        self._returns.append(realised_ret)

        # evolve latent imbalance as a persistent AR(1) process
        self.imbalance_factor = float(
            cfg.imbalance_persistence * self.imbalance_factor
            + (1.0 - cfg.imbalance_persistence) * self.rng.normal(0.0, 1.0)
        )
        self.imbalance_factor = float(np.clip(self.imbalance_factor, -1.0, 1.0))

        # evolve the spread around its base level (log-noise)
        self.spread = float(
            max(cfg.base_spread * np.exp(self.rng.normal(0.0, cfg.spread_vol)), cfg.tick_size)
        )

        self.market_volume = self._sample_volume()
        self.snapshot = self._make_snapshot()
        return self.snapshot

    def apply_latency(self, side: Side, latency_steps: float) -> float:
        """Move the mid to where it actually is when a delayed order reaches the book.

        Two models, selected by ``MarketConfig.latency_model``:

        ``'adverse'`` (default, and always used in legacy mode)
            The shift is a folded normal charged against the trader, so latency is a pure
            cost.  This is an *adverse-selection* model rather than a price model: it says
            that by the time the order arrives the favourable side of the market has gone.

        ``'bridge'``
            The mid at the fractional arrival time is drawn from the **Brownian bridge**
            between the current and the next exogenous price -- the correct conditional law
            of a Brownian path at an interior time given both endpoints.  It is unbiased:
            latency then hurts through *variance* and adverse selection on the fill, not
            through an assumed drift.  The deviation is transient and is cleared by the next
            :meth:`advance`.

        Returns the (signed) price shift applied.
        """
        if latency_steps <= 0:
            return 0.0

        if not self.legacy and self.config.latency_model == "bridge":
            return self._latency_bridge(latency_steps)

        drift = abs(self.config.drift) * latency_steps
        vol = self.sigma * np.sqrt(latency_steps) * abs(self.rng.normal())
        magnitude = self.mid * (drift + vol)
        # adverse direction: SELL -> mid falls, BUY -> mid rises
        self.mid = max(self.mid - side.sign * magnitude, self.config.tick_size)
        if self.legacy:
            self.snapshot = self._make_snapshot()
        else:
            self._intrastep_dev += self.mid - (self._base_mid + self.shift)
            self._sync_mid()
        return -side.sign * magnitude

    def _latency_bridge(self, latency_steps: float) -> float:
        """Brownian-bridge mid at a fractional delay inside the current step."""
        assert self.paths is not None and self.process is not None
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
