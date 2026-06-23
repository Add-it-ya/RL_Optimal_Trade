"""Stochastic market simulator: price dynamics, depth/imbalance evolution and impact.

The simulator owns the mid-price process and re-generates a :class:`LOBSnapshot` each
step.  It exposes two operations to the environment:

* :meth:`execute` -- fill a market order against the current book, applying temporary
  (transient) and permanent (persistent) market impact.
* :meth:`advance` -- evolve the market one step (drift / volatility / mean reversion /
  imbalance), refreshing the book and the ambient traded volume.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from rl_execution.config import MarketConfig, Side
from rl_execution.envs.lob import LOBSnapshot, build_book, walk_book


@dataclass
class ExecutionResult:
    """Outcome of executing a market order, with cost decomposition."""

    filled_shares: float
    avg_price: float            # effective price actually paid/received (incl. temp impact)
    raw_avg_price: float        # price from walking the book only (excl. temp impact)
    notional: float             # effective cash exchanged (avg_price * filled_shares)
    mid_before: float           # mid at execution time (pre permanent impact)
    mid_after: float            # mid after permanent impact has been applied
    slippage: float             # signed cost vs mid_before, as a fraction (+ = adverse)
    temp_impact_cost: float     # cash cost of transient impact
    perm_impact: float          # signed mid shift caused by this order (price units)


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
    """

    def __init__(
        self,
        config: MarketConfig,
        rng: np.random.Generator | None = None,
        vol_window: int = 10,
    ) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.vol_window = vol_window
        self.horizon = 1
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, horizon: int = 1) -> LOBSnapshot:
        """Reset the market to its initial state and return the first snapshot."""
        cfg = self.config
        self.horizon = max(int(horizon), 1)
        self.t = 0
        self.mid = float(cfg.initial_price)
        self.spread = float(cfg.base_spread)
        self.imbalance_factor = 0.0
        self.sigma = float(cfg.volatility)
        self._returns: deque[float] = deque(
            [cfg.volatility] * self.vol_window, maxlen=self.vol_window
        )
        self.market_volume = self._sample_volume()
        self.snapshot = self._make_snapshot()
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

    def _make_snapshot(self) -> LOBSnapshot:
        return build_book(
            mid=self.mid,
            spread=self.spread,
            config=self.config,
            imbalance_factor=self.imbalance_factor,
            rng=self.rng,
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
        mid_after = mid_before - side.sign * perm_amount
        mid_after = max(mid_after, self.config.tick_size)
        self.mid = mid_after
        self.snapshot = self._make_snapshot()  # book re-centres on the new mid

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

        # stochastic volatility (log-AR(0) shock around the base level)
        if cfg.vol_of_vol > 0:
            self.sigma = float(
                cfg.volatility * np.exp(self.rng.normal(0.0, cfg.vol_of_vol))
            )
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
            max(cfg.base_spread * np.exp(self.rng.normal(0.0, cfg.spread_vol)),
                cfg.tick_size)
        )

        self.market_volume = self._sample_volume()
        self.snapshot = self._make_snapshot()
        return self.snapshot

    def apply_latency(self, side: Side, latency_steps: float) -> float:
        """Shift the mid adversely to model acting on stale information.

        The order is decided at the current mid but fills only after a fractional
        ``latency_steps`` delay during which the market drifts.  The shift is always
        charged against the trader (worse fill price), scaled by drift plus
        sqrt-time volatility.  Returns the (signed) price shift applied.
        """
        if latency_steps <= 0:
            return 0.0
        drift = abs(self.config.drift) * latency_steps
        vol = self.sigma * np.sqrt(latency_steps) * abs(self.rng.normal())
        magnitude = self.mid * (drift + vol)
        # adverse direction: SELL -> mid falls, BUY -> mid rises
        self.mid = max(self.mid - side.sign * magnitude, self.config.tick_size)
        self.snapshot = self._make_snapshot()
        return -side.sign * magnitude
