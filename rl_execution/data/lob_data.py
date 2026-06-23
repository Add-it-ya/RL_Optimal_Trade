"""Real / historical limit-order-book data support.

The framework can replay recorded LOB data instead of the synthetic market.  Data is
normalised to a *standard* tabular schema (one row per snapshot)::

    mid, spread, volume, bid_px_1..L, bid_sz_1..L, ask_px_1..L, ask_sz_1..L

Loaders are provided for the common `LOBSTER <https://lobsterdata.com>`_ format and for
generic CSVs already in the standard schema, plus a synthetic generator so the pipeline is
runnable without proprietary data.  :class:`HistoricalMarketSource` plugs into
:class:`~rl_execution.envs.ExecutionEnv` (via its ``market_source`` argument) and replays
random windows of the data, overlaying the configured market-impact model on top of the
historical prices.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from rl_execution.config import MarketConfig, Side
from rl_execution.envs.lob import LOBSnapshot
from rl_execution.envs.market import ExecutionResult


def standard_columns(levels: int) -> List[str]:
    cols = ["mid", "spread", "volume"]
    for i in range(1, levels + 1):
        cols += [f"bid_px_{i}", f"bid_sz_{i}", f"ask_px_{i}", f"ask_sz_{i}"]
    return cols


STANDARD_COLUMNS = standard_columns  # callable alias


# --------------------------------------------------------------------------- loaders
def synthetic_lob_dataframe(
    n_steps: int = 5_000,
    config: Optional[MarketConfig] = None,
    levels: int = 5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a synthetic LOB dataset in the standard schema.

    Produced by running the :class:`~rl_execution.envs.market.MarketSimulator` forward and
    recording snapshots -- a convenient stand-in for real data when validating the
    historical-replay pipeline.
    """
    from rl_execution.envs.market import MarketSimulator

    config = config or MarketConfig()
    sim = MarketSimulator(config, rng=np.random.default_rng(seed))
    sim.reset(horizon=n_steps)
    rows = []
    for _ in range(n_steps):
        snap = sim.snapshot
        L = min(levels, len(snap.bid_prices))
        row = {"mid": snap.mid, "spread": snap.spread, "volume": sim.market_volume}
        for i in range(L):
            row[f"bid_px_{i+1}"] = snap.bid_prices[i]
            row[f"bid_sz_{i+1}"] = snap.bid_sizes[i]
            row[f"ask_px_{i+1}"] = snap.ask_prices[i]
            row[f"ask_sz_{i+1}"] = snap.ask_sizes[i]
        rows.append(row)
        sim.advance()
    return pd.DataFrame(rows, columns=standard_columns(levels))


def load_lob_csv(path: str) -> pd.DataFrame:
    """Load a CSV already in (a superset of) the standard schema."""
    df = pd.read_csv(path)
    if "mid" not in df.columns:
        if {"bid_px_1", "ask_px_1"}.issubset(df.columns):
            df["mid"] = 0.5 * (df["bid_px_1"] + df["ask_px_1"])
        else:
            raise ValueError("CSV must contain 'mid' or 'bid_px_1'/'ask_px_1' columns.")
    if "spread" not in df.columns and {"bid_px_1", "ask_px_1"}.issubset(df.columns):
        df["spread"] = df["ask_px_1"] - df["bid_px_1"]
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df


def load_lobster(
    orderbook_file: str,
    levels: int = 5,
    message_file: Optional[str] = None,
    price_scale: float = 1e4,
) -> pd.DataFrame:
    """Load a LOBSTER order-book file into the standard schema.

    LOBSTER order-book rows are ``[ask_px_1, ask_sz_1, bid_px_1, bid_sz_1, ask_px_2, ...]``
    with integer prices in units of ``price_scale`` (default 1e4 = $0.0001).  If a message
    file is supplied, per-snapshot traded volume is derived from execution events
    (event types 4 and 5); otherwise ``volume`` is left NaN (replay falls back to the
    config's ambient volume).
    """
    raw = pd.read_csv(orderbook_file, header=None)
    out = pd.DataFrame()
    out["mid"] = 0.5 * (raw[0] + raw[2]) / price_scale
    out["spread"] = (raw[0] - raw[2]) / price_scale
    for i in range(levels):
        out[f"ask_px_{i+1}"] = raw[4 * i + 0] / price_scale
        out[f"ask_sz_{i+1}"] = raw[4 * i + 1]
        out[f"bid_px_{i+1}"] = raw[4 * i + 2] / price_scale
        out[f"bid_sz_{i+1}"] = raw[4 * i + 3]

    if message_file is not None:
        msg = pd.read_csv(message_file, header=None)
        # columns: time, type, order_id, size, price, direction
        is_exec = msg[1].isin([4, 5])
        out["volume"] = (msg[3] * is_exec).to_numpy()[: len(out)]
    else:
        out["volume"] = np.nan
    return out[standard_columns(levels)]


# --------------------------------------------------------------------------- replay
class HistoricalMarketSource:
    """Wraps a standard-schema DataFrame and serves replay simulators.

    Pass an instance as ``market_source`` to :class:`~rl_execution.envs.ExecutionEnv`.
    """

    def __init__(
        self, df: pd.DataFrame, config: Optional[MarketConfig] = None, vol_window: int = 10
    ):
        self.config = config or MarketConfig()
        self.vol_window = vol_window
        self.levels = sum(c.startswith("bid_px_") for c in df.columns)
        if self.levels == 0:
            raise ValueError("DataFrame has no bid/ask ladder columns.")
        self.mid = df["mid"].to_numpy(dtype=float)
        self.spread = df["spread"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float) if "volume" in df else np.full(len(df), np.nan)
        self.volume = np.where(np.isnan(vol), self.config.base_market_volume, vol)
        L = self.levels
        self.bid_px = df[[f"bid_px_{i+1}" for i in range(L)]].to_numpy(dtype=float)
        self.bid_sz = df[[f"bid_sz_{i+1}" for i in range(L)]].to_numpy(dtype=float)
        self.ask_px = df[[f"ask_px_{i+1}" for i in range(L)]].to_numpy(dtype=float)
        self.ask_sz = df[[f"ask_sz_{i+1}" for i in range(L)]].to_numpy(dtype=float)
        self.n = len(df)

    def make_simulator(self, rng: np.random.Generator) -> "HistoricalMarketSimulator":
        return HistoricalMarketSimulator(self, rng, self.vol_window)


class HistoricalMarketSimulator:
    """Replays a random window of historical LOB data with an impact overlay.

    Implements the same interface as
    :class:`~rl_execution.envs.market.MarketSimulator` so it is a drop-in for the env.
    Permanent impact accumulates as a persistent shift added to all historical prices;
    temporary impact is a per-trade concession.
    """

    def __init__(
        self, source: HistoricalMarketSource, rng: np.random.Generator, vol_window: int = 10
    ):
        self.source = source
        self.config = source.config
        self.rng = rng
        self.vol_window = vol_window

    # ------------------------------------------------------------------ lifecycle
    def reset(self, horizon: int = 1) -> LOBSnapshot:
        self.horizon = max(int(horizon), 1)
        max_start = max(self.source.n - self.horizon - 1, 1)
        self.start = int(self.rng.integers(0, max_start))
        self.t = 0
        self.shift = 0.0
        self._refresh()
        return self.snapshot

    @property
    def idx(self) -> int:
        return min(self.start + self.t, self.source.n - 1)

    def _refresh(self) -> None:
        i = self.idx
        s = self.source
        self.mid = float(s.mid[i] + self.shift)
        self.market_volume = float(s.volume[i])
        self.snapshot = LOBSnapshot(
            bid_prices=s.bid_px[i] + self.shift,
            bid_sizes=s.bid_sz[i],
            ask_prices=s.ask_px[i] + self.shift,
            ask_sizes=s.ask_sz[i],
            mid=self.mid,
        )

    def recent_volatility(self) -> float:
        i = self.idx
        lo = max(self.start, i - self.vol_window)
        seg = self.source.mid[lo : i + 1]
        if len(seg) < 2:
            return float(self.config.volatility)
        rets = np.diff(seg) / seg[:-1]
        return float(np.std(rets)) if len(rets) > 0 else float(self.config.volatility)

    # ------------------------------------------------------------------ execution
    def execute(self, side: Side, shares: float) -> ExecutionResult:
        from rl_execution.envs.lob import walk_book

        cfg = self.config
        mid_before = self.mid
        fill = walk_book(self.snapshot, side, shares)
        filled = fill.filled_shares
        if filled <= 0:
            return ExecutionResult(
                0.0, mid_before, mid_before, 0.0, mid_before, mid_before, 0.0, 0.0, 0.0
            )

        temp_per_share = cfg.temporary_impact * filled
        eff_price = fill.avg_price - side.sign * temp_per_share
        temp_impact_cost = abs(temp_per_share) * filled

        perm_amount = cfg.permanent_impact * filled
        self.shift -= side.sign * perm_amount  # persists for the rest of the episode
        self._refresh()  # re-centre book on the impacted price
        mid_after = self.mid

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

    def advance(self) -> LOBSnapshot:
        self.t += 1
        self._refresh()
        return self.snapshot

    def apply_latency(self, side: Side, latency_steps: float) -> float:
        if latency_steps <= 0:
            return 0.0
        sigma = self.recent_volatility()
        magnitude = self.mid * sigma * np.sqrt(latency_steps) * abs(self.rng.normal())
        self.shift -= side.sign * magnitude
        self._refresh()
        return -side.sign * magnitude
