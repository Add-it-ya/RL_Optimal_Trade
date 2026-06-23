"""Execution-quality metrics.

All cost metrics are expressed in **basis points (bps) of the arrival notional** and signed
so that *positive = cost* (worse) and *negative = price improvement* (better), for both buy
and sell parent orders.

Definitions
-----------
* **Implementation shortfall (IS)** -- Perold's measure: the difference between the value of
  a costless "paper" execution at the arrival (decision) price and the realised execution,
  including commissions and the opportunity cost of any unexecuted shares.
* **Execution cost** -- price-only slippage of the average fill price vs the arrival price.
* **Market-impact cost** -- the cost of crossing the book: shares executed away from the
  contemporaneous mid (spread + temporary impact + book-walk), summed over fills.
* **Sharpe ratio** -- cross-episode risk-adjusted execution quality, mean(-IS) / std(-IS).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "implementation_shortfall_bps",
    "execution_cost_bps",
    "market_impact_bps",
    "avg_fill_price",
    "unexecuted_shares",
    "total_commission",
    "cum_reward",
]


def _sign(side: str) -> int:
    return 1 if str(side).lower() == "sell" else -1


def compute_episode_metrics(
    summary: Dict[str, Any], history: pd.DataFrame
) -> Dict[str, float]:
    """Compute per-episode metrics from an episode summary and step history."""
    side = summary["side"]
    sign = _sign(side)
    arrival = summary["arrival_price"]
    avg_fill = summary["avg_fill_price"]
    total = summary["total_inventory"]
    norm = arrival * total

    # price-only execution cost (slippage of the average fill vs arrival)
    execution_cost_bps = sign * (arrival - avg_fill) / arrival * 1e4

    # market-impact / slippage-from-mid cost, aggregated across fills
    if len(history) > 0:
        impact_cash = float(
            (sign * history["shares"] * (history["mid_before"] - history["exec_price"])).sum()
        )
        temp_cash = float(history["temp_impact_cost"].sum())
    else:
        impact_cash = 0.0
        temp_cash = 0.0
    market_impact_bps = impact_cash / norm * 1e4
    temp_impact_bps = temp_cash / norm * 1e4

    return {
        "implementation_shortfall_bps": float(summary["implementation_shortfall_bps"]),
        "execution_cost_bps": float(execution_cost_bps),
        "market_impact_bps": float(market_impact_bps),
        "temp_impact_bps": float(temp_impact_bps),
        "avg_fill_price": float(avg_fill),
        "arrival_price": float(arrival),
        "final_mid": float(summary["final_mid"]),
        "unexecuted_shares": float(summary["unexecuted_shares"]),
        "total_commission": float(summary["total_commission"]),
        "cum_reward": float(summary["cum_reward"]),
    }


def aggregate_metrics(episode_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate a list of per-episode metric dicts into summary statistics.

    Adds cross-episode Sharpe ratios for implementation shortfall (``is_sharpe``) and
    cumulative reward (``reward_sharpe``).  Higher Sharpe = more *consistently* good
    execution across episodes.
    """
    if not episode_metrics:
        return {}
    df = pd.DataFrame(episode_metrics)
    eps = 1e-9
    out: Dict[str, float] = {"n_episodes": float(len(df))}

    for col in df.columns:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=0))

    neg_is = -df["implementation_shortfall_bps"].to_numpy()
    out["is_sharpe"] = float(neg_is.mean() / (neg_is.std(ddof=0) + eps))
    rew = df["cum_reward"].to_numpy()
    out["reward_sharpe"] = float(rew.mean() / (rew.std(ddof=0) + eps))
    return out
