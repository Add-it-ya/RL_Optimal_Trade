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
* **Objective** -- what the agents are rewarded on: IS plus the Almgren-Chriss variance
  penalty ``lambda * sum_t (sigma_t x_t)^2`` for the inventory carried.
* **Effective steps traded** -- ``1 / sum_t p_t^2`` for the fraction ``p_t`` of the order
  traded at step ``t``: equal to the horizon for TWAP and 1 for a single block trade, so a
  degenerate "dump everything at once" policy is visible at a glance.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "objective_bps",
    "implementation_shortfall_bps",
    "risk_penalty_bps",
    "effective_steps",
    "execution_cost_bps",
    "market_impact_bps",
    "avg_fill_price",
    "unexecuted_shares",
    "total_commission",
    "cum_reward",
]


def _sign(side: str) -> int:
    return 1 if str(side).lower() == "sell" else -1


def compute_episode_metrics(summary: Dict[str, Any], history: pd.DataFrame) -> Dict[str, float]:
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

    traded = history["shares"].to_numpy(dtype=float) / total if len(history) > 0 else np.zeros(0)
    concentration = float(np.sum(traded**2))
    effective_steps = 1.0 / concentration if concentration > 0 else 0.0

    is_bps = float(summary["implementation_shortfall_bps"])
    risk_bps = float(summary.get("risk_penalty_bps", 0.0))
    return {
        "objective_bps": float(summary.get("objective_bps", is_bps + risk_bps)),
        "implementation_shortfall_bps": is_bps,
        "risk_penalty_bps": risk_bps,
        "effective_steps": effective_steps,
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
    """Aggregate per-episode metric dicts into ``<metric>_mean`` / ``<metric>_std``."""
    if not episode_metrics:
        return {}
    df = pd.DataFrame(episode_metrics)
    out: Dict[str, float] = {"n_episodes": float(len(df))}
    for col in df.columns:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=0))
    return out
