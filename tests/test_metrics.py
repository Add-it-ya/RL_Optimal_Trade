import numpy as np
import pandas as pd
import pytest

from rl_execution.metrics import aggregate_metrics, compute_episode_metrics


def _summary(side="sell", arrival=100.0, avg_fill=99.5, total=10_000, commission=10.0):
    realised = avg_fill * total
    paper = arrival * total
    is_cash = (paper - realised + commission) if side == "sell" else (realised - paper + commission)
    return {
        "side": side,
        "total_inventory": total,
        "arrival_price": arrival,
        "avg_fill_price": avg_fill,
        "final_mid": arrival,
        "executed_shares": total,
        "unexecuted_shares": 0.0,
        "executed_notional": realised,
        "total_commission": commission,
        "implementation_shortfall_bps": is_cash / paper * 1e4,
        "implementation_shortfall_cash": is_cash,
        "cum_reward": -is_cash / paper * 1e4,
    }


def _history(total=10_000, arrival=100.0, exec_price=99.5, steps=5):
    per = total / steps
    return pd.DataFrame(
        {
            "shares": [per] * steps,
            "mid_before": [arrival] * steps,
            "exec_price": [exec_price] * steps,
            "temp_impact_cost": [1.0] * steps,
        }
    )


def test_sell_cost_is_positive_when_filled_below_arrival():
    m = compute_episode_metrics(_summary("sell", avg_fill=99.5), _history(exec_price=99.5))
    assert m["execution_cost_bps"] > 0  # sold cheap -> a cost
    assert m["market_impact_bps"] > 0


def test_buy_cost_sign_convention():
    # buying above arrival is a cost (positive)
    m = compute_episode_metrics(_summary("buy", avg_fill=100.5), _history(exec_price=100.5))
    assert m["execution_cost_bps"] > 0


def test_price_improvement_is_negative_cost():
    m = compute_episode_metrics(_summary("sell", avg_fill=100.5), _history(exec_price=100.5))
    assert m["execution_cost_bps"] < 0  # sold above arrival -> improvement


def test_aggregate_reports_means_and_spreads():
    eps = [
        compute_episode_metrics(_summary(avg_fill=f), _history(exec_price=f))
        for f in [99.4, 99.5, 99.6]
    ]
    agg = aggregate_metrics(eps)
    assert agg["n_episodes"] == 3
    assert np.isfinite(agg["implementation_shortfall_bps_mean"])
    assert agg["implementation_shortfall_bps_std"] > 0


def test_effective_steps_traded():
    """TWAP over n steps counts as n steps; one block trade counts as 1."""
    twap = compute_episode_metrics(_summary(), _history(steps=5))
    assert twap["effective_steps"] == pytest.approx(5.0)
    block = pd.DataFrame(
        {
            "shares": [10_000.0],
            "mid_before": [100.0],
            "exec_price": [99.5],
            "temp_impact_cost": [0.0],
        }
    )
    assert compute_episode_metrics(_summary(), block)["effective_steps"] == pytest.approx(1.0)


def test_objective_defaults_to_shortfall_without_a_risk_penalty():
    m = compute_episode_metrics(_summary(), _history())
    assert m["objective_bps"] == pytest.approx(m["implementation_shortfall_bps"])
