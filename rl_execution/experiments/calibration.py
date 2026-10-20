"""Calibration: check that the simulated market poses a real execution problem.

A useful simulator needs three things at once.  Impact has to be expensive enough that
dumping the order is a bad idea, price risk has to be large enough that waiting is too, and
the predictive signal has to be about as weak as real ones, or any "edge" is an artefact.
This module measures each of those, and solves for the two free knobs:

* :func:`solve_alpha` -- the ``imbalance_alpha`` that gives a target information
  coefficient (IC) between the *observed* book imbalance and the next step's return;
* :func:`solve_risk_aversion` -- the reward's ``risk_aversion`` that gives a target
  Almgren-Chriss urgency ``kappa * T``, i.e. how front-loaded the optimal schedule is.

:data:`TARGETS` lists the acceptance bands that ``tests/test_calibration.py`` pins for the
default configuration and ``scripts/calibrate.py`` reports against.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np

from rl_execution.config import ExecutionConfig, MarketConfig, RewardConfig, ac_risk_aversion
from rl_execution.envs.brownian import NoiseStreams
from rl_execution.envs.market import MarketSimulator

#: Acceptance bands for the default market and order, as ``(low, high)``.
TARGETS: Dict[str, Tuple[float, float]] = {
    "dump_cost_bps": (100.0, 200.0),  # selling everything at arrival must hurt
    "twap_cost_bps": (10.0, 30.0),  # spreading it out must be much cheaper ...
    "twap_is_std_bps": (80.0, 130.0),  # ... but carry real price risk
    "ac_first_trade_vs_twap": (2.0, 3.5),  # the optimum is visibly front-loaded ...
    "ac_l1_vs_twap": (0.20, 1.0),  # ... so recovering it is distinguishable from TWAP
    "signal_ic": (0.05, 0.10),  # a realistic order-book signal
}

#: Seeds used for calibration measurements (disjoint from training, validation and test).
CALIBRATION_BASE_SEED = 3_000_000


# --------------------------------------------------------------------------- signal
def measure_signal(
    market: MarketConfig, n_episodes: int = 2000, horizon: int = 20, seed: int = 0
) -> Dict[str, float]:
    """Information coefficient of the imbalance with the next step's return.

    ``ic_observed`` uses the book imbalance an agent actually sees; ``ic_latent`` uses the
    hidden factor that drives the alpha.  ``r2`` is the per-step predictive R^2 of the
    observed feature.  No trading, so no impact: this is the exogenous market alone.
    The imbalance is persistent, so fewer than ~2,000 episodes give a noisy estimate.
    """
    observed, latent, returns = [], [], []
    for ep in range(n_episodes):
        sim = MarketSimulator(market, streams=NoiseStreams.from_seed(seed + ep))
        sim.reset(horizon=horizon)
        for _ in range(horizon):
            imb_obs, imb_lat, mid = sim.snapshot.imbalance, sim.imbalance_factor, sim.mid
            sim.advance()
            observed.append(imb_obs)
            latent.append(imb_lat)
            returns.append(sim.mid / mid - 1.0)
    r = np.asarray(returns)
    ic_obs = float(np.corrcoef(observed, r)[0, 1])
    return {
        "ic_observed": ic_obs,
        "ic_latent": float(np.corrcoef(latent, r)[0, 1]),
        "r2": ic_obs**2,
    }


def solve_alpha(
    market: MarketConfig,
    target_ic: float,
    n_episodes: int = 2000,
    seed: int = 0,
    upper: float = 0.01,
    iters: int = 30,
) -> float:
    """``imbalance_alpha`` giving the target observed IC (bisection on common seeds)."""
    lo, hi = 0.0, float(upper)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ic = measure_signal(replace(market, imbalance_alpha=mid), n_episodes, seed=seed)
        lo, hi = (mid, hi) if ic["ic_observed"] < target_ic else (lo, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- schedule shape
def ac_schedule(market: MarketConfig, execution: ExecutionConfig) -> Dict[str, float]:
    """Shape of the Almgren-Chriss schedule implied by the reward's risk aversion."""
    from rl_execution.baselines.almgren_chriss import optimal_holdings

    n = int(execution.horizon)
    lam = ac_risk_aversion(market, execution)
    holdings, kappa = optimal_holdings(market, n, execution.total_inventory, lam)
    trades = (holdings[:-1] - holdings[1:]) / execution.total_inventory
    return {
        "kappa_T": kappa * n * float(market.dt),
        "ac_first_trade_vs_twap": float(trades[0] * n),
        "ac_l1_vs_twap": float(np.abs(trades - 1.0 / n).sum()),
    }


def solve_risk_aversion(
    market: MarketConfig, execution: ExecutionConfig, target_kappa_t: float, iters: int = 60
) -> float:
    """Reward ``risk_aversion`` (per bps^2) giving the target AC urgency ``kappa * T``."""
    lo, hi = -12.0, 2.0  # bisection in log10(lambda)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ec = replace(execution, reward=replace(execution.reward, risk_aversion=10**mid))
        lo, hi = (mid, hi) if ac_schedule(market, ec)["kappa_T"] < target_kappa_t else (lo, mid)
    return float(10 ** (0.5 * (lo + hi)))


# --------------------------------------------------------------------------- costs
def cost_profile(
    market: MarketConfig,
    execution: ExecutionConfig,
    n_episodes: int = 400,
    base_seed: int = CALIBRATION_BASE_SEED,
):
    """Mean IS, IS spread and objective of the reference schedules (antithetic pairs)."""
    from rl_execution.backtest import compare_strategies, results_table
    from rl_execution.baselines import TWAP, AlmgrenChriss, DumpAtStart, WaitThenDump
    from rl_execution.envs import ExecutionEnv

    strategies = {
        "TWAP": TWAP(),
        "AlmgrenChriss": AlmgrenChriss(),
        "DumpAtStart": DumpAtStart(),
        "WaitThenDump": WaitThenDump(),
    }
    results = compare_strategies(
        lambda: ExecutionEnv(market, execution),
        strategies,
        n_episodes=n_episodes,
        base_seed=base_seed,
        progress=False,
        antithetic=True,
    )
    return results_table(results)


def calibration_report(
    market: Optional[MarketConfig] = None,
    execution: Optional[ExecutionConfig] = None,
    n_episodes: int = 400,
) -> Dict[str, float]:
    """Every quantity in :data:`TARGETS`, measured for ``market`` and ``execution``."""
    market = market or MarketConfig()
    execution = execution or ExecutionConfig(reward=RewardConfig())
    costs = cost_profile(market, execution, n_episodes=n_episodes)
    report = {
        "dump_cost_bps": float(costs.loc["DumpAtStart", "IS_bps"]),
        "twap_cost_bps": float(costs.loc["TWAP", "IS_bps"]),
        "twap_is_std_bps": float(costs.loc["TWAP", "IS_std"]),
        "signal_ic": measure_signal(market, horizon=execution.horizon)["ic_observed"],
    }
    report.update(ac_schedule(market, execution))
    for name in costs.index:
        report[f"objective_{name}"] = float(costs.loc[name, "Objective_bps"])
    return report
