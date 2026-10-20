#!/usr/bin/env python
"""Check the market calibration against its targets, and solve for the two free knobs.

    python scripts/calibrate.py                         # report the default market
    python scripts/calibrate.py --regime thin           # ... or any named regime
    python scripts/calibrate.py --solve-ic 0.075        # imbalance_alpha for a target IC
    python scripts/calibrate.py --solve-kappa-t 2.5     # risk_aversion for a target AC urgency

The targets (``rl_execution.experiments.calibration.TARGETS``) say that dumping the order
must be expensive, spreading it must carry real price risk, the Almgren-Chriss optimum must
be visibly front-loaded, and the order-book signal must be about as weak as a real one.
"""

from __future__ import annotations

import argparse

from rl_execution.config import ExecutionConfig, MarketConfig
from rl_execution.experiments.calibration import (
    TARGETS,
    calibration_report,
    solve_alpha,
    solve_risk_aversion,
)
from rl_execution.experiments.regimes import get_regime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--regime", default=None, help="named regime (default: MarketConfig())")
    parser.add_argument("--episodes", type=int, default=1000, help="antithetic cost episodes")
    parser.add_argument("--solve-ic", type=float, default=None, metavar="IC")
    parser.add_argument("--solve-kappa-t", type=float, default=None, metavar="KT")
    args = parser.parse_args()

    market = get_regime(args.regime) if args.regime else MarketConfig()
    execution = ExecutionConfig()

    if args.solve_ic is not None:
        alpha = solve_alpha(market, args.solve_ic)
        print(f"imbalance_alpha for an observed IC of {args.solve_ic}: {alpha:.3e}")
    if args.solve_kappa_t is not None:
        lam = solve_risk_aversion(market, execution, args.solve_kappa_t)
        print(f"risk_aversion for an AC kappa*T of {args.solve_kappa_t}: {lam:.3e}")

    report = calibration_report(market, execution, n_episodes=args.episodes)
    print(f"\nCalibration of {args.regime or 'the default market'}:")
    for key, (low, high) in TARGETS.items():
        ok = "ok  " if low <= report[key] <= high else "MISS"
        print(f"  {ok} {key:<24} {report[key]:9.3f}   target [{low}, {high}]")
    print(f"       {'AC kappa*T':<24} {report['kappa_T']:9.3f}")
    print("\nObjective (IS + risk penalty, bps) of the reference schedules:")
    for key in sorted(k for k in report if k.startswith("objective_")):
        print(f"  {key.removeprefix('objective_'):<16} {report[key]:8.2f}")


if __name__ == "__main__":
    main()
