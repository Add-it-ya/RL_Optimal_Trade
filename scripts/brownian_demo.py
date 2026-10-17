#!/usr/bin/env python
"""Visual and numerical sanity checks for the Brownian-motion layer.

Produces four figures in ``results/brownian_demo/`` and prints the diagnostics behind them:

1. ``brownian_paths.png``  -- sample GBM price paths against the ``+/- sigma sqrt(t)`` envelope;
2. ``brownian_qq.png``     -- normal QQ-plot of the simulated returns;
3. ``brownian_vol_acf.png``-- autocorrelation of ``|returns|`` with and without clustering;
4. ``brownian_antithetic.png`` -- an antithetic pair of price paths, mirrored about the start.

Run with ``python scripts/brownian_demo.py [--paths 2000] [--steps 60]``.
"""

from __future__ import annotations

import argparse

import numpy as np

from rl_execution.config import MarketConfig
from rl_execution.envs.brownian import BrownianEngine, NoiseStreams, normals
from rl_execution.envs.processes import generate_market_paths, make_price_process
from rl_execution.utils.io import RESULTS_DIR, ensure_dir
from rl_execution.viz import plot_brownian_paths, plot_return_qq, plot_vol_acf, save_fig

OUT_DIR = RESULTS_DIR / "brownian_demo"


def _price_paths(cfg: MarketConfig, n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    """``n_paths`` independent exogenous price paths under ``cfg``."""
    process = make_price_process(cfg)
    rng = np.random.default_rng(seed)
    sigma = np.full(n_steps, cfg.volatility)
    zero = np.zeros(n_steps)
    return np.stack([process.path(normals(rng, n_steps), sigma, zero) for _ in range(n_paths)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=2_000, help="number of simulated paths")
    parser.add_argument("--steps", type=int, default=60, help="steps per path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(OUT_DIR)
    cfg = MarketConfig(price_process="gbm", volatility=0.02, drift=0.0)

    # ---------------------------------------------------------------- 1. paths + envelope
    paths = _price_paths(cfg, args.paths, args.steps, args.seed)
    terminal = paths[:, -1]
    print("=== Geometric Brownian motion ===")
    print(f"  paths                 : {args.paths} x {args.steps} steps")
    print(
        f"  E[S_T] / S_0          : {terminal.mean() / cfg.initial_price:.5f}  (1.0 = martingale)"
    )
    print(f"  std(S_T)              : {terminal.std():.4f}")
    # exact lognormal standard deviation, S0 * sqrt(exp(sigma^2 T) - 1) at zero drift
    var_t = cfg.volatility**2 * args.steps * cfg.dt
    print(f"  theoretical std       : {cfg.initial_price * np.sqrt(np.expm1(var_t)):.4f}")
    print(f"  min price over all    : {paths.min():.4f}  (GBM is positive by construction)")
    save_fig(
        plot_brownian_paths(paths, cfg.volatility, cfg.initial_price),
        f"{OUT_DIR}/brownian_paths.png",
    )

    # ---------------------------------------------------------------- 2. return normality
    log_ret = np.diff(np.log(paths), axis=1)
    expected = (cfg.drift - 0.5 * cfg.volatility**2) * cfg.dt
    print("\n=== Return distribution ===")
    print(f"  mean log-return       : {log_ret.mean():+.6f}")
    print(f"  Ito-corrected target  : {expected:+.6f}   (mu - sigma^2/2) dt")
    print(f"  skew / excess kurtosis: {_skew(log_ret):+.3f} / {_kurt(log_ret):+.3f}")
    save_fig(plot_return_qq(log_ret[:200]), f"{OUT_DIR}/brownian_qq.png")

    # ---------------------------------------------------------------- 3. vol clustering
    n = 20_000
    clustered = generate_market_paths(
        MarketConfig(
            price_process="gbm",
            vol_process="expou",
            vol_of_vol=0.5,
            vol_persistence=0.9,
        ),
        n,
        NoiseStreams.from_seed(args.seed + 1),
    )
    flat = generate_market_paths(
        MarketConfig(price_process="gbm", vol_process="iid_lognormal", vol_of_vol=0.5),
        n,
        NoiseStreams.from_seed(args.seed + 1),
    )
    print("\n=== Volatility clustering ===")
    print(f"  lag-1 ACF of |r|, exp-OU        : {_acf1(np.abs(clustered.returns())):+.4f}")
    print(f"  lag-1 ACF of |r|, i.i.d. lognorm: {_acf1(np.abs(flat.returns())):+.4f}")
    save_fig(
        plot_vol_acf(
            {"exp-OU (clustering)": clustered.returns(), "i.i.d. lognormal": flat.returns()}
        ),
        f"{OUT_DIR}/brownian_vol_acf.png",
    )

    # ---------------------------------------------------------------- 4. quadratic variation
    print("\n=== Quadratic variation (sum of squared increments -> T) ===")
    rng = np.random.default_rng(args.seed + 2)
    for steps in (25, 250, 2_500, 25_000):
        qv = BrownianEngine.draw(rng, steps, dt=1.0 / steps).quadratic_variation()
        print(f"  n = {steps:>6d} : QV = {qv:.5f}   |QV - T| = {abs(qv - 1.0):.5f}")

    # ---------------------------------------------------------------- 5. antithetic pair
    twin_cfg = MarketConfig(price_process="gbm", vol_process="none", volatility=0.02, drift=0.0)
    plain = generate_market_paths(twin_cfg, args.steps, NoiseStreams.from_seed(7))
    twin = generate_market_paths(twin_cfg, args.steps, NoiseStreams.from_seed(7, antithetic=True))
    residual = np.abs(
        np.diff(np.log(plain.mid))
        + np.diff(np.log(twin.mid))
        + twin_cfg.volatility**2 * twin_cfg.dt
    ).max()
    print("\n=== Antithetic variates ===")
    print(f"  max |r + r' + sigma^2 dt| : {residual:.3e}   (exact mirroring => ~0)")
    save_fig(
        plot_brownian_paths(
            np.stack([plain.mid, twin.mid]), twin_cfg.volatility, twin_cfg.initial_price
        ),
        f"{OUT_DIR}/brownian_antithetic.png",
    )

    print(f"\nFigures written to {OUT_DIR}")


def _acf1(x: np.ndarray) -> float:
    d = np.asarray(x, dtype=float).ravel()
    d = d - d.mean()
    return float((d[:-1] * d[1:]).mean() / d.var()) if d.var() > 0 else 0.0


def _skew(x: np.ndarray) -> float:
    d = np.asarray(x, dtype=float).ravel()
    return float(((d - d.mean()) ** 3).mean() / d.std() ** 3)


def _kurt(x: np.ndarray) -> float:
    d = np.asarray(x, dtype=float).ravel()
    return float(((d - d.mean()) ** 4).mean() / d.std() ** 4 - 3.0)


if __name__ == "__main__":
    main()
