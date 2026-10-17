"""Price and volatility processes driven by Brownian motion.

Each process is an explicit SDE with an **exact** discretisation wherever one exists, so
there is no Euler discretisation bias to defend:

============  ==================================================  ===========================
Process       SDE                                                 Exact step
============  ==================================================  ===========================
``abm``       ``dS = mu S0 dt + sigma S0 dW``                      ``S + mu S0 dt + sigma S0 sqrt(dt) Z``
``gbm``       ``dS = mu S dt + sigma S dW``                        ``S exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)``
``ou``        ``dS = kappa (S0 - S) dt + sigma S0 dW``             mean-reverting, exact moments
``expou``     ``d ln v = kv (ln vbar - ln v) dt + xi dW_v``        stationary AR(1) in ``ln v``
============  ==================================================  ===========================

``mu`` (``MarketConfig.drift``) and ``sigma`` (``MarketConfig.volatility``) are *relative*
quantities per unit time; the arithmetic processes convert them with the reference price
``S0 = initial_price``, which is exactly the convention the Almgren-Chriss baseline assumes
(``sigma_abs = volatility * initial_price``).  Under ``abm`` with zero alpha and linear
impact the closed-form AC schedule is therefore genuinely optimal, which makes that regime
a ground-truth benchmark for the learned policy.

The whole episode is generated up front by :func:`generate_market_paths`: price, volatility,
imbalance, spread, ambient volume and the per-level depth noise.  Because these paths are
drawn before the agent acts, **nothing the agent does can perturb the market it faces** --
the property that makes paired (common-random-number) strategy comparisons valid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Type

import numpy as np

from rl_execution.config import MarketConfig
from rl_execution.envs.brownian import NoiseStreams, Stream, bridge_value, correlate

__all__ = [
    "PriceProcess",
    "ArithmeticBM",
    "GeometricBM",
    "OrnsteinUhlenbeck",
    "make_price_process",
    "volatility_path",
    "MarketPaths",
    "generate_market_paths",
    "PRICE_PROCESSES",
    "VOL_PROCESSES",
]


# --------------------------------------------------------------------------- price processes
class PriceProcess(ABC):
    """A price process that turns standard normals into an exogenous mid-price path."""

    name: str = "abstract"

    def __init__(self, config: MarketConfig) -> None:
        self.config = config

    # -- helpers --------------------------------------------------------------------
    @property
    def s0(self) -> float:
        return float(self.config.initial_price)

    @property
    def dt(self) -> float:
        return float(self.config.dt)

    @abstractmethod
    def path(self, z: np.ndarray, sigma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Return the mid-price path of length ``len(z) + 1`` starting at ``initial_price``.

        ``sigma`` is the per-step relative volatility in force during each increment and
        ``alpha`` the per-step relative drift contributed by the latent imbalance factor.
        """

    def step_variance(self, mid: float, sigma: float) -> float:
        """Variance of one step's price change, in price units (used for bridging)."""
        return float((sigma * self.s0) ** 2 * self.dt)

    def bridge(self, prev: float, nxt: float, delta: float, sigma: float, z: float) -> float:
        """Mid-price at fraction ``delta`` inside a step, conditional on both endpoints."""
        return bridge_value(prev, nxt, delta, self.step_variance(prev, sigma), z)


class ArithmeticBM(PriceProcess):
    """``dS = mu S0 dt + sigma S0 dW`` -- the Almgren-Chriss price model.

    Gaussian increments of constant absolute size.  A floor at one tick is applied for
    numerical safety; over realistic horizons it never binds (reaching zero from
    ``S0 = 100`` at ``sigma = 2%`` over 20 steps is an 11-sigma event).
    """

    name = "abm"

    def path(self, z: np.ndarray, sigma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        cfg = self.config
        drift = (cfg.drift + alpha) * self.s0 * self.dt
        shock = sigma * self.s0 * np.sqrt(self.dt) * z
        out = self.s0 + np.concatenate([[0.0], np.cumsum(drift + shock)])
        return np.maximum(out, cfg.tick_size)


class GeometricBM(PriceProcess):
    """``dS = mu S dt + sigma S dW`` -- exact log-Euler, positive by construction.

    The ``- sigma^2 / 2`` Ito correction is what makes ``E[S_T] = S_0 exp(mu T)``; without
    it (as in a naive ``S * (1 + r)`` update) the mean drifts upward even at zero drift.
    """

    name = "gbm"

    def path(self, z: np.ndarray, sigma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        cfg = self.config
        log_ret = (cfg.drift + alpha - 0.5 * sigma**2) * self.dt + sigma * np.sqrt(self.dt) * z
        return self.s0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))

    def step_variance(self, mid: float, sigma: float) -> float:
        return float((sigma * max(mid, self.config.tick_size)) ** 2 * self.dt)

    def bridge(self, prev: float, nxt: float, delta: float, sigma: float, z: float) -> float:
        # bridge in log space, where the process is an arithmetic Brownian motion
        tick = self.config.tick_size
        a, b = max(float(prev), tick), max(float(nxt), tick)
        return float(np.exp(bridge_value(np.log(a), np.log(b), delta, sigma**2 * self.dt, z)))


class OrnsteinUhlenbeck(PriceProcess):
    """``dS = kappa (S0 - S) dt + sigma S0 dW`` -- exact OU discretisation.

    The exact transition over ``dt`` is
    ``S' = S0 + (S - S0) e^{-kappa dt} + sigma_abs sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) Z``,
    which preserves the stationary variance ``sigma_abs^2 / (2 kappa)`` at any step size --
    unlike the Euler pull term it replaces.  Falls back to :class:`ArithmeticBM` at
    ``kappa = 0``.
    """

    name = "ou"

    def path(self, z: np.ndarray, sigma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        cfg = self.config
        kappa = float(cfg.mean_reversion)
        if kappa <= 0:
            return ArithmeticBM(cfg).path(z, sigma, alpha)

        decay = float(np.exp(-kappa * self.dt))
        # exact conditional std for unit sigma; scales linearly in sigma_abs
        unit_std = float(np.sqrt((1.0 - decay**2) / (2.0 * kappa)))
        theta = self.s0
        out = np.empty(len(z) + 1)
        out[0] = theta
        for t, zt in enumerate(z):
            sigma_abs = float(sigma[t]) * self.s0
            drift = (cfg.drift + float(alpha[t])) * self.s0 * self.dt
            out[t + 1] = theta + (out[t] - theta) * decay + sigma_abs * unit_std * zt + drift
        return np.maximum(out, cfg.tick_size)


PRICE_PROCESSES: Dict[str, Type[PriceProcess]] = {
    "abm": ArithmeticBM,
    "gbm": GeometricBM,
    "ou": OrnsteinUhlenbeck,
}


def make_price_process(config: MarketConfig) -> PriceProcess:
    """Instantiate the process named by ``config.price_process``."""
    name = str(config.price_process)
    if name not in PRICE_PROCESSES:
        raise ValueError(
            f"Unknown price_process '{name}'. Expected one of {sorted(PRICE_PROCESSES)}."
        )
    return PRICE_PROCESSES[name](config)


# --------------------------------------------------------------------------- volatility
VOL_PROCESSES = ("none", "iid_lognormal", "expou")


def volatility_path(config: MarketConfig, z_vol: np.ndarray) -> np.ndarray:
    """Per-step relative volatility, one value per increment.

    * ``none``          -- constant ``config.volatility``.
    * ``iid_lognormal`` -- ``sigma_bar exp(xi Z)``, i.i.d.; no clustering.
    * ``expou``         -- exponential Ornstein-Uhlenbeck, i.e. a stationary AR(1) in
      ``ln sigma`` with per-step persistence ``phi = vol_persistence ** dt`` and innovation
      ``xi sqrt(1 - phi^2)``.  The innovation scaling is what keeps the stationary standard
      deviation of ``ln sigma`` equal to ``xi`` for every ``phi``, so turning clustering on
      does not silently change the average volatility level.

    ``expou`` is what produces **volatility clustering**, which is what makes the simulator's
    ``volatility`` observation feature carry forecast value at all.
    """
    n = int(np.asarray(z_vol).size)
    sigma_bar = float(config.volatility)
    xi = float(config.vol_of_vol)
    mode = str(config.vol_process)

    if mode == "none" or xi <= 0:
        return np.full(n, sigma_bar)
    if mode == "iid_lognormal":
        return sigma_bar * np.exp(xi * np.asarray(z_vol, dtype=float))
    if mode != "expou":
        raise ValueError(f"Unknown vol_process '{mode}'. Expected one of {VOL_PROCESSES}.")

    phi = float(np.clip(config.vol_persistence, 0.0, 0.999999)) ** float(config.dt)
    inn = xi * float(np.sqrt(1.0 - phi**2))
    y = np.empty(n)
    y[0] = xi * float(z_vol[0])  # start from the stationary distribution (no burn-in)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + inn * float(z_vol[t])
    return sigma_bar * np.exp(y)


def _imbalance_path(config: MarketConfig, z: np.ndarray) -> np.ndarray:
    """Latent order-book imbalance as a discrete Ornstein-Uhlenbeck (AR(1)) process.

    ``imbalance_persistence`` is the per-unit-time autocorrelation, i.e. ``e^{-kappa}``, so
    the per-step coefficient is ``phi = persistence ** dt``.

    Two innovation scalings are available:

    * ``sqrt(1 - phi^2)`` (``imbalance_ou_scaling=True``) -- the stationary scaling, which
      gives the latent factor unit variance at any ``phi``;
    * ``1 - phi`` (default) -- the original scaling, whose stationary standard deviation is
      only ``sqrt((1 - phi) / (1 + phi))`` (about 0.28 at ``phi = 0.85``), i.e. the realised
      skew and alpha are several times weaker than the configured values.

    The factor is then clipped to ``[-1, 1]``, which is a real constraint on the stationary
    version: a unit-variance Gaussian saturates about 32% of the time, so the *observed*
    standard deviation is about 0.72 rather than 1.0.  That is still 2.5x the historical
    scaling, so switching this on materially strengthens both the book skew and the alpha.
    """
    n = int(np.asarray(z).size)
    phi = float(np.clip(config.imbalance_persistence, 0.0, 0.999999)) ** float(config.dt)
    stationary = bool(config.imbalance_ou_scaling)
    inn = float(np.sqrt(1.0 - phi**2)) if stationary else (1.0 - phi)

    out = np.empty(n + 1)
    # a stationary process starts from its stationary law; the original one starts at zero
    out[0] = float(z[0]) if stationary else 0.0
    for t in range(n):
        out[t + 1] = phi * out[t] + inn * float(z[t])
    return np.clip(out, -1.0, 1.0)


# --------------------------------------------------------------------------- full market
@dataclass
class MarketPaths:
    """Every exogenous path of one episode, generated before the agent acts.

    All arrays are indexed by decision step.  ``mid`` and the ladder inputs carry
    ``n_points`` entries; ``sigma`` carries one value per increment.
    """

    mid: np.ndarray  # (n_points,) exogenous mid-price, before any market impact
    sigma: np.ndarray  # (n_points - 1,) relative volatility in force during each step
    imbalance: np.ndarray  # (n_points,) latent imbalance factor in [-1, 1]
    spread: np.ndarray  # (n_points,) absolute bid-ask spread
    volume: np.ndarray  # (n_points,) ambient traded volume
    depth_bid: np.ndarray  # (n_points, n_levels) multiplicative depth noise
    depth_ask: np.ndarray  # (n_points, n_levels)

    @property
    def n_points(self) -> int:
        return int(self.mid.size)

    def returns(self) -> np.ndarray:
        """Realised per-step relative returns of the exogenous path."""
        return np.diff(self.mid) / self.mid[:-1]

    def sigma_at(self, t: int) -> float:
        """Volatility in force during step ``t`` (clamped at the end of the path)."""
        return float(self.sigma[min(int(t), self.sigma.size - 1)])


def generate_market_paths(
    config: MarketConfig,
    n_points: int,
    streams: NoiseStreams,
    process: Optional[PriceProcess] = None,
) -> MarketPaths:
    """Draw the complete exogenous market for one episode.

    Draw order by stream (each stream is independent, so this order is not load-bearing):

    1. ``PRICE``     -- ``n`` standard normals driving the price increments;
    2. ``VOL``       -- ``n + 1`` normals combined with the *previous* step's price shock by
       ``leverage_corr`` to produce the leverage effect (vol rises after the price falls);
    3. ``IMBALANCE`` -- ``n`` normals driving the AR(1) imbalance factor;
    4. ``SPREAD``    -- ``n_points`` lognormal shocks on the spread;
    5. ``VOLUME``    -- ``n_points`` lognormal shocks on ambient volume;
    6. ``DEPTH``     -- ``(n_points, 2, n_levels)`` lognormal shocks on resting size.

    The volatility shock is deliberately driven by the *lagged* price shock: the leverage
    effect is the statement that a fall in price is followed by a rise in volatility, so the
    innovation entering ``sigma_{t+1}`` is what must correlate with the return over step
    ``t``.  Correlating them contemporaneously instead would make the volatility *level*
    covary with the same step's return and flip the sign of the measured effect.
    """
    n_points = max(int(n_points), 2)
    n = n_points - 1
    process = process or make_price_process(config)

    # --- price shocks, and volatility shocks correlated with the *previous* price shock ---
    z_price = streams.normals(Stream.PRICE, n)
    z_indep = streams.normals(Stream.VOL, n + 1)
    z_lagged_price = np.concatenate([z_indep[n : n + 1], z_price[:-1]])
    z_vol = correlate(z_lagged_price, z_indep[:n], float(config.leverage_corr))
    sigma = volatility_path(config, z_vol)

    # --- latent imbalance and the alpha it contributes to the next return ---
    z_imb = streams.normals(Stream.IMBALANCE, n)
    imbalance = _imbalance_path(config, z_imb)
    alpha = float(config.imbalance_alpha) * imbalance[:n]

    mid = process.path(z_price, sigma, alpha)

    # --- spread: lognormal noise around the base level, floored at one tick ---
    spread = np.maximum(
        config.base_spread * streams.lognormals(Stream.SPREAD, config.spread_vol, n_points),
        config.tick_size,
    )
    spread[0] = float(config.base_spread)  # the book opens at its baseline spread

    # --- ambient volume: U-shaped intraday profile times lognormal noise ---
    frac = np.arange(n_points, dtype=float) / max(n_points - 2, 1)
    u_shape = 1.0 + config.volume_u_shape * (2.0 * (frac - 0.5)) ** 2
    volume = np.maximum(
        config.base_market_volume
        * u_shape
        * streams.lognormals(Stream.VOLUME, config.volume_vol, n_points),
        1.0,
    )

    # --- per-level resting-depth noise, drawn once so re-quoting the book after a trade
    #     reuses the same liquidity realisation instead of re-randomising it ---
    depth = streams.lognormals(Stream.DEPTH, config.depth_vol, (n_points, 2, config.n_levels))

    return MarketPaths(
        mid=mid,
        sigma=sigma,
        imbalance=imbalance,
        spread=spread,
        volume=volume,
        depth_bid=depth[:, 0, :],
        depth_ask=depth[:, 1, :],
    )
