"""Market-regime presets for stress-testing execution strategies.

Each regime is a :class:`~rl_execution.config.MarketConfig`.  Regimes are grouped into:

* **volatility**  -- low / medium / high realised volatility,
* **liquidity**   -- thin / normal / deep order books,
* **trend**       -- bull (drift up) / bear (drift down) / sideways (mean-reverting),
* **brownian**    -- alternative price models and volatility dynamics.

:func:`randomized_market_config` samples a config from across these axes and is used for
*domain-randomised* training so a single agent generalises to unseen conditions.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np

from rl_execution.config import MarketConfig

# A modest, exploitable order-book-imbalance alpha is present across regimes so that a
# *state-reactive* policy can add value over static schedules; drift magnitudes are kept
# small relative to per-step volatility so that timing skill (not trivial front-loading)
# is what differentiates strategies.  The price follows arithmetic Brownian motion (the
# Almgren-Chriss world) unless a regime says otherwise.
_BASE = MarketConfig(imbalance_alpha=0.05, imbalance_strength=0.40, imbalance_persistence=0.85)

REGIMES: Dict[str, MarketConfig] = {
    # --- volatility ---
    "low_vol": replace(_BASE, volatility=0.008, vol_of_vol=0.0),
    "medium_vol": replace(_BASE, volatility=0.02, vol_of_vol=0.2),
    "high_vol": replace(_BASE, volatility=0.05, vol_of_vol=0.5),
    # --- liquidity ---
    "thin": replace(
        _BASE,
        base_depth=200.0,
        base_spread=0.05,
        base_market_volume=800.0,
        temporary_impact=3.0e-5,
        permanent_impact=1.2e-5,
    ),
    "normal_liquidity": replace(_BASE),
    "deep": replace(
        _BASE,
        base_depth=1500.0,
        base_spread=0.01,
        base_market_volume=6000.0,
        temporary_impact=4.0e-6,
        permanent_impact=2.0e-6,
    ),
    # --- trend (drift small vs sqrt-horizon volatility) ---
    "bull": replace(_BASE, drift=0.0008),
    "bear": replace(_BASE, drift=-0.0008),
    "sideways": replace(_BASE, price_process="ou", drift=0.0, mean_reversion=0.10),
}

# --- Brownian-process regimes -------------------------------------------------------
# Alternative price models and volatility dynamics, kept as a separate, labelled grid.
_BROWNIAN = replace(_BASE, price_process="gbm")

REGIMES.update(
    {
        # Arithmetic Brownian motion, no alpha, no drift, constant volatility: exactly the
        # world in which the closed-form Almgren-Chriss schedule is optimal.  A learned
        # policy should converge to it, which makes this the framework's ground truth.
        "abm_ac_optimal": replace(
            _BROWNIAN,
            price_process="abm",
            drift=0.0,
            imbalance_alpha=0.0,
            vol_process="none",
            vol_of_vol=0.0,
            base_depth=1500.0,  # deep book => book-walk cost stays close to AC's linear model
            base_spread=0.01,
        ),
        # Geometric Brownian motion: same economics as the default regime, different price
        # model.  Used to check that strategy rankings are not an artefact of the dynamics.
        "gbm_medium": replace(_BROWNIAN, volatility=0.02, vol_process="none", vol_of_vol=0.0),
        # Volatility clustering via an exponential-OU log-volatility process.  This is what
        # gives the env's `volatility` observation feature genuine forecast value.
        "clustered_vol": replace(
            _BROWNIAN, volatility=0.02, vol_process="expou", vol_of_vol=0.5, vol_persistence=0.90
        ),
        # Clustering plus the leverage effect: volatility rises as the price falls, which is
        # the asymmetry a seller working a large order actually faces.
        "leverage": replace(
            _BROWNIAN,
            volatility=0.02,
            vol_process="expou",
            vol_of_vol=0.5,
            vol_persistence=0.90,
            leverage_corr=-0.5,
        ),
    }
)

REGIME_GROUPS: Dict[str, List[str]] = {
    "volatility": ["low_vol", "medium_vol", "high_vol"],
    "liquidity": ["thin", "normal_liquidity", "deep"],
    "trend": ["bull", "bear", "sideways"],
    "brownian": ["abm_ac_optimal", "gbm_medium", "clustered_vol", "leverage"],
}


#: The nine regimes that make up the standard evaluation grid.
CLASSIC_REGIMES: List[str] = [
    name for group in ("volatility", "liquidity", "trend") for name in REGIME_GROUPS[group]
]
#: Regimes that swap in an explicit Brownian price process (opt-in; see :func:`list_regimes`).
BROWNIAN_REGIMES: List[str] = list(REGIME_GROUPS["brownian"])


def list_regimes(include_brownian: bool = False) -> List[str]:
    """Names of the standard evaluation regimes.

    The Brownian-process regimes are **excluded by default**.  They run different price
    models, so folding them into the standard grid would enlarge the family of tests that
    :func:`~rl_execution.backtest.engine.corrected_significance` corrects over -- which
    moves the adjusted p-value of every other comparison.

    Pass ``include_brownian=True``, or name them explicitly
    (``rlx evaluate --regimes abm_ac_optimal gbm_medium ...``), to evaluate them as a
    separate, labelled grid.
    """
    return CLASSIC_REGIMES + BROWNIAN_REGIMES if include_brownian else list(CLASSIC_REGIMES)


def get_regime(name: str) -> MarketConfig:
    if name not in REGIMES:
        raise KeyError(f"Unknown regime '{name}'. Available: {list(REGIMES)}")
    return replace(REGIMES[name])  # return a copy


def randomized_market_config(
    rng: np.random.Generator | None = None, brownian: bool = False
) -> MarketConfig:
    """Sample a market config spanning the regime axes (for robust training).

    With ``brownian=True`` the sampler additionally randomises over price process,
    volatility clustering and the leverage correlation, so a domain-randomised agent also
    generalises across *price models*, not just across parameters of one model.
    """
    rng = rng or np.random.default_rng()
    cfg = MarketConfig(
        volatility=float(rng.uniform(0.008, 0.05)),
        vol_of_vol=float(rng.uniform(0.0, 0.5)),
        drift=float(rng.uniform(-0.001, 0.001)),
        mean_reversion=float(rng.choice([0.0, 0.0, 0.10])),
        base_depth=float(rng.uniform(200.0, 1500.0)),
        base_spread=float(rng.uniform(0.01, 0.05)),
        base_market_volume=float(rng.uniform(800.0, 6000.0)),
        temporary_impact=float(rng.uniform(4.0e-6, 3.0e-5)),
        permanent_impact=float(rng.uniform(2.0e-6, 1.2e-5)),
        imbalance_alpha=float(rng.uniform(0.03, 0.07)),
        imbalance_strength=0.40,
        imbalance_persistence=0.85,
    )
    if cfg.mean_reversion > 0:
        cfg = replace(cfg, price_process="ou")
    if not brownian:
        return cfg
    mean_reversion = float(cfg.mean_reversion)
    return replace(
        cfg,
        price_process=str(
            rng.choice(["gbm", "abm", "ou"] if mean_reversion > 0 else ["gbm", "abm"])
        ),
        vol_process=str(rng.choice(["none", "expou"])),
        vol_persistence=float(rng.uniform(0.5, 0.95)),
        leverage_corr=float(rng.uniform(-0.7, 0.0)),
    )
