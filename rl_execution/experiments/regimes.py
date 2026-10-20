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

# Every regime starts from the calibrated default market (see MarketConfig and
# rl_execution.experiments.calibration): arithmetic Brownian motion, impact and price risk
# that genuinely trade off, and a realistic order-book signal (IC ~0.075).  Regimes then
# move one axis at a time.  Drifts are kept small (4% of the per-step volatility) so that
# timing skill, not trivial front-loading, is what differentiates strategies.
_BASE = MarketConfig()
_SIGMA = _BASE.volatility

REGIMES: Dict[str, MarketConfig] = {
    # --- volatility ---
    "low_vol": replace(_BASE, volatility=0.5 * _SIGMA),
    "medium_vol": replace(_BASE, vol_process="iid_lognormal", vol_of_vol=0.2),
    "high_vol": replace(
        _BASE, volatility=2.0 * _SIGMA, vol_process="iid_lognormal", vol_of_vol=0.5
    ),
    # --- liquidity (impact scales with how thin the book is) ---
    "thin": replace(
        _BASE,
        base_depth=200.0,
        base_spread=0.05,
        base_market_volume=800.0,
        temporary_impact=2.0 * _BASE.temporary_impact,
        permanent_impact=2.0 * _BASE.permanent_impact,
    ),
    "normal_liquidity": replace(_BASE),
    "deep": replace(
        _BASE,
        base_depth=1500.0,
        base_spread=0.01,
        base_market_volume=6000.0,
        temporary_impact=0.5 * _BASE.temporary_impact,
        permanent_impact=0.5 * _BASE.permanent_impact,
    ),
    # --- trend ---
    "bull": replace(_BASE, drift=0.04 * _SIGMA),
    "bear": replace(_BASE, drift=-0.04 * _SIGMA),
    "sideways": replace(_BASE, price_process="ou", mean_reversion=0.10),
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
        "gbm_medium": replace(_BROWNIAN),
        # Volatility clustering via an exponential-OU log-volatility process.  This is what
        # gives the env's `volatility` observation feature genuine forecast value.
        "clustered_vol": replace(
            _BROWNIAN, vol_process="expou", vol_of_vol=0.5, vol_persistence=0.90
        ),
        # Clustering plus the leverage effect: volatility rises as the price falls, which is
        # the asymmetry a seller working a large order actually faces.
        "leverage": replace(
            _BROWNIAN,
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

    The ranges sit around the calibrated default and deliberately stop short of the
    ``low_vol`` / ``high_vol`` / ``thin`` / ``deep`` presets, so those can serve as
    out-of-distribution tests for an agent trained on this family.

    With ``brownian=True`` the sampler additionally randomises over price process,
    volatility clustering and the leverage correlation, so a domain-randomised agent also
    generalises across *price models*, not just across parameters of one model.
    """
    rng = rng or np.random.default_rng()
    cfg = replace(
        _BASE,
        volatility=float(rng.uniform(0.75, 1.25) * _SIGMA),
        vol_process="iid_lognormal",
        vol_of_vol=float(rng.uniform(0.0, 0.3)),
        drift=float(rng.uniform(-0.025, 0.025) * _SIGMA),
        mean_reversion=float(rng.choice([0.0, 0.0, 0.10])),
        base_depth=float(rng.uniform(300.0, 1000.0)),
        base_spread=float(rng.uniform(0.01, 0.03)),
        base_market_volume=float(rng.uniform(1500.0, 3000.0)),
        temporary_impact=float(rng.uniform(0.7, 1.4) * _BASE.temporary_impact),
        permanent_impact=float(rng.uniform(0.7, 1.4) * _BASE.permanent_impact),
        imbalance_alpha=float(rng.uniform(0.5, 1.5) * _BASE.imbalance_alpha),
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
