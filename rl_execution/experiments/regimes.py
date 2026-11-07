"""Market-regime presets for stress-testing execution strategies.

Each regime is a :class:`~rl_execution.config.MarketConfig`.  Regimes are grouped into:

* **volatility**  -- low / medium / high realised volatility,
* **liquidity**   -- thin / normal / deep order books,
* **trend**       -- bull (drift up) / bear (drift down) / sideways (mean-reverting),
* **brownian**    -- alternative price models and volatility dynamics,
* **signal**      -- the ground-truth market at five order-book signal strengths (E2).

:func:`randomized_market_config` samples a config from across these axes and is used for
*domain-randomised* training so a single agent generalises to unseen conditions;
:data:`IN_DISTRIBUTION` and :data:`HELD_OUT` split the named regimes by whether that family
covers them (E3).
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Dict, List, Sequence, Tuple

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

# --- Signal-strength sweep (E2) -----------------------------------------------------
# The ground-truth market with the order-book signal switched back on, at five strengths.
# The sweep is built on ``abm_ac_optimal`` rather than the default market so that the IC = 0
# end is exactly the world where Almgren-Chriss is optimal: on the default 500-share book
# the nonlinear book walk leaves AC slightly suboptimal even without a signal, and an
# "edge" at IC = 0 could not be told apart from exploiting the book's shape.
#
# ``imbalance_alpha`` was solved by ``solve_alpha(get_regime("abm_ac_optimal"), ic,
# n_episodes=2000, seed=CALIBRATION_BASE_SEED)``.  IC is not linear in alpha (the observed
# imbalance is clipped), hence solved rather than scaled.  Observed IC measured on 4,000
# fresh episodes (seeds from CALIBRATION_BASE_SEED + 500,000): -0.001, 0.028, 0.053, 0.103,
# 0.202; ``tests/test_experiments.py`` re-measures them.
SIGNAL_ALPHAS: Dict[str, float] = {
    "signal_ic000": 0.0,
    "signal_ic025": 1.59e-4,
    "signal_ic050": 2.97e-4,
    "signal_ic100": 5.76e-4,
    "signal_ic200": 1.15e-3,
}
#: Target observed IC of each signal regime.
SIGNAL_TARGET_IC: Dict[str, float] = {name: int(name[-3:]) / 1000 for name in SIGNAL_ALPHAS}
REGIMES.update(
    {
        name: replace(REGIMES["abm_ac_optimal"], imbalance_alpha=alpha)
        for name, alpha in SIGNAL_ALPHAS.items()
    }
)

REGIME_GROUPS: Dict[str, List[str]] = {
    "volatility": ["low_vol", "medium_vol", "high_vol"],
    "liquidity": ["thin", "normal_liquidity", "deep"],
    "trend": ["bull", "bear", "sideways"],
    "brownian": ["abm_ac_optimal", "gbm_medium", "clustered_vol", "leverage"],
    "signal": list(SIGNAL_ALPHAS),
}


#: The nine regimes that make up the standard evaluation grid.
CLASSIC_REGIMES: List[str] = [
    name for group in ("volatility", "liquidity", "trend") for name in REGIME_GROUPS[group]
]
#: Regimes that swap in an explicit Brownian price process (opt-in; see :func:`list_regimes`).
BROWNIAN_REGIMES: List[str] = list(REGIME_GROUPS["brownian"])
#: The E2 signal sweep (never part of the default grid; name them or use ``signal``).
SIGNAL_REGIMES: List[str] = list(REGIME_GROUPS["signal"])


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


#: Support of :func:`randomized_market_config` (the default, non-Brownian family):
#: ``field -> (low, high, unit)``, each drawn as ``uniform(low, high) * unit``.  Every
#: other field is held at the default, except ``mean_reversion`` (drawn from
#: :data:`_RANDOMIZED_MEAN_REVERSION`, switching the price process to OU when positive).
_RANDOMIZED_UNIFORM: Dict[str, Tuple[float, float, float]] = {
    "volatility": (0.75, 1.25, _SIGMA),
    "vol_of_vol": (0.0, 0.3, 1.0),
    "drift": (-0.025, 0.025, _SIGMA),
    "base_depth": (300.0, 1000.0, 1.0),
    "base_spread": (0.01, 0.03, 1.0),
    "base_market_volume": (1500.0, 3000.0, 1.0),
    "temporary_impact": (0.7, 1.4, _BASE.temporary_impact),
    "permanent_impact": (0.7, 1.4, _BASE.permanent_impact),
    "imbalance_alpha": (0.5, 1.5, _BASE.imbalance_alpha),
}
_RANDOMIZED_MEAN_REVERSION = [0.0, 0.0, 0.10]  # a third of the draws are OU


def randomized_market_config(
    rng: np.random.Generator | None = None, brownian: bool = False
) -> MarketConfig:
    """Sample a market config spanning the regime axes (for robust training).

    The ranges (:data:`_RANDOMIZED_UNIFORM`) sit around the calibrated default and
    deliberately stop short of the ``low_vol`` / ``high_vol`` / ``thin`` / ``deep``
    presets, so those can serve as out-of-distribution tests for an agent trained on this
    family (see :data:`HELD_OUT` and :func:`outside_randomized_support`).

    With ``brownian=True`` the sampler additionally randomises over price process,
    volatility clustering and the leverage correlation, so a domain-randomised agent also
    generalises across *price models*, not just across parameters of one model.
    """
    rng = rng or np.random.default_rng()

    def draw(name: str) -> float:
        low, high, unit = _RANDOMIZED_UNIFORM[name]
        return float(rng.uniform(low, high) * unit)

    cfg = replace(  # keyword order is the draw order; keep it stable for reproducibility
        _BASE,
        volatility=draw("volatility"),
        vol_process="iid_lognormal",
        vol_of_vol=draw("vol_of_vol"),
        drift=draw("drift"),
        mean_reversion=float(rng.choice(_RANDOMIZED_MEAN_REVERSION)),
        base_depth=draw("base_depth"),
        base_spread=draw("base_spread"),
        base_market_volume=draw("base_market_volume"),
        temporary_impact=draw("temporary_impact"),
        permanent_impact=draw("permanent_impact"),
        imbalance_alpha=draw("imbalance_alpha"),
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


def outside_randomized_support(cfg: MarketConfig) -> List[str]:
    """The fields on which ``cfg`` lies outside the default domain-randomised family.

    An empty list means :func:`randomized_market_config` (``brownian=False``) can draw a
    market with every parameter of ``cfg``, up to a rounding tolerance at the range ends,
    so an agent trained on that family has seen markets like it.  Otherwise the names say
    on which axes ``cfg`` is out of distribution.  ``vol_process="none"`` counts as the
    ``vol_of_vol = 0`` end of the sampled lognormal family.
    """
    outside = []
    for f in fields(MarketConfig):
        name, value = f.name, getattr(cfg, f.name)
        if name in _RANDOMIZED_UNIFORM:
            low, high, unit = _RANDOMIZED_UNIFORM[name]
            low, high = sorted((low * unit, high * unit))
            tol = 1e-9 * max(abs(low), abs(high))
            inside = low - tol <= value <= high + tol
        elif name == "mean_reversion":
            inside = value in _RANDOMIZED_MEAN_REVERSION
        elif name == "vol_process":
            inside = value in ("none", "iid_lognormal")
        elif name == "price_process":
            inside = value == ("ou" if cfg.mean_reversion > 0 else "abm")
        else:
            inside = value == getattr(_BASE, name)
        if not inside:
            outside.append(name)
    return outside


# --- E3 split ------------------------------------------------------------------------
#: Regimes inside the support of the domain-randomised training family: an agent trained
#: with ``rlx train --randomized`` is scored here *in distribution*.
IN_DISTRIBUTION: List[str] = ["normal_liquidity", "medium_vol", "sideways"]
#: Regimes outside it on at least one axis (``tests/test_experiments.py`` checks both
#: lists against :func:`outside_randomized_support`).  ``bull`` / ``bear`` (drift
#: +-0.04 sigma vs a +-0.025 sigma training range) and ``gbm_medium`` (GBM vs ABM over a
#: 20-step horizon) are only mildly shifted; the others move volatility, liquidity or the
#: volatility dynamics well beyond anything seen in training.
HELD_OUT: List[str] = [
    "low_vol",
    "high_vol",
    "thin",
    "deep",
    "bull",
    "bear",
    "gbm_medium",
    "clustered_vol",
    "leverage",
]
SPLITS: Dict[str, List[str]] = {"in_distribution": IN_DISTRIBUTION, "held_out": HELD_OUT}


def resolve_regimes(names: Sequence[str]) -> List[str]:
    """Expand group names (``signal``, ``held_out``, ...) into regimes, keeping order.

    Each name is a regime, a key of :data:`REGIME_GROUPS` or a key of :data:`SPLITS`.
    Duplicates are dropped; an unknown name raises ``KeyError``.
    """
    out: List[str] = []
    for name in names:
        members = REGIME_GROUPS.get(name) or SPLITS.get(name)
        if members is None:
            get_regime(name)  # raises a helpful KeyError for an unknown name
            members = [name]
        out.extend(m for m in members if m not in out)
    return out
