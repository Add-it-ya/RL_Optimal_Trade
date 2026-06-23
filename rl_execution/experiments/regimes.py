"""Market-regime presets for stress-testing execution strategies.

Each regime is a :class:`~rl_execution.config.MarketConfig`.  Regimes are grouped into:

* **volatility**  -- low / medium / high realised volatility,
* **liquidity**   -- thin / normal / deep order books,
* **trend**       -- bull (drift up) / bear (drift down) / sideways (mean-reverting).

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
# is what differentiates strategies.
_BASE = MarketConfig(imbalance_alpha=0.05, imbalance_strength=0.40,
                     imbalance_persistence=0.85)

REGIMES: Dict[str, MarketConfig] = {
    # --- volatility ---
    "low_vol": replace(_BASE, volatility=0.008, vol_of_vol=0.0),
    "medium_vol": replace(_BASE, volatility=0.02, vol_of_vol=0.2),
    "high_vol": replace(_BASE, volatility=0.05, vol_of_vol=0.5),
    # --- liquidity ---
    "thin": replace(_BASE, base_depth=200.0, base_spread=0.05,
                    base_market_volume=800.0, temporary_impact=3.0e-5,
                    permanent_impact=1.2e-5),
    "normal_liquidity": replace(_BASE),
    "deep": replace(_BASE, base_depth=1500.0, base_spread=0.01,
                    base_market_volume=6000.0, temporary_impact=4.0e-6,
                    permanent_impact=2.0e-6),
    # --- trend (drift small vs sqrt-horizon volatility) ---
    "bull": replace(_BASE, drift=0.0008),
    "bear": replace(_BASE, drift=-0.0008),
    "sideways": replace(_BASE, drift=0.0, mean_reversion=0.10),
}

REGIME_GROUPS: Dict[str, List[str]] = {
    "volatility": ["low_vol", "medium_vol", "high_vol"],
    "liquidity": ["thin", "normal_liquidity", "deep"],
    "trend": ["bull", "bear", "sideways"],
}


def list_regimes() -> List[str]:
    return list(REGIMES.keys())


def get_regime(name: str) -> MarketConfig:
    if name not in REGIMES:
        raise KeyError(f"Unknown regime '{name}'. Available: {list_regimes()}")
    return replace(REGIMES[name])  # return a copy


def randomized_market_config(rng: np.random.Generator | None = None) -> MarketConfig:
    """Sample a market config spanning the regime axes (for robust training)."""
    rng = rng or np.random.default_rng()
    return MarketConfig(
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
