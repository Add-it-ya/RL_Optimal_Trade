"""Typed configuration objects for the execution framework.

All tunable parameters live here as frozen-ish dataclasses so that experiments are
fully described by a small set of serialisable configs.  Configs can be round-tripped
to / from plain ``dict`` (and therefore YAML / JSON) via :meth:`to_dict` / :meth:`from_dict`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Any, Dict


class Side(str, Enum):
    """Direction of the parent order."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for a SELL (we receive cash, prefer high prices),
        -1 for a BUY (we pay cash, prefer low prices).

        Used so that a single cost expression handles both directions:
        ``shortfall = sign * quantity * (arrival_price - exec_price)``.
        """
        return 1 if self is Side.SELL else -1


class ActionType(str, Enum):
    """Action-space parametrisation exposed by the environment."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass
class MarketConfig:
    """Parameters of the synthetic market / limit-order-book simulator.

    The price process is an arithmetic mid-price with optional drift, stochastic
    volatility scaling and Ornstein-Uhlenbeck mean reversion (for sideways regimes).
    The order book is regenerated each step around the current mid with a depth
    profile that grows away from the touch and an imbalance driven by a latent AR(1)
    factor that is (weakly) predictive of the next price move.
    """

    # --- price level / discretisation ---
    initial_price: float = 100.0
    tick_size: float = 0.01
    n_levels: int = 10                 # price levels per side exposed in the book

    # --- spread & depth ---
    base_spread: float = 0.02          # baseline absolute bid-ask spread (price units)
    spread_vol: float = 0.30           # relative log-noise applied to the spread
    base_depth: float = 500.0          # ~shares resting at the best level
    level_growth: float = 0.15         # fractional depth increase per level into the book
    depth_vol: float = 0.25            # relative log-noise applied to resting size

    # --- price dynamics (per step) ---
    volatility: float = 0.02           # std of per-step mid return (fraction of price)
    drift: float = 0.0                 # per-step expected return (regime: +bull / -bear)
    mean_reversion: float = 0.0        # OU pull toward initial price (sideways regime)
    vol_of_vol: float = 0.0            # stochastic-volatility log-noise on per-step sigma

    # --- microstructure / impact ---
    permanent_impact: float = 5.0e-6   # mid shift per share executed (Kyle-lambda style)
    temporary_impact: float = 1.0e-5   # extra per-share concession beyond walking the book
    imbalance_persistence: float = 0.85  # AR(1) coefficient of the latent imbalance factor
    imbalance_strength: float = 0.30   # how strongly imbalance skews depth between sides
    imbalance_alpha: float = 0.0       # how strongly imbalance forecasts the next return

    # --- ambient (background) traded volume, used by POV / VWAP baselines ---
    base_market_volume: float = 2000.0  # ~shares traded by the rest of the market per step
    volume_vol: float = 0.35           # relative log-noise on market volume
    volume_u_shape: float = 0.5        # intraday U-shape strength (higher at open/close)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RewardConfig:
    """Weights of the multi-component reward.

    The per-step reward is the negative *implementation-shortfall contribution* of the
    fill (which already embeds slippage, spread and temporary impact through the realised
    price), minus an inventory-risk term and an explicit temporary-impact penalty.  The
    sum over an episode therefore approximates ``-(implementation_shortfall + risk)``.
    """

    risk_aversion: float = 0.5          # weight on running inventory variance (Almgren-Chriss lambda)
    temp_impact_penalty: float = 1.0    # extra penalty on temporary impact cost
    unexecuted_penalty: float = 10.0    # multiplier on the cost of force-liquidated leftover
    reward_scale: float = 1.0e4         # scale shortfall (fractions) into basis points

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ExecutionConfig:
    """Parameters of the execution *task* (the parent order to be worked)."""

    total_inventory: float = 10_000.0   # shares of the parent order
    horizon: int = 20                   # number of decision steps (e.g. trading windows)
    side: Side = Side.SELL
    action_type: ActionType = ActionType.CONTINUOUS
    n_discrete_actions: int = 11        # granularity of the discrete action grid in [0, 1]

    # --- frictions ---
    commission_per_share: float = 0.0   # fixed cash cost per share
    commission_bps: float = 1.0         # proportional fee in basis points of notional
    latency_steps: float = 0.0          # fractional step delay between decision and fill

    # --- termination behaviour ---
    force_liquidation: bool = True      # liquidate any leftover on the final step
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        # allow passing plain strings / dicts (e.g. from YAML)
        if isinstance(self.side, str):
            self.side = Side(self.side)
        if isinstance(self.action_type, str):
            self.action_type = ActionType(self.action_type)
        if isinstance(self.reward, dict):
            self.reward = RewardConfig.from_dict(self.reward)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        d["action_type"] = self.action_type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)
