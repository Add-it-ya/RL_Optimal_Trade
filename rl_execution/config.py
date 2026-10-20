"""Typed configuration objects for the execution framework.

All tunable parameters live here as frozen-ish dataclasses so that experiments are
fully described by a small set of serialisable configs.  Configs can be round-tripped
to / from plain ``dict`` (and therefore YAML / JSON) via :meth:`to_dict` / :meth:`from_dict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Dict, Optional

_TRUE_TOKENS = {"true", "yes", "1"}
_FALSE_TOKENS = {"false", "no", "0"}


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a number, got {value!r}") from exc


def _as_int(value: Any, name: str) -> int:
    as_float = _as_float(value, name)
    if as_float != int(as_float):
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    return int(as_float)


def _as_bool(value: Any, name: str) -> bool:
    """Accept real booleans and the YAML boolean spellings -- and nothing else.

    A bare ``bool(value)`` would happily take the string ``'false# oops'`` (a missing
    space before a YAML comment) as ``True`` and silently invalidate a whole run.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        raise ValueError(f"{name}: expected a boolean, got {value!r}")
    try:  # numpy bools and plain 0 / 1
        as_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a boolean, got {value!r}") from exc
    if as_int in (0, 1) and value == as_int:
        return bool(as_int)
    raise ValueError(f"{name}: expected a boolean, got {value!r}")


def _as_token(value: Any, name: str) -> str:
    """Normalise an enum-like string so surrounding whitespace / case is not a failure."""
    if not isinstance(value, str):
        raise ValueError(f"{name}: expected a string, got {value!r}")
    return value.strip().lower()


#: Coercion by declared field type.  ``from __future__ import annotations`` makes every
#: annotation a string, so a new field is covered the moment it is declared.
_COERCERS: Dict[str, Callable[[Any, str], Any]] = {
    "float": _as_float,
    "int": _as_int,
    "bool": _as_bool,
    "str": _as_token,
}


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

    The mid-price follows an explicit Brownian process (``price_process``: arithmetic,
    geometric or Ornstein-Uhlenbeck) whose whole path is drawn before the agent acts.
    The order book is re-quoted each step around the current mid with a depth profile
    that grows away from the touch and an imbalance driven by a latent AR(1) factor that
    is (weakly) predictive of the next price move.

    The defaults are calibrated (see :mod:`rl_execution.experiments.calibration` and
    ``scripts/calibrate.py``) so that impact and price risk genuinely trade off for the
    default order of 10,000 shares over 20 steps of ~20 minutes: dumping the order at
    arrival costs ~150 bps, TWAP ~20 bps with ~100 bps of price risk, and the order-book
    imbalance predicts the next return with a realistic information coefficient (~0.075).
    """

    # --- price level / discretisation ---
    initial_price: float = 100.0
    tick_size: float = 0.01
    n_levels: int = 10  # price levels per side exposed in the book

    # --- spread & depth ---
    base_spread: float = 0.02  # baseline absolute bid-ask spread (price units)
    spread_vol: float = 0.30  # relative log-noise applied to the spread
    base_depth: float = 500.0  # ~shares resting at the best level
    level_growth: float = 0.15  # fractional depth increase per level into the book
    depth_vol: float = 0.25  # relative log-noise applied to resting size

    # --- price dynamics (per unit time; one step spans ``dt``) ---
    volatility: float = 0.004  # std of the mid return per unit time (fraction of price)
    drift: float = 0.0  # expected return per unit time (regime: +bull / -bear)
    mean_reversion: float = 0.0  # OU pull toward initial price (sideways regime)
    vol_of_vol: float = 0.0  # stochastic-volatility log-noise on sigma

    # --- Brownian-motion layer (see rl_execution.envs.processes) ---
    price_process: str = "abm"  # abm | gbm | ou
    dt: float = 1.0  # time spanned by one decision step, in volatility units
    normal_method: str = "ziggurat"  # ziggurat | inverse_cdf (monotone; needed for QMC)
    vol_process: str = "none"  # none | iid_lognormal | expou (clustering)
    vol_persistence: float = 0.0  # AR(1) persistence of ln(sigma) under 'expou'
    leverage_corr: float = 0.0  # corr(dW_price, dW_vol); < 0 = vol rises as price falls
    imbalance_ou_scaling: bool = True  # sqrt(1-phi^2) innovations => unit-variance imbalance

    # --- microstructure / impact ---
    permanent_impact: float = 2.0e-5  # mid shift per share executed (Kyle-lambda style)
    temporary_impact: float = 1.5e-4  # extra per-share concession beyond walking the book
    imbalance_persistence: float = 0.85  # AR(1) coefficient of the latent imbalance factor
    imbalance_strength: float = 0.40  # how strongly imbalance skews depth between sides
    imbalance_alpha: float = 4.5e-4  # next-return forecast power of imbalance (IC ~0.075)

    # --- ambient (background) traded volume, used by POV / VWAP baselines ---
    base_market_volume: float = 2000.0  # ~shares traded by the rest of the market per step
    volume_vol: float = 0.35  # relative log-noise on market volume
    volume_u_shape: float = 0.5  # intraday U-shape strength (higher at open/close)

    def __post_init__(self) -> None:
        """Coerce every field to its declared type, rejecting junk at construction.

        Without this a YAML typo reaches :meth:`RunConfig.validate` as a *string* --
        ``imbalance_ou_scaling: false# oops`` is truthy, passes every membership check
        and silently switches the stationary imbalance scaling on.  Range and membership
        checks stay in :meth:`RunConfig.validate`; this is only about types.
        """
        for f in fields(self):
            # a string under PEP 563, a real type if some caller evaluates the annotations
            annotation = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
            coerce = _COERCERS.get(annotation)
            if coerce is not None:
                setattr(self, f.name, coerce(getattr(self, f.name), f.name))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RewardConfig:
    """The execution objective the agent is rewarded on.

    Every episode is scored on the Almgren-Chriss mean-variance objective, in basis points
    of the arrival notional::

        J = implementation_shortfall + risk_aversion * sum_t (sigma_t * x_t)^2

    where ``x_t`` is the fraction of the order still held over step ``t`` and ``sigma_t``
    that step's price volatility in bps.  The per-step rewards sum to exactly
    ``-J * reward_scale`` (see :class:`~rl_execution.envs.ExecutionEnv`), and the
    Almgren-Chriss baseline uses the same ``risk_aversion`` (converted by
    :func:`ac_risk_aversion`), so in a signal-free arithmetic-Brownian market its schedule
    is the optimum of exactly this objective.
    """

    risk_aversion: float = 1.4e-3  # lambda per bps^2 of carried variance (AC kappa*T ~2.5)
    reward_scale: float = 1.0  # reward units per bps of objective

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ExecutionConfig:
    """Parameters of the execution *task* (the parent order to be worked)."""

    total_inventory: float = 10_000.0  # shares of the parent order
    horizon: int = 20  # number of decision steps (e.g. trading windows); the last one
    # always liquidates whatever is left
    side: Side = Side.SELL
    action_type: ActionType = ActionType.CONTINUOUS

    # --- frictions ---
    commission_per_share: float = 0.0  # fixed cash cost per share
    commission_bps: float = 1.0  # proportional fee in basis points of notional
    latency_steps: float = 0.0  # fractional step delay between decision and fill

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


def ac_risk_aversion(market: MarketConfig, execution: ExecutionConfig) -> float:
    """Almgren-Chriss ``lambda`` (per cash^2) equivalent to the reward's ``risk_aversion``.

    The reward penalises variance in bps of arrival notional ``N = S0 * X``; Almgren-Chriss
    penalises it in cash.  Since ``cost_bps = cost_cash * 1e4 / N``, the objectives
    ``E[c_bps] + lambda_bps Var[c_bps]`` and ``E[c] + lambda_AC Var[c]`` share a minimiser
    exactly when ``lambda_AC = lambda_bps * 1e4 / N``.
    """
    notional = float(market.initial_price) * float(execution.total_inventory)
    return float(execution.reward.risk_aversion) * 1e4 / notional


@dataclass
class RunConfig:
    """Everything needed to reproduce a single training run from one serialised object.

    A run is fully described by ``{agent, hyperparameters, training schedule, seed,
    market, task}``.  Round-trips to YAML (see :mod:`rl_execution.utils.config_io`) so
    ``rlx train --config run.yaml --seeds 3`` is sufficient to reproduce, and validates
    parameter ranges via :meth:`validate`.
    """

    # --- what to train ---
    agent: str = "ppo"
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- training schedule ---
    timesteps: int = 80_000
    randomized: bool = True
    regime: Optional[str] = None
    seed: int = 0
    device: str = "cpu"

    # --- evaluation ---
    eval_episodes: int = 100
    eval_base_seed: int = 10_000

    # --- market + execution task ---
    market: MarketConfig = field(default_factory=MarketConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # --- bookkeeping ---
    experiment: str = "default"  # groups runs under runs/<experiment>/

    def __post_init__(self) -> None:
        # allow nested configs to arrive as plain dicts (e.g. from YAML / JSON)
        if isinstance(self.market, dict):
            self.market = MarketConfig.from_dict(self.market)
        if isinstance(self.execution, dict):
            self.execution = ExecutionConfig.from_dict(self.execution)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "agent_kwargs": dict(self.agent_kwargs),
            "timesteps": self.timesteps,
            "randomized": self.randomized,
            "regime": self.regime,
            "seed": self.seed,
            "device": self.device,
            "eval_episodes": self.eval_episodes,
            "eval_base_seed": self.eval_base_seed,
            "market": self.market.to_dict(),
            "execution": self.execution.to_dict(),
            "experiment": self.experiment,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        if "market" in kwargs and isinstance(kwargs["market"], dict):
            kwargs["market"] = MarketConfig.from_dict(kwargs["market"])
        if "execution" in kwargs and isinstance(kwargs["execution"], dict):
            kwargs["execution"] = ExecutionConfig.from_dict(kwargs["execution"])
        return cls(**kwargs)

    def validate(self) -> "RunConfig":
        """Validate parameter ranges; returns ``self`` so it chains. Raises ``ValueError``."""
        m, e = self.market, self.execution
        checks = {
            "agent in {ppo, dqn, doubledqn}": self.agent.lower() in {"ppo", "dqn", "doubledqn"},
            "timesteps > 0": self.timesteps > 0,
            "seed >= 0": self.seed >= 0,
            "eval_episodes > 0": self.eval_episodes > 0,
            "0 <= imbalance_alpha <= 1": 0.0 <= m.imbalance_alpha <= 1.0,
            "0 <= imbalance_persistence < 1": 0.0 <= m.imbalance_persistence < 1.0,
            "volatility >= 0": m.volatility >= 0.0,
            "initial_price > 0": m.initial_price > 0.0,
            "horizon > 0": e.horizon > 0,
            "total_inventory > 0": e.total_inventory > 0.0,
            # --- Brownian layer ---
            "price_process in {abm, gbm, ou}": m.price_process in {"abm", "gbm", "ou"},
            "dt > 0": m.dt > 0.0,
            "normal_method in {ziggurat, inverse_cdf}": m.normal_method
            in {"ziggurat", "inverse_cdf"},
            "vol_process in {none, iid_lognormal, expou}": m.vol_process
            in {"none", "iid_lognormal", "expou"},
            "0 <= vol_persistence < 1": 0.0 <= m.vol_persistence < 1.0,
            "-1 <= leverage_corr <= 1": -1.0 <= m.leverage_corr <= 1.0,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError("Invalid RunConfig: " + "; ".join(failed))
        return self
