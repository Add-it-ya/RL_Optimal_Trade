"""Gymnasium environment for optimal trade execution.

A single episode works a parent order (buy or sell of ``total_inventory`` shares) over a
fixed ``horizon`` of decision steps.  At each step the agent chooses a trading *pace*; the
environment fills it as a market order against the simulated limit order book, books the
cost, and moves the market along its pre-drawn path.  The final step always liquidates
whatever is left.

**Action -- relative to TWAP.**  The continuous action ``a`` in ``[-1, 1]`` sets a
multiplier ``m`` on the TWAP pace ``remaining / steps_left``::

    a = -1  ->  m = 0               wait
    a =  0  ->  m = 1               trade at the TWAP pace
    a = +1  ->  m = steps_left      liquidate everything now

with ``m = 1 + a`` below zero and ``m = steps_left ** a`` above it.  Every schedule is
reachable, and an untrained policy (``a ~ 0``) starts out as TWAP rather than as "do
nothing".  The discrete action space is the grid of multipliers :data:`MULTIPLIERS`.

**Observation** (7 features, all O(1); see :data:`ExecutionEnv.FEATURE_NAMES`)::

    [ inventory_remaining, time_remaining, price_move, imbalance, volatility, spread, depth ]

``price_move`` is the move since arrival in units of ``sigma * sqrt(t)``; it and the book
``imbalance`` are signed by side, so positive always means good for this trader (a higher
price, or buying pressure, for a seller).  ``volatility`` is the step volatility relative to
its configured level, ``spread`` is in units of 10 bps and ``depth`` is the top-five-level
book depth relative to a fixed reference, so it differs between thin and deep markets.

**Reward -- the Almgren-Chriss objective, exactly.**  In bps of the arrival notional::

    r_t = - [ cost of this fill against the decision-time mid, including commission ]
          + [ P&L of the inventory still held, over the next step ]       (mark-to-market)
          - lambda * (sigma_t * x_{t+1})^2                                 (variance penalty)

times ``reward_scale``.  The first two terms telescope, so an episode's rewards sum to
exactly ``-(implementation_shortfall + lambda * sum_t sigma_t^2 x_t^2)``: the mean-variance
objective that Almgren-Chriss minimises in closed form.  Marking the held inventory to
market every step also keeps the price move since arrival -- already sunk, and the biggest
source of variance in implementation shortfall -- out of every future return.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_execution.config import ActionType, ExecutionConfig, MarketConfig, Side
from rl_execution.envs.brownian import make_streams
from rl_execution.envs.market import MarketSimulator

#: TWAP multipliers available to discrete-action agents.
MULTIPLIERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)

#: Shares in the top five levels of both sides of a reference book (``base_depth=500``),
#: used to scale the ``depth`` feature so it tells thin and deep markets apart.
DEPTH_REFERENCE = 5_000.0


class ExecutionEnv(gym.Env):
    """Custom Gymnasium environment for single-order optimal execution."""

    metadata = {"render_modes": ["human"]}

    FEATURE_NAMES = (
        "inventory_remaining",
        "time_remaining",
        "price_move",
        "imbalance",
        "volatility",
        "spread",
        "depth",
    )

    def __init__(
        self,
        market_config: Optional[MarketConfig] = None,
        exec_config: Optional[ExecutionConfig] = None,
    ) -> None:
        super().__init__()
        self.market_config = market_config or MarketConfig()
        self.exec_config = exec_config or ExecutionConfig()

        self.side = self.exec_config.side
        self.horizon = int(self.exec_config.horizon)
        self.total_inventory = float(self.exec_config.total_inventory)

        self.discrete = self.exec_config.action_type is ActionType.DISCRETE
        self.action_space: spaces.Space
        if self.discrete:
            self.action_space = spaces.Discrete(len(MULTIPLIERS))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        low = np.array([0.0, 0.0, -5.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 5.0, 1.0, 5.0, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.market: Optional[MarketSimulator] = None
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ actions
    @property
    def steps_left(self) -> int:
        """Decision steps remaining, including the current one."""
        return max(self.horizon - getattr(self, "t", 0), 1)

    def action_to_fraction(self, action) -> float:
        """Fraction of the *remaining* inventory that ``action`` trades this step."""
        n = self.steps_left
        if self.discrete:
            idx = int(np.clip(int(np.asarray(action).reshape(-1)[0]), 0, len(MULTIPLIERS) - 1))
            multiplier = MULTIPLIERS[idx]
        else:
            a = float(np.clip(float(np.asarray(action, dtype=np.float64).reshape(-1)[0]), -1, 1))
            multiplier = 1.0 + a if a <= 0 else float(n) ** a
        return float(min(multiplier / n, 1.0))

    def fraction_to_action(self, fraction: float):
        """The action that trades ``fraction`` of the remaining inventory this step.

        Exact inverse of :meth:`action_to_fraction` for the continuous space; the nearest
        multiplier on the discrete grid.  Lets rule-based strategies run through exactly
        the same action interface as the agents.
        """
        n = self.steps_left
        multiplier = float(np.clip(fraction, 0.0, 1.0)) * n
        if self.discrete:
            return int(np.argmin(np.abs(np.asarray(MULTIPLIERS) - multiplier)))
        if multiplier <= 1.0 or n <= 1:
            a = multiplier - 1.0
        else:
            a = float(np.log(multiplier) / np.log(n))
        return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32)

    # ------------------------------------------------------------------ observation
    def _build_obs(self) -> np.ndarray:
        assert self.market is not None
        m, cfg = self.market, self.market_config
        snap = m.snapshot
        sign = self.side.sign
        sigma_ref = max(float(cfg.volatility) * np.sqrt(float(cfg.dt)), 1e-12)
        obs = np.array(
            [
                self.remaining / self.total_inventory,
                (self.horizon - self.t) / self.horizon,
                sign * (m.mid / self.arrival_price - 1.0) / (sigma_ref * np.sqrt(max(self.t, 1))),
                sign * snap.imbalance,
                m.sigma / cfg.volatility if cfg.volatility > 0 else 1.0,
                snap.spread / m.mid * 1e4 / 10.0,
                snap.depth(5) / DEPTH_REFERENCE,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------ gym API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # derive a fresh market RNG from the env's seeded RNG for reproducibility
        market_seed = int(self.np_random.integers(0, 2**31 - 1))
        rng = np.random.default_rng(market_seed)
        # ``options={"antithetic": True}`` mirrors every normal draw (Z -> -Z).  Pairing a
        # seed's ordinary episode with its antithetic twin is the variance-reduction trick
        # used by ``evaluate(antithetic=True)``.
        antithetic = bool((options or {}).get("antithetic", False))
        streams = make_streams(rng, antithetic=antithetic, method=self.market_config.normal_method)
        self.market = MarketSimulator(self.market_config, rng=rng, streams=streams)
        self.market.reset(horizon=self.horizon)

        self.t = 0
        self.remaining = self.total_inventory
        self.arrival_price = float(self.market.mid)
        self.executed_shares = 0.0
        self.executed_notional = 0.0
        self.total_commission = 0.0
        self.risk_penalty_bps = 0.0
        self.cum_reward = 0.0
        self.history = []

        return self._build_obs(), {"arrival_price": self.arrival_price}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.market is not None, "Call reset() before step()."
        cfg = self.exec_config
        m = self.market
        sign = self.side.sign
        bps = 1e4 / (self.arrival_price * self.total_inventory)  # cash -> bps of notional

        fraction = self.action_to_fraction(action)
        if self.t >= self.horizon - 1:
            fraction = 1.0  # the last step liquidates whatever is left
        shares = float(np.clip(fraction * self.remaining, 0.0, self.remaining))

        decision_mid = float(m.mid)  # the mark the agent saw when it chose
        if cfg.latency_steps > 0 and shares > 0:
            m.apply_latency(cfg.latency_steps)  # fill lands part-way into the step

        exec_result = m.execute(self.side, shares)
        filled = exec_result.filled_shares
        commission = (
            cfg.commission_per_share * filled + (cfg.commission_bps / 1e4) * exec_result.notional
        )
        self.remaining = max(self.remaining - filled, 0.0)
        self.executed_shares += filled
        self.executed_notional += exec_result.notional
        self.total_commission += commission

        # ---- cost of this fill against the decision-time mark ----
        fill_cost_bps = (sign * filled * (decision_mid - exec_result.avg_price) + commission) * bps

        # ---- carry the rest of the order through the next step ----
        self.t += 1
        terminated = self.remaining <= 1e-9 or self.t >= self.horizon
        hold_pnl_bps = 0.0
        risk_bps = 0.0
        if not terminated:
            held = self.remaining
            step_std_bps = m.step_std() / self.arrival_price * 1e4
            risk_bps = cfg.reward.risk_aversion * (step_std_bps * held / self.total_inventory) ** 2
            m.advance()
            hold_pnl_bps = sign * held * (m.mid - decision_mid) * bps

        self.risk_penalty_bps += risk_bps
        reward = (hold_pnl_bps - fill_cost_bps - risk_bps) * cfg.reward.reward_scale
        self.cum_reward += reward

        record = {
            "t": self.t - 1,
            "action_fraction": fraction,
            "shares": filled,
            "exec_price": exec_result.avg_price,
            "raw_exec_price": exec_result.raw_avg_price,
            "decision_mid": decision_mid,
            "mid_before": exec_result.mid_before,
            "mid_after": exec_result.mid_after,
            "arrival_price": self.arrival_price,
            "slippage": exec_result.slippage,
            "temp_impact_cost": exec_result.temp_impact_cost,
            "perm_impact": exec_result.perm_impact,
            "notional": exec_result.notional,
            "commission": commission,
            "fill_cost_bps": fill_cost_bps,
            "hold_pnl_bps": hold_pnl_bps,
            "risk_bps": risk_bps,
            "inventory_after": self.remaining,
            "market_volume": m.market_volume,
            "reward": reward,
        }
        self.history.append(record)

        obs = self._build_obs()
        info: Dict[str, Any] = dict(record)
        if terminated:
            info["episode_summary"] = self._episode_summary()
        return obs, float(reward), terminated, False, info

    # ------------------------------------------------------------------ summaries
    def _episode_summary(self) -> Dict[str, Any]:
        """Aggregate execution-quality statistics for the finished episode."""
        assert self.market is not None
        avg_fill = (
            self.executed_notional / self.executed_shares
            if self.executed_shares > 0
            else self.arrival_price
        )
        # implementation shortfall in basis points (signed: + = cost)
        paper_notional = self.arrival_price * self.total_inventory
        realised = self.executed_notional
        if self.side is Side.SELL:
            # ideal proceeds minus actual proceeds, plus commissions
            is_cash = paper_notional - realised + self.total_commission
        else:
            is_cash = realised - paper_notional + self.total_commission
        is_bps = (is_cash / paper_notional) * 1e4
        return {
            "side": self.side.value,
            "total_inventory": self.total_inventory,
            "executed_shares": self.executed_shares,
            "unexecuted_shares": self.remaining,
            "avg_fill_price": avg_fill,
            "arrival_price": self.arrival_price,
            "final_mid": self.market.mid,
            "executed_notional": self.executed_notional,
            "total_commission": self.total_commission,
            "implementation_shortfall_cash": is_cash,
            "implementation_shortfall_bps": is_bps,
            "risk_penalty_bps": self.risk_penalty_bps,
            "objective_bps": is_bps + self.risk_penalty_bps,
            "cum_reward": self.cum_reward,
        }

    def render(self) -> None:  # pragma: no cover - simple text rendering
        if not self.history:
            return
        r = self.history[-1]
        print(
            f"t={r['t']:>3d} | traded {r['shares']:>9.1f} @ {r['exec_price']:.4f} "
            f"| mid {r['mid_after']:.4f} | inv left {r['inventory_after']:>10.1f} "
            f"| reward {r['reward']:+.4f}"
        )

    def episode_dataframe(self):
        """Return the per-step history of the most recent episode as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.history)
