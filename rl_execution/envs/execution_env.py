"""Gymnasium environment for optimal trade execution.

A single episode works a parent order (buy or sell of ``total_inventory`` shares) over a
fixed ``horizon`` of decision steps.  At each step the agent chooses what *fraction of the
remaining inventory* to execute as a market order; the environment fills it against the
simulated limit order book, books the realised cost, and evolves the market.

Observation (8 features, see :data:`ExecutionEnv.FEATURE_NAMES`)::

    [ remaining_inventory_frac, time_remaining_frac, mid/arrival - 1,
      spread/mid, recent_volatility, book_imbalance, normalised_depth, prev_action ]

Reward (per step) approximates the negative implementation-shortfall contribution of the
fill, net of transaction costs and an explicit temporary-impact penalty, minus a running
inventory-risk term.  Summed over an episode it approximates
``-(implementation_shortfall + risk)`` expressed in basis points.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_execution.config import ActionType, ExecutionConfig, MarketConfig, Side
from rl_execution.envs.market import MarketSimulator


class ExecutionEnv(gym.Env):
    """Custom Gymnasium environment for single-order optimal execution."""

    metadata = {"render_modes": ["human"]}

    FEATURE_NAMES = (
        "inventory_remaining",
        "time_remaining",
        "price_return",
        "rel_spread",
        "volatility",
        "imbalance",
        "depth",
        "prev_action",
    )

    def __init__(
        self,
        market_config: Optional[MarketConfig] = None,
        exec_config: Optional[ExecutionConfig] = None,
        market_source: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.market_config = market_config or MarketConfig()
        self.exec_config = exec_config or ExecutionConfig()
        # optional historical / external market source (see rl_execution.data)
        self.market_source = market_source

        self.side = self.exec_config.side
        self.horizon = int(self.exec_config.horizon)
        self.total_inventory = float(self.exec_config.total_inventory)

        # ---- action space ----
        if self.exec_config.action_type is ActionType.DISCRETE:
            self.n_actions = int(self.exec_config.n_discrete_actions)
            self.action_space = spaces.Discrete(self.n_actions)
            self._action_grid = np.linspace(0.0, 1.0, self.n_actions)
        else:
            self.n_actions = None
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self._action_grid = None

        # ---- observation space (bounded for NN stability) ----
        low = np.array([0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.market: Optional[MarketSimulator] = None
        self.history: List[Dict[str, Any]] = []
        self._depth_norm = max(self.market_config.base_depth * 5.0, 1.0)

    # ------------------------------------------------------------------ helpers
    def _action_to_fraction(self, action) -> float:
        """Map a raw env action to a fraction of remaining inventory in [0, 1]."""
        if self.exec_config.action_type is ActionType.DISCRETE:
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, self.n_actions - 1))
            return float(self._action_grid[idx])
        val = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        return float(np.clip(val, 0.0, 1.0))

    def _build_obs(self) -> np.ndarray:
        m = self.market
        snap = m.snapshot
        inv_frac = self.remaining / self.total_inventory if self.total_inventory > 0 else 0.0
        time_frac = (self.horizon - self.t) / self.horizon
        price_ret = m.mid / self.arrival_price - 1.0
        rel_spread = snap.spread / m.mid if m.mid > 0 else 0.0
        vol = m.recent_volatility()
        imb = snap.imbalance
        depth = snap.depth(5) / self._depth_norm
        obs = np.array(
            [inv_frac, time_frac, price_ret, rel_spread, vol, imb, depth, self.prev_action],
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
        if self.market_source is not None:
            # replay historical / real LOB data instead of the synthetic market
            self.market = self.market_source.make_simulator(rng)
        else:
            self.market = MarketSimulator(self.market_config, rng=rng)
        self.market.reset(horizon=self.horizon)

        self.t = 0
        self.remaining = self.total_inventory
        self.arrival_price = float(self.market.mid)
        self.prev_action = 0.0
        self.executed_shares = 0.0
        self.executed_notional = 0.0
        self.total_commission = 0.0
        self.cum_reward = 0.0
        self.history = []

        return self._build_obs(), {"arrival_price": self.arrival_price}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.market is not None, "Call reset() before step()."
        cfg = self.exec_config
        rcfg = cfg.reward
        m = self.market

        fraction = self._action_to_fraction(action)
        is_last_step = self.t >= self.horizon - 1

        # force liquidation of any leftover on the final step
        if is_last_step and cfg.force_liquidation:
            fraction = 1.0

        shares = min(fraction * self.remaining, self.remaining)
        shares = max(shares, 0.0)

        # latency: market drifts adversely before our order fills
        if cfg.latency_steps > 0 and shares > 0:
            m.apply_latency(self.side, cfg.latency_steps)

        exec_result = m.execute(self.side, shares)
        filled = exec_result.filled_shares

        # ---- cost accounting (all in cash) ----
        # implementation-shortfall contribution of this fill vs the arrival price
        shortfall_cash = self.side.sign * filled * (self.arrival_price - exec_result.avg_price)
        commission = (
            cfg.commission_per_share * filled
            + (cfg.commission_bps / 1e4) * exec_result.notional
        )

        self.remaining = max(self.remaining - filled, 0.0)
        self.executed_shares += filled
        self.executed_notional += exec_result.notional
        self.total_commission += commission

        # ---- reward ----
        norm = self.arrival_price * self.total_inventory
        price_reward = -shortfall_cash / norm
        txn_reward = -commission / norm
        temp_reward = -rcfg.temp_impact_penalty * exec_result.temp_impact_cost / norm
        inv_frac_after = self.remaining / self.total_inventory
        risk_reward = -rcfg.risk_aversion * (m.recent_volatility() * inv_frac_after) ** 2
        reward = (price_reward + txn_reward + temp_reward + risk_reward) * rcfg.reward_scale

        # ---- termination ----
        self.t += 1
        terminated = self.remaining <= 1e-9 or self.t >= self.horizon
        truncated = False

        # penalty for any inventory left unexecuted when not force-liquidating
        unexecuted = 0.0
        if terminated and self.remaining > 1e-9:
            unexecuted = self.remaining
            penalty = rcfg.unexecuted_penalty * (unexecuted / self.total_inventory)
            reward -= penalty * rcfg.reward_scale

        self.cum_reward += reward
        self.prev_action = fraction

        # advance exogenous market dynamics for the next decision (if any)
        if not terminated:
            m.advance()

        record = {
            "t": self.t - 1,
            "action_fraction": fraction,
            "shares": filled,
            "exec_price": exec_result.avg_price,
            "raw_exec_price": exec_result.raw_avg_price,
            "mid_before": exec_result.mid_before,
            "mid_after": exec_result.mid_after,
            "arrival_price": self.arrival_price,
            "slippage": exec_result.slippage,
            "temp_impact_cost": exec_result.temp_impact_cost,
            "perm_impact": exec_result.perm_impact,
            "notional": exec_result.notional,
            "commission": commission,
            "shortfall_cash": shortfall_cash,
            "inventory_after": self.remaining,
            "market_volume": m.market_volume,
            "reward": reward,
        }
        self.history.append(record)

        obs = self._build_obs()
        info: Dict[str, Any] = dict(record)
        if terminated:
            info["episode_summary"] = self._episode_summary(unexecuted)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ summaries
    def _episode_summary(self, unexecuted: float) -> Dict[str, Any]:
        """Aggregate execution-quality statistics for the finished episode."""
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
            "unexecuted_shares": unexecuted,
            "avg_fill_price": avg_fill,
            "arrival_price": self.arrival_price,
            "final_mid": self.market.mid,
            "executed_notional": self.executed_notional,
            "total_commission": self.total_commission,
            "implementation_shortfall_cash": is_cash,
            "implementation_shortfall_bps": is_bps,
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
