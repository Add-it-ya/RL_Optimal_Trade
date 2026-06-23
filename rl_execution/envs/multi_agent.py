"""Multi-agent execution simulation.

Several traders each work their own parent order simultaneously against a *shared* market.
Within each step the participants' market orders are executed sequentially in randomised
order against the common book, so each trader's permanent impact moves the mid that the
others subsequently face -- capturing competition / crowding for liquidity.

Participants may be classical baselines or trained RL agents (anything exposing the
:class:`~rl_execution.baselines.base.BaseStrategy` interface).  This is a lightweight
simulator (not a PettingZoo env); it is intended for studying interaction effects rather
than for training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from rl_execution.config import ActionType, MarketConfig, Side
from rl_execution.envs.market import MarketSimulator


@dataclass
class Participant:
    """One trader in the multi-agent simulation."""

    name: str
    strategy: Any
    side: Side = Side.SELL
    total_inventory: float = 10_000.0
    action_type: ActionType = ActionType.CONTINUOUS
    n_discrete_actions: int = 11

    # runtime state
    remaining: float = field(init=False, default=0.0)
    arrival_price: float = field(init=False, default=0.0)
    executed_notional: float = field(init=False, default=0.0)
    executed_shares: float = field(init=False, default=0.0)
    history: List[Dict[str, Any]] = field(init=False, default_factory=list)


class _AgentView:
    """Minimal env-like view passed to baseline strategies bound to a participant."""

    def __init__(
        self,
        participant: Participant,
        market: MarketSimulator,
        horizon: int,
        market_config: MarketConfig,
    ):
        self.participant = participant
        self.market = market
        self.horizon = horizon
        self.market_config = market_config
        self.t = 0

    @property
    def total_inventory(self) -> float:
        return self.participant.total_inventory

    @property
    def remaining(self) -> float:
        return self.participant.remaining


class MultiAgentSimulator:
    """Simulate several execution strategies competing on one shared market."""

    def __init__(self, market_config: Optional[MarketConfig] = None, horizon: int = 20):
        self.market_config = market_config or MarketConfig()
        self.horizon = int(horizon)
        self.participants: List[Participant] = []

    def add_participant(self, participant: Participant) -> None:
        self.participants.append(participant)

    # ------------------------------------------------------------------ obs
    def _build_obs(self, p: Participant, market: MarketSimulator, t: int) -> np.ndarray:
        snap = market.snapshot
        inv_frac = p.remaining / p.total_inventory if p.total_inventory > 0 else 0.0
        time_frac = (self.horizon - t) / self.horizon
        price_ret = market.mid / p.arrival_price - 1.0 if p.arrival_price else 0.0
        rel_spread = snap.spread / market.mid if market.mid > 0 else 0.0
        depth_norm = max(self.market_config.base_depth * 5.0, 1.0)
        prev_action = p.history[-1]["action_fraction"] if p.history else 0.0
        obs = np.array(
            [
                inv_frac,
                time_frac,
                price_ret,
                rel_spread,
                market.recent_volatility(),
                snap.imbalance,
                snap.depth(5) / depth_norm,
                prev_action,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, [0, 0, -1, 0, 0, -1, 0, 0], [1, 1, 1, 1, 1, 1, 100, 1])

    def _fraction(
        self, p: Participant, view: _AgentView, obs: np.ndarray, last_step: bool
    ) -> float:
        strat = p.strategy
        if last_step:
            return 1.0
        if hasattr(strat, "_decide_fraction"):  # classical baseline
            return float(np.clip(strat._decide_fraction(obs, {}), 0.0, 1.0))
        if hasattr(strat, "agent"):  # AgentStrategy wrapper
            action = strat.agent.predict(obs, deterministic=True)
            if p.action_type is ActionType.DISCRETE:
                grid = np.linspace(0.0, 1.0, p.n_discrete_actions)
                return float(grid[int(np.asarray(action).reshape(-1)[0])])
            return float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], 0.0, 1.0))
        if callable(strat):  # bare fraction function
            return float(np.clip(strat(obs), 0.0, 1.0))
        raise TypeError(f"Unsupported strategy type for participant {p.name}")

    # ------------------------------------------------------------------ run
    def run(self, seed: Optional[int] = None) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        market = MarketSimulator(self.market_config, rng=rng)
        market.reset(horizon=self.horizon)
        arrival = float(market.mid)

        views = {}
        for p in self.participants:
            p.remaining = p.total_inventory
            p.arrival_price = arrival
            p.executed_notional = 0.0
            p.executed_shares = 0.0
            p.history = []
            v = _AgentView(p, market, self.horizon, self.market_config)
            if hasattr(p.strategy, "reset"):
                p.strategy.reset(v)
            views[p.name] = v

        for t in range(self.horizon):
            last_step = t == self.horizon - 1
            order = list(range(len(self.participants)))
            rng.shuffle(order)  # randomise execution priority each step
            for j in order:
                p = self.participants[j]
                if p.remaining <= 1e-9:
                    continue
                views[p.name].t = t
                obs = self._build_obs(p, market, t)
                frac = self._fraction(p, views[p.name], obs, last_step)
                shares = min(frac * p.remaining, p.remaining)
                res = market.execute(p.side, shares)
                p.remaining = max(p.remaining - res.filled_shares, 0.0)
                p.executed_shares += res.filled_shares
                p.executed_notional += res.notional
                p.history.append(
                    {
                        "t": t,
                        "action_fraction": frac,
                        "shares": res.filled_shares,
                        "exec_price": res.avg_price,
                        "mid": market.mid,
                        "inventory_after": p.remaining,
                    }
                )
            market.advance()

        return self._summaries(arrival)

    def _summaries(self, arrival: float) -> Dict[str, Any]:
        rows = []
        per_agent = {}
        for p in self.participants:
            avg_fill = p.executed_notional / p.executed_shares if p.executed_shares > 0 else arrival
            is_bps = p.side.sign * (arrival - avg_fill) / arrival * 1e4
            rows.append(
                {
                    "agent": p.name,
                    "side": p.side.value,
                    "executed_shares": p.executed_shares,
                    "avg_fill_price": avg_fill,
                    "arrival_price": arrival,
                    "implementation_shortfall_bps": is_bps,
                }
            )
            per_agent[p.name] = pd.DataFrame(p.history)
        return {"table": pd.DataFrame(rows), "histories": per_agent, "arrival_price": arrival}
