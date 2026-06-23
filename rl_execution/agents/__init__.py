"""Model-free RL agents and the strategy adapter.

Two custom agents (DQN, Double DQN) are implemented from scratch in PyTorch; PPO, A2C, SAC
and SB3's DQN are provided through Stable-Baselines3.  Torch / SB3 are imported lazily so
the rest of the package works without a deep-learning backend.

Each algorithm requires a particular action-space parametrisation, exposed via
:data:`ALGO_ACTION_TYPE` so callers can build a compatible :class:`ExecutionEnv`.
"""

from __future__ import annotations

from typing import Optional

from rl_execution.agents.base import AgentStrategy, BaseAgent
from rl_execution.config import ActionType

# value-based agents need a discrete action grid; policy/actor-critic use continuous
ALGO_ACTION_TYPE = {
    "dqn": ActionType.DISCRETE,
    "doubledqn": ActionType.DISCRETE,
    "ddqn": ActionType.DISCRETE,
    "sb3dqn": ActionType.DISCRETE,
    "ppo": ActionType.CONTINUOUS,
    "a2c": ActionType.CONTINUOUS,
    "sac": ActionType.CONTINUOUS,
}

ALL_AGENTS = ["DQN", "DoubleDQN", "PPO", "A2C", "SAC"]


def _normalize(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def required_action_type(name: str) -> ActionType:
    key = _normalize(name)
    if key not in ALGO_ACTION_TYPE:
        raise KeyError(f"Unknown agent '{name}'. Available: {sorted(ALGO_ACTION_TYPE)}")
    return ALGO_ACTION_TYPE[key]


def make_agent(name: str, env, seed: Optional[int] = None, **kwargs) -> BaseAgent:
    """Factory: build an agent by name, bound to ``env``.

    Names (case-insensitive): ``dqn``, ``doubledqn`` (``ddqn``), ``ppo``, ``a2c``, ``sac``,
    ``sb3dqn``.  The custom value-based agents (``dqn`` / ``doubledqn``) use the from-scratch
    PyTorch implementation; the others use Stable-Baselines3.
    """
    key = _normalize(name)
    if key in ("dqn", "doubledqn", "ddqn"):
        from rl_execution.agents.dqn import DQNAgent

        return DQNAgent(env, double=key in ("doubledqn", "ddqn"), seed=seed, **kwargs)

    if key in ("ppo", "a2c", "sac", "sb3dqn"):
        from rl_execution.agents.sb3_agents import SB3Agent

        algo = "dqn" if key == "sb3dqn" else key
        return SB3Agent(algo, env=env, seed=seed, **kwargs)

    raise KeyError(f"Unknown agent '{name}'. Available: {sorted(ALGO_ACTION_TYPE)}")


__all__ = [
    "BaseAgent",
    "AgentStrategy",
    "make_agent",
    "required_action_type",
    "ALGO_ACTION_TYPE",
    "ALL_AGENTS",
]
