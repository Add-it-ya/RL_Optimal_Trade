"""Model-free RL agents and the strategy adapter.

Two agents are supported: PPO (through Stable-Baselines3) and DQN / Double DQN
(implemented from scratch in PyTorch).  A third, ``vecppo``, is PPO trained on the batched
torch environment (:mod:`rl_execution.vec`, the GPU path); it is trained by
``rl_execution.vec.train`` and loaded here like the others.  Torch / SB3 are imported
lazily so the rest of the package works without a deep-learning backend.

Each algorithm requires a particular action-space parametrisation, exposed via
:data:`ALGO_ACTION_TYPE` so callers can build a compatible :class:`ExecutionEnv`.
"""

from __future__ import annotations

from typing import Optional

from rl_execution.agents.base import AgentStrategy, BaseAgent
from rl_execution.config import ActionType

# value-based agents need a discrete action grid; PPO uses the continuous one
ALGO_ACTION_TYPE = {
    "dqn": ActionType.DISCRETE,
    "doubledqn": ActionType.DISCRETE,
    "ppo": ActionType.CONTINUOUS,
    "vecppo": ActionType.CONTINUOUS,
}

#: Canonical display name of each agent, used for run folders and result tables.
DISPLAY_NAMES = {"dqn": "DQN", "doubledqn": "DoubleDQN", "ppo": "PPO", "vecppo": "VecPPO"}

ALL_AGENTS = list(DISPLAY_NAMES.values())


def normalize_agent_name(name: str) -> str:
    """Canonical lower-case key: ``"Double-DQN"`` / ``"ddqn"`` -> ``"doubledqn"``."""
    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    key = "doubledqn" if key == "ddqn" else key
    if key not in ALGO_ACTION_TYPE:
        raise KeyError(f"Unknown agent '{name}'. Available: {sorted(ALGO_ACTION_TYPE)}")
    return key


def required_action_type(name: str) -> ActionType:
    return ALGO_ACTION_TYPE[normalize_agent_name(name)]


def make_agent(name: str, env, seed: Optional[int] = None, **kwargs) -> BaseAgent:
    """Factory: build an agent by name (``ppo``, ``dqn``, ``doubledqn``) bound to ``env``."""
    key = normalize_agent_name(name)
    if key == "ppo":
        from rl_execution.agents.ppo import PPOAgent

        return PPOAgent(env=env, seed=seed, **kwargs)
    if key == "vecppo":
        raise ValueError("vecppo trains on the batched env; use rl_execution.vec.train")

    from rl_execution.agents.dqn import DQNAgent

    return DQNAgent(env, double=key == "doubledqn", seed=seed, **kwargs)


def load_agent_file(name: str, path: str, env=None) -> BaseAgent:
    """Load a saved agent of type ``name`` from ``path``."""
    key = normalize_agent_name(name)
    if key == "ppo":
        from rl_execution.agents.ppo import PPOAgent

        return PPOAgent.load(path, env=env)
    if key == "vecppo":
        from rl_execution.vec.ppo import TorchPolicyAgent

        return TorchPolicyAgent.load(path)

    from rl_execution.agents.dqn import DQNAgent

    return DQNAgent.load(path, env=env)


__all__ = [
    "BaseAgent",
    "AgentStrategy",
    "make_agent",
    "load_agent_file",
    "normalize_agent_name",
    "required_action_type",
    "ALGO_ACTION_TYPE",
    "DISPLAY_NAMES",
    "ALL_AGENTS",
]
