"""Common agent interface and a strategy adapter for the backtest engine.

Every agent (custom or Stable-Baselines3-backed) exposes::

    agent.train(total_timesteps)          -> self
    agent.predict(obs, deterministic)     -> action (in the env's action space)
    agent.save(path) / Agent.load(path, env)

:class:`AgentStrategy` wraps any such agent so it satisfies the
:class:`~rl_execution.baselines.base.BaseStrategy` interface and can be dropped into the
backtest engine alongside the classical baselines.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rl_execution.baselines.base import BaseStrategy


def resolve_device(device: str = "cpu") -> str:
    """Resolve a device string.

    ``"auto"`` -> ``"cuda"`` if a CUDA build of torch sees a GPU, else ``"cpu"``.
    NB: for the small MLP policies used here the environment step (CPU) dominates, so CPU
    is usually as fast or faster than GPU; ``"cpu"`` is the recommended default.
    """
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class BaseAgent:
    """Abstract model-free RL agent."""

    name: str = "agent"

    def train(self, total_timesteps: int, **kwargs) -> "BaseAgent":
        raise NotImplementedError

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, env=None) -> "BaseAgent":
        raise NotImplementedError


class AgentStrategy(BaseStrategy):
    """Adapt a trained :class:`BaseAgent` to the backtest strategy interface."""

    def __init__(self, agent: BaseAgent, name: str | None = None, deterministic: bool = True):
        self.agent = agent
        self.deterministic = deterministic
        self.name = name or getattr(agent, "name", agent.__class__.__name__)

    def reset(self, env) -> None:
        super().reset(env)

    def act(self, obs: np.ndarray, info: Dict[str, Any]):
        # the agent already emits actions in the env's action space
        return self.agent.predict(np.asarray(obs, dtype=np.float32), deterministic=self.deterministic)
