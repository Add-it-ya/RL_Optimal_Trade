"""High-level training / persistence helpers shared by the CLI scripts and dashboard.

Builds an action-space-compatible environment for a given algorithm (optionally a fixed
regime or the domain-randomised distribution), trains the agent while logging per-episode
returns, and saves the model plus a JSON sidecar describing how to rebuild it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym

from rl_execution.agents import make_agent, required_action_type
from rl_execution.config import ExecutionConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import get_regime
from rl_execution.experiments.runner import DomainRandomizedEnv
from rl_execution.utils.io import MODELS_DIR, ensure_dir, load_json, save_json

_CUSTOM = {"dqn", "doubledqn", "ddqn"}


def _norm(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


class EpisodeRewardLogger(gym.Wrapper):
    """Records the (undiscounted) return of every completed episode into a list."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards: List[float] = []
        self._acc = 0.0

    def reset(self, **kwargs):
        self._acc = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._acc += float(reward)
        if terminated or truncated:
            self.episode_rewards.append(self._acc)
            self._acc = 0.0
        return obs, reward, terminated, truncated, info


def make_env(
    agent_name: str,
    exec_config: Optional[ExecutionConfig] = None,
    regime: Optional[str] = None,
    randomized: bool = False,
    seed: Optional[int] = None,
) -> ExecutionEnv:
    """Build an :class:`ExecutionEnv` whose action space matches ``agent_name``."""
    exec_config = exec_config or ExecutionConfig()
    ec = replace(exec_config, action_type=required_action_type(agent_name))
    if randomized:
        return DomainRandomizedEnv(ec, seed=seed)
    market_config = get_regime(regime) if regime else None
    from rl_execution.config import MarketConfig

    return ExecutionEnv(market_config or MarketConfig(), ec)


def make_env_factory(
    agent_name: str,
    exec_config: Optional[ExecutionConfig] = None,
    regime: Optional[str] = None,
    randomized: bool = False,
) -> Callable[[], ExecutionEnv]:
    """Return a zero-arg factory for evaluation envs matching ``agent_name``."""

    def factory():
        return make_env(agent_name, exec_config, regime=regime, randomized=randomized)

    return factory


def train_agent(
    agent_name: str,
    total_timesteps: int = 50_000,
    exec_config: Optional[ExecutionConfig] = None,
    regime: Optional[str] = None,
    randomized: bool = True,
    seed: Optional[int] = 0,
    progress: bool = False,
    device: str = "cpu",
    agent_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[float]]:
    """Train an agent; returns ``(agent, episode_reward_log)``.

    ``device`` is "cpu" (recommended for these small MLP policies), "cuda", or "auto".
    """
    env = make_env(agent_name, exec_config, regime=regime, randomized=randomized, seed=seed)
    logged = EpisodeRewardLogger(env)
    kwargs = dict(agent_kwargs or {})
    kwargs.setdefault("device", device)
    agent = make_agent(agent_name, logged, seed=seed, **kwargs)
    agent.train(total_timesteps, progress=progress)
    return agent, logged.episode_rewards


# --------------------------------------------------------------------------- persistence
def save_agent(
    agent,
    agent_name: str,
    tag: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist a trained agent and a JSON sidecar; returns the model path (without ext)."""
    ensure_dir(MODELS_DIR)
    tag = tag or _norm(agent_name)
    backend = "custom" if _norm(agent_name) in _CUSTOM else "sb3"
    ext = ".pt" if backend == "custom" else ".zip"
    model_path = str(MODELS_DIR / f"{tag}{ext}")
    agent.save(model_path if backend == "custom" else str(MODELS_DIR / tag))

    sidecar = {
        "agent_name": agent_name,
        "backend": backend,
        "algo": "dqn" if backend == "sb3" and _norm(agent_name) == "sb3dqn" else _norm(agent_name),
        "action_type": required_action_type(agent_name).value,
    }
    if meta:
        sidecar.update(meta)
    save_json(sidecar, str(MODELS_DIR / f"{tag}.json"))
    return str(MODELS_DIR / tag)


def load_agent(tag: str, env) -> Any:
    """Rebuild a saved agent bound to ``env`` from its JSON sidecar."""
    meta = load_json(str(MODELS_DIR / f"{tag}.json"))
    backend = meta["backend"]
    if backend == "custom":
        from rl_execution.agents.dqn import DQNAgent

        return DQNAgent.load(str(MODELS_DIR / f"{tag}.pt"), env=env)
    from rl_execution.agents.sb3_agents import SB3Agent

    return SB3Agent.load(str(MODELS_DIR / f"{tag}.zip"), env=env, algo=meta["algo"])
