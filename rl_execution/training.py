"""High-level training / persistence helpers shared by the CLI scripts and dashboard.

Builds an action-space-compatible environment for a given algorithm (optionally a fixed
regime or the domain-randomised distribution), trains the agent while logging per-episode
returns, and saves the model plus a JSON sidecar describing how to rebuild it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym

from rl_execution.agents import make_agent, required_action_type
from rl_execution.config import ExecutionConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import get_regime
from rl_execution.experiments.runner import DomainRandomizedEnv
from rl_execution.utils.io import MODELS_DIR, ensure_dir, load_json, save_json
from rl_execution.utils.provenance import capture_provenance
from rl_execution.utils.seeding import set_global_seeds

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
    if seed is not None:
        # Seed every framework RNG (python/numpy/torch/sb3) for reproducible training.
        set_global_seeds(seed)
    env = make_env(agent_name, exec_config, regime=regime, randomized=randomized, seed=seed)
    logged = EpisodeRewardLogger(env)
    if seed is not None:
        # Seed the env's market RNG deterministically: the agents' training loops call
        # ``reset()`` without a seed, which would otherwise lazily initialise Gymnasium's
        # np_random from OS entropy. Seeding once here makes every subsequent (unseeded)
        # episode reset -- and hence the whole price-path sequence -- reproducible.
        logged.reset(seed=seed)
    kwargs = dict(agent_kwargs or {})
    kwargs.setdefault("device", device)
    agent = make_agent(agent_name, logged, seed=seed, **kwargs)
    agent.train(total_timesteps, progress=progress)
    return agent, logged.episode_rewards


# --------------------------------------------------------------------------- persistence
def model_artifact_path(agent_name: str, tag: Optional[str] = None) -> Path:
    """The on-disk model file (with extension) that :func:`save_agent` writes for an agent."""
    tag = tag or _norm(agent_name)
    ext = ".pt" if _norm(agent_name) in _CUSTOM else ".zip"
    return MODELS_DIR / f"{tag}{ext}"


def save_agent(
    agent,
    agent_name: str,
    tag: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    *,
    seed: Optional[int] = None,
    config: Any = None,
    provenance: bool = True,
) -> str:
    """Persist a trained agent and a JSON sidecar; returns the model path (without ext).

    When ``provenance`` is true the sidecar is stamped with a lineage block
    ({git_sha, config_hash, lib_versions, seed, ...}) so the artifact is self-describing
    and auditable.  ``config`` (e.g. a :class:`RunConfig`) is hashed into ``config_hash``.
    """
    ensure_dir(MODELS_DIR)
    tag = tag or _norm(agent_name)
    backend = "custom" if _norm(agent_name) in _CUSTOM else "sb3"
    ext = ".pt" if backend == "custom" else ".zip"
    model_path = str(MODELS_DIR / f"{tag}{ext}")
    agent.save(model_path if backend == "custom" else str(MODELS_DIR / tag))

    sidecar: Dict[str, Any] = {
        "agent_name": agent_name,
        "backend": backend,
        "algo": "dqn" if backend == "sb3" and _norm(agent_name) == "sb3dqn" else _norm(agent_name),
        "action_type": required_action_type(agent_name).value,
    }
    if meta:
        sidecar.update(meta)
    if provenance:
        sidecar["provenance"] = capture_provenance(seed=seed, config=config)
    save_json(sidecar, str(MODELS_DIR / f"{tag}.json"))
    return str(MODELS_DIR / tag)


def load_agent(tag: str, env, models_dir: Union[str, Path] = MODELS_DIR) -> Any:
    """Rebuild a saved agent bound to ``env`` from its JSON sidecar.

    ``models_dir`` defaults to the project ``models/`` directory but can point elsewhere
    (e.g. a temp dir produced by the model registry's ``load_registered``).
    """
    models_dir = Path(models_dir)
    meta = load_json(str(models_dir / f"{tag}.json"))
    backend = meta["backend"]
    if backend == "custom":
        from rl_execution.agents.dqn import DQNAgent

        return DQNAgent.load(str(models_dir / f"{tag}.pt"), env=env)
    from rl_execution.agents.sb3_agents import SB3Agent

    return SB3Agent.load(str(models_dir / f"{tag}.zip"), env=env, algo=meta["algo"])
