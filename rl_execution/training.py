"""Training and run-folder persistence.

A *run* is one (agent, seed) training job, and every run lives in its own folder::

    runs/<experiment>/<agent>/seed_<k>/
        config.yaml        the RunConfig that produced it (enough to retrain it)
        model.zip | .pt    trained weights
        provenance.json    git SHA, config hash, library versions and the weights' SHA-256
        train_log.csv      undiscounted return of every training episode

:func:`load_run` re-hashes the weights and refuses to load them if they do not match
``provenance.json``, so a model file can never silently be swapped for another run's.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym

from rl_execution.agents import (
    DISPLAY_NAMES,
    load_agent_file,
    make_agent,
    normalize_agent_name,
    required_action_type,
)
from rl_execution.config import ExecutionConfig, MarketConfig, RunConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import get_regime
from rl_execution.experiments.runner import DomainRandomizedEnv
from rl_execution.utils.config_io import dump_run_config, load_run_config
from rl_execution.utils.io import ensure_dir, load_json, save_json
from rl_execution.utils.provenance import capture_provenance, file_sha256
from rl_execution.utils.seeding import set_global_seeds

PathLike = Union[str, Path]

#: Weight file written for each agent inside its run folder.
MODEL_FILES = {"ppo": "model.zip", "dqn": "model.pt", "doubledqn": "model.pt"}


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


# --------------------------------------------------------------------------- environments
def make_env(
    agent_name: str,
    exec_config: Optional[ExecutionConfig] = None,
    regime: Optional[str] = None,
    randomized: bool = False,
    seed: Optional[int] = None,
    market_config: Optional[MarketConfig] = None,
) -> ExecutionEnv:
    """Build an :class:`ExecutionEnv` whose action space matches ``agent_name``.

    The market is the domain-randomised family when ``randomized``, else the named
    ``regime``, else ``market_config`` (default :class:`MarketConfig`).
    """
    ec = replace(exec_config or ExecutionConfig(), action_type=required_action_type(agent_name))
    if randomized:
        return DomainRandomizedEnv(ec, seed=seed)
    if regime:
        return ExecutionEnv(get_regime(regime), ec)
    return ExecutionEnv(market_config or MarketConfig(), ec)


def make_env_factory(
    agent_name: str,
    exec_config: Optional[ExecutionConfig] = None,
    regime: Optional[str] = None,
    randomized: bool = False,
    market_config: Optional[MarketConfig] = None,
) -> Callable[[], ExecutionEnv]:
    """Return a zero-arg factory for evaluation envs matching ``agent_name``."""

    def factory():
        return make_env(
            agent_name,
            exec_config,
            regime=regime,
            randomized=randomized,
            market_config=market_config,
        )

    return factory


# --------------------------------------------------------------------------- training
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
    market_config: Optional[MarketConfig] = None,
) -> Tuple[Any, List[float]]:
    """Train an agent; returns ``(agent, episode_reward_log)``.

    ``device`` is "cpu" (recommended for these small MLP policies), "cuda", or "auto".
    """
    if seed is not None:
        # Seed every framework RNG (python/numpy/torch/sb3) for reproducible training.
        set_global_seeds(seed)
    env = make_env(
        agent_name,
        exec_config,
        regime=regime,
        randomized=randomized,
        seed=seed,
        market_config=market_config,
    )
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


def train_run(rc: RunConfig, progress: bool = False) -> Tuple[Any, List[float]]:
    """Train the agent described by a :class:`RunConfig`."""
    return train_agent(
        rc.agent,
        total_timesteps=rc.timesteps,
        exec_config=rc.execution,
        regime=rc.regime,
        randomized=rc.randomized,
        seed=rc.seed,
        progress=progress,
        device=rc.device,
        agent_kwargs=rc.agent_kwargs or None,
        market_config=rc.market,
    )


def env_factory_for(rc: RunConfig) -> Callable[[], ExecutionEnv]:
    """Evaluation-env factory matching the market a run was trained on."""
    return make_env_factory(
        rc.agent, rc.execution, regime=rc.regime, randomized=rc.randomized, market_config=rc.market
    )


# --------------------------------------------------------------------------- run folders
def run_dir(root: PathLike, experiment: str, agent_name: str, seed: int) -> Path:
    """``<root>/<experiment>/<Agent>/seed_<seed>``."""
    agent = DISPLAY_NAMES[normalize_agent_name(agent_name)]
    return Path(root) / experiment / agent / f"seed_{int(seed)}"


def save_run(
    agent: Any,
    rc: RunConfig,
    directory: PathLike,
    train_log: Optional[List[float]] = None,
) -> Path:
    """Write a trained agent and everything needed to audit it into ``directory``."""
    out = ensure_dir(directory)
    model_path = out / MODEL_FILES[normalize_agent_name(rc.agent)]
    agent.save(str(model_path))
    dump_run_config(rc, str(out / "config.yaml"))

    provenance = capture_provenance(seed=rc.seed, config=rc)
    provenance["model_file"] = model_path.name
    provenance["weights_sha256"] = file_sha256(model_path)
    save_json(provenance, out / "provenance.json")

    with open(out / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return"])
        writer.writerows(enumerate(train_log or []))
    return out


def load_run(directory: PathLike, env=None, verify: bool = True) -> Tuple[Any, RunConfig]:
    """Load ``(agent, run_config)`` from a run folder.

    With ``verify`` (the default) the weights are re-hashed and compared with the SHA-256
    recorded at save time; a mismatch raises ``ValueError`` instead of loading a model that
    did not produce this folder's config.
    """
    directory = Path(directory)
    rc = load_run_config(str(directory / "config.yaml"), validate=False)
    provenance = load_json(directory / "provenance.json")
    model_path = directory / provenance["model_file"]
    if verify:
        actual = file_sha256(model_path)
        if actual != provenance["weights_sha256"]:
            raise ValueError(
                f"Weights at {model_path} do not match provenance.json "
                f"(sha256 {actual[:12]}... != {provenance['weights_sha256'][:12]}...)."
            )
    if env is None:
        env = env_factory_for(rc)()
    return load_agent_file(rc.agent, str(model_path), env=env), rc


def find_runs(root: PathLike) -> List[Path]:
    """Every run folder (one containing ``provenance.json``) under ``root``, sorted."""
    return sorted(p.parent for p in Path(root).rglob("provenance.json"))
