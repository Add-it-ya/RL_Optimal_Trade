"""Training, checkpoint selection and run-folder persistence.

A *run* is one (agent, seed) training job, and every run lives in its own folder::

    runs/<experiment>/<agent>/seed_<k>/
        config.yaml        the RunConfig that produced it (enough to retrain it)
        model.zip | .pt    the selected weights
        provenance.json    git SHA, config hash, library versions, the weights' SHA-256 and
                           which checkpoint was selected
        train_log.csv      every training episode: env steps so far, return, objective
        val_curve.csv      the validation objective at every checkpoint

**Checkpoint selection.**  Every ``eval_every`` environment steps (and once before training)
the deterministic policy is scored on a fixed block of *validation* seeds, and the weights
with the lowest mean objective are the ones saved.  The block is disjoint from the test
seeds ``rlx evaluate`` reports on, so selection never peeks at the test set.  Scoring the
untrained policy too means a run that never improves keeps its starting point -- and
``val_curve.csv`` shows that it did not improve.

:func:`load_run` re-hashes the weights and refuses to load them if they do not match
``provenance.json``, so a model file can never silently be swapped for another run's.
"""

from __future__ import annotations

import csv
import functools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym

from rl_execution.agents import (
    DISPLAY_NAMES,
    AgentStrategy,
    load_agent_file,
    make_agent,
    normalize_agent_name,
    required_action_type,
)
from rl_execution.config import ExecutionConfig, MarketConfig, RunConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments.regimes import get_regime
from rl_execution.experiments.runner import VALIDATION_BASE_SEED, DomainRandomizedEnv
from rl_execution.utils.config_io import dump_run_config, load_run_config
from rl_execution.utils.io import ensure_dir, load_json, save_json
from rl_execution.utils.provenance import capture_provenance, file_sha256
from rl_execution.utils.seeding import set_global_seeds

PathLike = Union[str, Path]

#: Weight file written for each agent inside its run folder.
MODEL_FILES = {"ppo": "model.zip", "dqn": "model.pt", "doubledqn": "model.pt", "vecppo": "model.pt"}

#: Training environments of seed ``s`` are seeded from ``s * ENV_SEED_STRIDE``, so the
#: sub-environments of different training seeds never share a seed (and, for ``s`` below
#: 1000, never reach the validation block at 1,000,000).
ENV_SEED_STRIDE = 1_000


class EpisodeLogger(gym.Wrapper):
    """Records every finished episode as ``(env_steps_so_far, return, objective_bps)``."""

    def __init__(self, env):
        super().__init__(env)
        self.episodes: List[Tuple[int, float, float]] = []
        self._return = 0.0
        self._steps = 0

    def reset(self, **kwargs):
        self._return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._return += float(reward)
        self._steps += 1
        if terminated or truncated:
            objective = float(info["episode_summary"]["objective_bps"])
            self.episodes.append((self._steps, self._return, objective))
            self._return = 0.0
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

    The market is the domain-randomised family when ``randomized`` (``seed`` fixes the
    sequence of sampled markets), else the named ``regime``, else ``market_config``.
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
    seed: Optional[int] = None,
) -> Callable[[], ExecutionEnv]:
    """Return a zero-arg factory for evaluation envs matching ``agent_name``."""

    def factory():
        return make_env(
            agent_name,
            exec_config,
            regime=regime,
            randomized=randomized,
            seed=seed,
            market_config=market_config,
        )

    return factory


def _logged_env(*args, **kwargs) -> EpisodeLogger:
    return EpisodeLogger(make_env(*args, **kwargs))


# --------------------------------------------------------------------------- selection
class CheckpointSelector:
    """Scores the policy on validation seeds every ``every`` steps; keeps the best weights.

    Call it with the number of env steps taken so far (the agents' training callback does);
    :meth:`finish` scores the final policy and restores the best weights into the agent.
    """

    def __init__(
        self,
        agent: Any,
        env_factory: Callable[[], ExecutionEnv],
        every: int,
        n_episodes: int,
        base_seed: int = VALIDATION_BASE_SEED,
        antithetic: bool = True,
    ) -> None:
        self.agent = agent
        self.env_factory = env_factory
        self.every = int(every)
        self.n_episodes = int(n_episodes)
        self.base_seed = int(base_seed)
        self.antithetic = antithetic
        self.next_step = self.every
        self.curve: List[Dict[str, float]] = []
        self.best_objective = float("inf")
        self.best_step: Optional[int] = None
        self._best_weights: Any = None

    def score(self, step: int) -> Dict[str, float]:
        from rl_execution.backtest import evaluate

        res = evaluate(
            self.env_factory,
            AgentStrategy(self.agent),
            n_episodes=self.n_episodes,
            base_seed=self.base_seed,
            antithetic=self.antithetic,
        )
        agg = res.aggregate
        row = {
            "step": int(step),
            "objective_bps": agg["objective_bps_mean"],
            "implementation_shortfall_bps": agg["implementation_shortfall_bps_mean"],
            "effective_steps": agg["effective_steps_mean"],
        }
        self.curve.append(row)
        if row["objective_bps"] < self.best_objective:
            self.best_objective = row["objective_bps"]
            self.best_step = int(step)
            self._best_weights = self.agent.get_weights()
        return row

    def __call__(self, step: int) -> None:
        if step >= self.next_step:
            self.score(step)
            self.next_step = (step // self.every + 1) * self.every

    def finish(self, step: int) -> None:
        if not self.curve or self.curve[-1]["step"] != int(step):
            self.score(step)
        self.agent.set_weights(self._best_weights)


# --------------------------------------------------------------------------- training
@dataclass
class TrainingResult:
    """A trained agent plus everything logged while training it."""

    agent: Any
    episodes: List[Dict[str, float]] = field(default_factory=list)
    validation: List[Dict[str, float]] = field(default_factory=list)
    selected_step: Optional[int] = None

    def selected(self) -> Optional[Dict[str, float]]:
        """The validation row of the kept checkpoint (``None`` without selection)."""
        return next((r for r in self.validation if r["step"] == self.selected_step), None)


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
    n_envs: int = 1,
    eval_every: Optional[int] = None,
    val_episodes: int = 1_000,
    val_base_seed: int = VALIDATION_BASE_SEED,
) -> TrainingResult:
    """Train an agent, optionally selecting the best checkpoint on validation seeds.

    PPO runs on a ``DummyVecEnv`` of ``n_envs`` environments; the DQN agents use one.
    ``device`` is "cpu" (recommended for these small MLP policies), "cuda", or "auto".
    With ``eval_every`` set, the returned agent carries the selected weights.
    """
    if seed is not None:
        # Seed every framework RNG (python/numpy/torch/sb3) for reproducible training.
        set_global_seeds(seed)
    key = normalize_agent_name(agent_name)
    env_seed = None if seed is None else int(seed) * ENV_SEED_STRIDE
    build = functools.partial(
        _logged_env,
        agent_name,
        exec_config,
        regime=regime,
        randomized=randomized,
        market_config=market_config,
    )

    if key == "ppo":
        from stable_baselines3.common.vec_env import DummyVecEnv

        n = max(int(n_envs), 1)
        env: Any = DummyVecEnv(
            [
                functools.partial(build, seed=None if env_seed is None else env_seed + i)
                for i in range(n)
            ]
        )
    else:
        n = 1
        env = build(seed=env_seed)
        if env_seed is not None:
            # The training loop resets without a seed; seeding once here makes the whole
            # sequence of training episodes reproducible.
            env.reset(seed=env_seed)

    kwargs = dict(agent_kwargs or {})
    kwargs.setdefault("device", device)
    agent = make_agent(agent_name, env, seed=seed, **kwargs)
    if key == "ppo" and env_seed is not None:
        # SB3 seeds sub-environment i with ``seed + i``, which would make sub-env 1 of seed
        # 0 replay sub-env 0 of seed 1; spread the training seeds apart instead.
        env.seed(env_seed)

    selector = None
    if eval_every:
        val_factory = make_env_factory(
            agent_name,
            exec_config,
            regime=regime,
            randomized=randomized,
            market_config=market_config,
            seed=val_base_seed,
        )
        selector = CheckpointSelector(
            agent,
            val_factory,
            every=eval_every,
            n_episodes=val_episodes,
            base_seed=val_base_seed,
            # antithetic pairs share a market; a randomised env resamples it every reset
            antithetic=not randomized,
        )
        selector.score(0)

    agent.train(total_timesteps, progress=progress, callback=selector)
    if selector is not None:
        selector.finish(total_timesteps)

    logs = env.get_attr("episodes") if key == "ppo" else [env.episodes]
    episodes = sorted(
        (
            {"step": steps * n, "return": ret, "objective_bps": obj}
            for per_env in logs
            for steps, ret, obj in per_env
        ),
        key=lambda row: row["step"],
    )
    return TrainingResult(
        agent=agent,
        episodes=episodes,
        validation=selector.curve if selector else [],
        selected_step=selector.best_step if selector else None,
    )


def train_run(rc: RunConfig, progress: bool = False) -> TrainingResult:
    """Train the agent described by a :class:`RunConfig`, with checkpoint selection."""
    if normalize_agent_name(rc.agent) == "vecppo":
        from rl_execution.vec.train import train_vec_run

        return train_vec_run(rc, progress=progress)
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
        n_envs=rc.n_envs,
        eval_every=rc.eval_every,
        val_episodes=rc.val_episodes,
        val_base_seed=rc.val_base_seed,
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


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: row.get(c) for c in columns} for row in rows)


def save_run(result: TrainingResult, rc: RunConfig, directory: PathLike) -> Path:
    """Write a trained agent and everything needed to audit it into ``directory``."""
    out = ensure_dir(directory)
    model_path = out / MODEL_FILES[normalize_agent_name(rc.agent)]
    result.agent.save(str(model_path))
    dump_run_config(rc, str(out / "config.yaml"))

    provenance = capture_provenance(seed=rc.seed, config=rc)
    provenance["model_file"] = model_path.name
    provenance["weights_sha256"] = file_sha256(model_path)
    selected = result.selected()
    if selected is not None:
        provenance["selected_step"] = result.selected_step
        provenance["selected_val_objective_bps"] = selected["objective_bps"]
        provenance["initial_val_objective_bps"] = result.validation[0]["objective_bps"]
        provenance["final_val_objective_bps"] = result.validation[-1]["objective_bps"]
    save_json(provenance, out / "provenance.json")

    episodes = [{"episode": i, **row} for i, row in enumerate(result.episodes)]
    _write_csv(out / "train_log.csv", episodes, ["episode", "step", "return", "objective_bps"])
    if result.validation:
        _write_csv(
            out / "val_curve.csv",
            result.validation,
            ["step", "objective_bps", "implementation_shortfall_bps", "effective_steps"],
        )
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
