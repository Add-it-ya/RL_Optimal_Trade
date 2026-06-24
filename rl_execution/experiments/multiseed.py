"""Multi-seed training/evaluation with across-seed confidence intervals (Step 2).

The honest unit of analysis for "this method beats the benchmark" is the **training seed**,
not the episode: thousands of evaluation episodes of *one* trained policy are pseudo-replicates
of a single draw, and a CI built over them dramatically understates uncertainty.  This module
trains (or loads from the registry) one policy per seed, evaluates each across regimes with
common-random-number pairing, and reports two clearly-labelled uncertainties:

1. **across-seed CI** (the headline) -- bootstrap CI of the mean per-seed ``vs_<bench>`` across
   seeds; this is what guards against pseudo-replication.
2. **within-run episode CI** (secondary) -- paired bootstrap over episodes for a single seed,
   showing episode-level noise.

Already-trained ``{config_hash, seed}`` combinations are loaded from the MLflow model registry
instead of being retrained, so re-running an analysis (or resuming after a crash) is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from rl_execution.agents import AgentStrategy, required_action_type
from rl_execution.backtest.engine import episode_is
from rl_execution.config import RunConfig
from rl_execution.experiments.regimes import list_regimes
from rl_execution.experiments.runner import build_baselines, evaluate_across_regimes
from rl_execution.metrics.stats import bootstrap_ci, paired_bootstrap_ci
from rl_execution.training import make_env, model_artifact_path, save_agent, train_agent
from rl_execution.utils.provenance import config_hash, git_sha

DISPLAY = {"dqn": "DQN", "doubledqn": "DoubleDQN", "ppo": "PPO", "a2c": "A2C", "sac": "SAC"}


def _norm(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _display(agent_name: str) -> str:
    return DISPLAY.get(_norm(agent_name), agent_name.upper())


# --------------------------------------------------------------------------- registry caching
def _find_cached_version(name: str, cfg_hash: str, seed: int) -> Optional[str]:
    """Highest registered version of model ``name`` tagged with this ``{config_hash, seed}``.

    Returns ``None`` when MLflow is unavailable or nothing matches -- caching is a pure
    optimisation, so any failure simply falls through to (re)training.
    """
    try:
        from mlflow.tracking import MlflowClient

        from rl_execution.tracking.registry import configure_mlflow

        configure_mlflow()
        versions = MlflowClient().search_model_versions(f"name='{name}'")
    except Exception:
        return None

    best: Optional[int] = None
    for v in versions:
        tags = getattr(v, "tags", None) or {}
        if tags.get("config_hash") == str(cfg_hash) and str(tags.get("seed")) == str(seed):
            num = int(v.version)
            best = num if best is None or num > best else best
    return str(best) if best is not None else None


def _get_or_train_agent(agent_name: str, run_config: RunConfig, seed: int, register: bool) -> Any:
    """Load the cached policy for ``{config, seed}`` if registered, else train + register it."""
    base = _norm(agent_name)
    rc_seed = replace(run_config, agent=agent_name, seed=seed)
    cfg_hash = config_hash(rc_seed)

    cached = _find_cached_version(base, cfg_hash, seed)
    if cached is not None:
        from rl_execution.tracking.registry import load_registered_agent

        env = make_env(
            agent_name,
            run_config.execution,
            regime=run_config.regime,
            randomized=run_config.randomized,
            seed=seed,
        )
        return load_registered_agent(base, env, version=cached)

    agent, _ = train_agent(
        agent_name,
        total_timesteps=run_config.timesteps,
        exec_config=run_config.execution,
        regime=run_config.regime,
        randomized=run_config.randomized,
        seed=seed,
        device=run_config.device,
        agent_kwargs=run_config.agent_kwargs or None,
    )
    if register:
        file_tag = f"{base}-s{seed}"
        save_agent(agent, agent_name, tag=file_tag, seed=seed, config=rc_seed)
        try:
            from rl_execution.tracking.registry import register_model

            register_model(
                str(model_artifact_path(agent_name, file_tag)),
                base,
                {"agent": base, "seed": seed, "config_hash": cfg_hash, "git_sha": git_sha()},
            )
        except Exception:  # pragma: no cover - registry is optional (needs the tracking extra)
            pass
    return agent


def _run_one_seed(
    agent_name: str,
    run_config: RunConfig,
    seed: int,
    regimes: List[str],
    benchmark: str,
    episodes: int,
    register: bool,
    keep_pairs: bool,
) -> Dict[str, Any]:
    """Train/load one seed, evaluate it across regimes, return per-regime paired summaries."""
    disp = _display(agent_name)
    agent = _get_or_train_agent(agent_name, run_config, seed, register)
    strategies = {benchmark: build_baselines()[benchmark], disp: AgentStrategy(agent, disp)}
    action_types = {disp: required_action_type(agent_name)}
    results = evaluate_across_regimes(
        strategies,
        run_config.execution,
        regimes=regimes,
        action_types=action_types,
        n_episodes=episodes,
        base_seed=run_config.eval_base_seed,
        progress=False,
    )

    vs_bench: Dict[str, float] = {}
    pairs: Dict[str, Any] = {}
    for regime in regimes:
        strat_is = episode_is(results[regime][disp])
        bench_is = episode_is(results[regime][benchmark])
        n = min(len(strat_is), len(bench_is))
        diff: np.ndarray = strat_is[:n] - bench_is[:n]
        vs_bench[regime] = float(diff.mean())
        if keep_pairs:
            pairs[regime] = (strat_is[:n], bench_is[:n])
    return {"seed": seed, "vs_bench": vs_bench, "pairs": pairs}


@dataclass
class MultiSeedResult:
    """Across-seed evaluation of one agent vs a benchmark, per regime."""

    agent: str
    benchmark: str
    seeds: List[int]
    regimes: List[str]
    per_seed_vs_bench: Dict[str, np.ndarray]  # regime -> array of per-seed mean vs_bench
    episode_pairs: Dict[str, Any] = field(default_factory=dict)  # regime -> (strat_is, bench_is)
    alpha: float = 0.05

    def across_seed_frame(self, n_boot: int = 10_000, rng: Any = 0) -> pd.DataFrame:
        """Headline table: per-regime mean ``vs_<bench>`` with a bootstrap CI **across seeds**."""
        gen = np.random.default_rng(rng)
        rows = []
        for regime in self.regimes:
            vals = np.asarray(self.per_seed_vs_bench[regime], dtype=float)
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, alpha=self.alpha, rng=gen)
            rows.append(
                {
                    "regime": regime,
                    "strategy": self.agent,
                    f"vs_{self.benchmark}_mean": (
                        float(np.mean(vals)) if vals.size else float("nan")
                    ),
                    "across_seed_ci_low": lo,
                    "across_seed_ci_high": hi,
                    "n_seeds": int(vals.size),
                }
            )
        return pd.DataFrame(rows)

    def within_run_frame(self, n_boot: int = 10_000, rng: Any = 0) -> pd.DataFrame:
        """Secondary table: per-regime paired episode CI for the first seed (episode-level noise)."""
        gen = np.random.default_rng(rng)
        rows = []
        for regime, (strat_is, bench_is) in self.episode_pairs.items():
            lo, hi = paired_bootstrap_ci(
                strat_is, bench_is, n_boot=n_boot, alpha=self.alpha, rng=gen
            )
            n = min(len(strat_is), len(bench_is))
            diff = np.asarray(strat_is)[:n] - np.asarray(bench_is)[:n]
            rows.append(
                {
                    "regime": regime,
                    "strategy": self.agent,
                    f"vs_{self.benchmark}_mean": float(diff.mean()),
                    "episode_ci_low": lo,
                    "episode_ci_high": hi,
                    "n_episodes": int(n),
                }
            )
        return pd.DataFrame(rows)


def run_multiseed(
    agent_name: str,
    run_config: Optional[RunConfig] = None,
    seeds: Sequence[int] = range(10),
    *,
    regimes: Optional[List[str]] = None,
    benchmark: str = "TWAP",
    episodes: Optional[int] = None,
    register: bool = True,
    n_jobs: int = 1,
) -> MultiSeedResult:
    """Train/evaluate ``agent_name`` across ``seeds`` and aggregate with across-seed CIs.

    For each seed the policy is trained (or loaded from the registry if that
    ``{config_hash, seed}`` already exists) and evaluated across ``regimes`` against
    ``benchmark`` on a fixed set of evaluation seeds (so eval noise is shared and the only
    varying ingredient is the training seed).  ``n_jobs`` parallelises seeds with joblib when
    ``> 1`` (each seed writes a distinct ``<agent>-s<seed>`` artifact, so there is no clobber).
    """
    run_config = run_config or RunConfig(agent=agent_name)
    regimes = list(regimes or list_regimes())
    episodes = int(episodes or run_config.eval_episodes)
    seed_list = [int(s) for s in seeds]

    args = [
        (agent_name, run_config, s, regimes, benchmark, episodes, register, i == 0)
        for i, s in enumerate(seed_list)
    ]
    outputs = _map_seeds(args, n_jobs)

    per_seed: Dict[str, List[float]] = {r: [] for r in regimes}
    episode_pairs: Dict[str, Any] = {}
    for out in outputs:
        for regime in regimes:
            per_seed[regime].append(out["vs_bench"][regime])
        if out["pairs"]:
            episode_pairs = out["pairs"]

    return MultiSeedResult(
        agent=_display(agent_name),
        benchmark=benchmark,
        seeds=seed_list,
        regimes=regimes,
        per_seed_vs_bench={r: np.asarray(v, dtype=float) for r, v in per_seed.items()},
        episode_pairs=episode_pairs,
    )


def _map_seeds(args: List[tuple], n_jobs: int) -> List[Dict[str, Any]]:
    """Run ``_run_one_seed`` over ``args``; serial for ``n_jobs == 1`` else joblib-parallel."""
    if n_jobs == 1:
        return [_run_one_seed(*a) for a in args]
    try:
        from joblib import Parallel, delayed
    except ImportError:  # pragma: no cover - joblib ships with the tracking extra
        return [_run_one_seed(*a) for a in args]
    return list(
        Parallel(n_jobs=n_jobs)(delayed(_run_one_seed)(*a) for a in args)
    )  # pragma: no cover
