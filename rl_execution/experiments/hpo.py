"""Hyperparameter optimization with Optuna (Step 2).

Principled, *resumable* tuning that does not cheat:

* **Resumable & parallelizable** -- studies are persisted to a SQLite file
  (``sqlite:///optuna.db``), so a sweep can be stopped, resumed, or run from several
  processes against the same study.
* **No tuning on the test set** -- the objective selects on a **validation** split of
  seeds/regimes that is disjoint from the final-evaluation split (train/val/test at the
  *experiment* level), and the chosen ``n_trials`` is recorded so the report can apply a
  deflated-Sharpe haircut for the search.
* **Search spaces match the agents** -- keys map straight onto the custom-DQN and
  Stable-Baselines3 constructors, so the best params drop into a :class:`RunConfig`.

The heavy ``optimize`` / objective paths are exercised only in full sweeps (they train
agents); the study plumbing and search spaces are unit-tested.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rl_execution.config import RunConfig
from rl_execution.utils.io import PROJECT_ROOT


def _norm(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def default_storage() -> str:
    """Local SQLite URI for persisted (resumable) studies; gitignored alongside ``mlflow.db``."""
    return f"sqlite:///{(PROJECT_ROOT / 'optuna.db').as_posix()}"


def search_space(trial: Any, algo: str) -> Dict[str, Any]:
    """Per-algorithm hyperparameter search space.

    Returns a dict of ``agent_kwargs`` suitable for :func:`rl_execution.agents.make_agent`
    (custom DQN keys: ``lr``/``gamma``/``batch_size``/``target_update_interval``/``eps_fraction``/
    ``hidden``; SB3 keys: ``learning_rate``/``n_steps``|``buffer_size``/``ent_coef``/``net_arch``).
    """
    key = _norm(algo)
    if key in ("dqn", "doubledqn", "ddqn"):
        h = trial.suggest_categorical("hidden", [64, 128, 256])
        return {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "target_update_interval": trial.suggest_categorical(
                "target_update_interval", [250, 500, 1000]
            ),
            "eps_fraction": trial.suggest_float("eps_fraction", 0.05, 0.5),
            "hidden": (h, h),
        }
    if key in ("ppo", "a2c"):
        n_steps = (
            trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
            if key == "ppo"
            else trial.suggest_categorical("n_steps", [5, 8, 16, 32])
        )
        s = trial.suggest_categorical("net_arch", [64, 128, 256])
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "n_steps": n_steps,
            "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
            "net_arch": (s, s),
        }
    if key == "sac":
        s = trial.suggest_categorical("net_arch", [64, 128, 256])
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
            "net_arch": (s, s),
        }
    raise KeyError(f"No search space for agent '{algo}'.")


def make_study(
    study_name: str,
    *,
    storage: Optional[str] = None,
    direction: str = "minimize",
    load_if_exists: bool = True,
    seed: int = 0,
) -> Any:
    """Create or resume a persisted Optuna study (TPE sampler, seeded for reproducibility).

    ``direction='minimize'`` because the objective is ``vs_<bench>`` improvement, where more
    negative = better execution.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=study_name,
        storage=storage or default_storage(),
        direction=direction,
        load_if_exists=load_if_exists,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )


def run_study(study: Any, objective: Callable[[Any], float], n_trials: int) -> Any:
    """Optimise ``objective`` for ``n_trials`` and return the (resumable) study."""
    study.optimize(objective, n_trials=n_trials)
    return study


def build_objective(
    agent_name: str,
    base_config: RunConfig,
    val_seeds: Sequence[int],
    val_regimes: Optional[List[str]] = None,
    benchmark: str = "TWAP",
    timesteps: Optional[int] = None,
) -> Callable[[Any], float]:
    """Objective: mean across-seed ``vs_<bench>`` on the *validation* split (lower = better)."""

    def objective(trial: Any) -> float:  # pragma: no cover - trains agents; full-sweep only
        from rl_execution.experiments.multiseed import run_multiseed

        params = search_space(trial, agent_name)
        rc = replace(
            base_config,
            agent=agent_name,
            agent_kwargs={**(base_config.agent_kwargs or {}), **params},
        )
        if timesteps is not None:
            rc = replace(rc, timesteps=timesteps)
        result = run_multiseed(
            agent_name,
            rc,
            seeds=val_seeds,
            regimes=val_regimes,
            benchmark=benchmark,
            episodes=rc.eval_episodes,
        )
        frame = result.across_seed_frame()
        return float(frame[f"vs_{benchmark}_mean"].mean())

    return objective


def optimize(
    agent_name: str,
    base_config: Optional[RunConfig] = None,
    *,
    n_trials: int = 20,
    val_seeds: Sequence[int] = range(3),
    val_regimes: Optional[List[str]] = None,
    benchmark: str = "TWAP",
    timesteps: Optional[int] = None,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    tracker: Any = None,
) -> Tuple[Dict[str, Any], RunConfig, Any]:  # pragma: no cover - full-sweep entry point
    """Run an Optuna sweep and fold the best params into a :class:`RunConfig`.

    Returns ``(best_params, best_config, study)``.  ``n_trials`` is logged to ``tracker`` so the
    report can quote it for the deflated-Sharpe haircut.
    """
    base_config = base_config or RunConfig(agent=agent_name)
    study = make_study(study_name or f"hpo-{_norm(agent_name)}", storage=storage)
    objective = build_objective(
        agent_name, base_config, list(val_seeds), val_regimes, benchmark, timesteps
    )
    study.optimize(objective, n_trials=n_trials)

    best = dict(study.best_params)
    best_config = replace(
        base_config, agent=agent_name, agent_kwargs={**(base_config.agent_kwargs or {}), **best}
    )
    if tracker is not None:
        tracker.log_params({f"hpo/{k}": v for k, v in best.items()})
        tracker.log_metrics(
            {"hpo/best_value": float(study.best_value), "hpo/n_trials": float(len(study.trials))}
        )
    return best, best_config, study
