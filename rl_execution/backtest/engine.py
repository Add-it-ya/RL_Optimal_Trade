"""A uniform backtesting engine.

Any object implementing the ``reset(env)`` / ``act(obs, info) -> action`` interface
(baselines *and* RL-agent wrappers) can be evaluated against an :class:`ExecutionEnv` over
many randomised episodes.  Results bundle per-episode metrics, inventory trajectories,
execution schedules and aggregate statistics for downstream reporting / plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from rl_execution.metrics.metrics import aggregate_metrics, compute_episode_metrics


@dataclass
class BacktestResult:
    """Container for the outcome of evaluating one strategy over many episodes."""

    name: str
    episode_metrics: List[Dict[str, float]] = field(default_factory=list)
    summaries: List[Dict[str, Any]] = field(default_factory=list)
    inventory_trajectories: List[np.ndarray] = field(default_factory=list)
    schedules: List[np.ndarray] = field(default_factory=list)
    reward_curves: List[np.ndarray] = field(default_factory=list)
    price_paths: List[np.ndarray] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)

    # -- convenience views ------------------------------------------------------
    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_metrics)

    def mean_inventory_trajectory(self) -> np.ndarray:
        return _stack_mean(self.inventory_trajectories)

    def mean_schedule(self) -> np.ndarray:
        return _stack_mean(self.schedules)

    def mean_reward_curve(self) -> np.ndarray:
        return _stack_mean(self.reward_curves)

    def summary_row(self) -> Dict[str, Any]:
        """A flat one-row summary (means + Sharpe) suitable for a comparison table."""
        agg = self.aggregate
        return {
            "strategy": self.name,
            "IS_bps": agg.get("implementation_shortfall_bps_mean", float("nan")),
            "IS_std": agg.get("implementation_shortfall_bps_std", float("nan")),
            "ExecCost_bps": agg.get("execution_cost_bps_mean", float("nan")),
            "MktImpact_bps": agg.get("market_impact_bps_mean", float("nan")),
            "AvgFill": agg.get("avg_fill_price_mean", float("nan")),
            "Unexecuted": agg.get("unexecuted_shares_mean", float("nan")),
            "Reward": agg.get("cum_reward_mean", float("nan")),
            "IS_Sharpe": agg.get("is_sharpe", float("nan")),
        }


def _stack_mean(arrays: List[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([])
    n = min(len(a) for a in arrays)
    if n == 0:
        return np.array([])
    stacked = np.stack([np.asarray(a)[:n] for a in arrays])
    return stacked.mean(axis=0)


def run_episode(env, strategy, seed: Optional[int] = None):
    """Run a single episode of ``strategy`` on ``env``.

    Returns ``(summary, history_df, rewards, price_path)``.
    """
    obs, info = env.reset(seed=seed)
    strategy.reset(env)
    rewards: List[float] = []
    price_path: List[float] = [env.market.mid]

    done = False
    while not done:
        action = strategy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        price_path.append(env.market.mid)
        done = terminated or truncated

    summary = info["episode_summary"]
    history = env.episode_dataframe()
    return summary, history, np.asarray(rewards), np.asarray(price_path)


def evaluate(
    env_factory: Callable[[], Any],
    strategy: Any,
    n_episodes: int = 100,
    base_seed: int = 0,
    name: Optional[str] = None,
    progress: bool = False,
) -> BacktestResult:
    """Evaluate one strategy over ``n_episodes`` randomised episodes.

    ``env_factory`` is a zero-arg callable returning a fresh :class:`ExecutionEnv`; the
    same factory is reused so the environment configuration is held fixed while the random
    seed is varied per episode.
    """
    env = env_factory()
    name = name or getattr(strategy, "name", strategy.__class__.__name__)
    result = BacktestResult(name=name)

    iterator = range(n_episodes)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=name, leave=False)
        except ImportError:
            pass

    for i in iterator:
        seed = base_seed + i
        summary, history, rewards, price_path = run_episode(env, strategy, seed=seed)
        result.summaries.append(summary)
        result.episode_metrics.append(compute_episode_metrics(summary, history))

        total = summary["total_inventory"]
        inv = np.concatenate([[total], history["inventory_after"].to_numpy()])
        result.inventory_trajectories.append(inv)
        result.schedules.append(history["shares"].to_numpy())
        result.reward_curves.append(rewards)
        result.price_paths.append(price_path)

    result.aggregate = aggregate_metrics(result.episode_metrics)
    return result


def compare_strategies(
    env_factory: Callable[[], Any],
    strategies: Dict[str, Any],
    n_episodes: int = 100,
    base_seed: int = 0,
    progress: bool = True,
) -> Dict[str, BacktestResult]:
    """Evaluate several strategies on the *same* sequence of episodes (paired seeds)."""
    results: Dict[str, BacktestResult] = {}
    for name, strat in strategies.items():
        results[name] = evaluate(
            env_factory,
            strat,
            n_episodes=n_episodes,
            base_seed=base_seed,
            name=name,
            progress=progress,
        )
    return results


def results_table(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Build a tidy comparison table (one row per strategy), sorted by mean IS."""
    rows = [res.summary_row() for res in results.values()]
    df = pd.DataFrame(rows).set_index("strategy")
    return df.sort_values("IS_bps")


def paired_is_table(results: Dict[str, BacktestResult], benchmark: str = "TWAP") -> pd.DataFrame:
    """Paired comparison of implementation shortfall against a benchmark strategy.

    Because every strategy is evaluated on the *same* sequence of seeds (common random
    numbers), the per-episode IS difference cancels the shared price-path risk, giving a
    far lower-variance estimate of skill than comparing absolute means.  Columns:

    * ``IS_bps``        -- mean implementation shortfall (lower = better).
    * ``vs_<bench>``    -- mean IS improvement over the benchmark (negative = better).
    * ``win_rate_%``    -- fraction of episodes with lower IS than the benchmark.
    * ``t_stat``        -- paired t-statistic of the improvement (negative & large = robust).
    """

    def is_array(res: BacktestResult) -> np.ndarray:
        return np.array([m["implementation_shortfall_bps"] for m in res.episode_metrics])

    if benchmark not in results:
        raise KeyError(f"Benchmark '{benchmark}' not in results.")
    bench = is_array(results[benchmark])

    rows = []
    for name, res in results.items():
        arr = is_array(res)
        n = min(len(arr), len(bench))
        diff = arr[:n] - bench[:n]
        sd = diff.std(ddof=1) if n > 1 else 0.0
        rows.append(
            {
                "strategy": name,
                "IS_bps": float(arr.mean()),
                f"vs_{benchmark}": float(diff.mean()),
                "win_rate_%": (
                    float(100.0 * np.mean(diff < 0)) if name != benchmark else float("nan")
                ),
                "t_stat": (
                    float(diff.mean() / (sd / np.sqrt(n) + 1e-12))
                    if name != benchmark
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy").sort_values("IS_bps")
