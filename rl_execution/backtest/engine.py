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


def episode_is(res: BacktestResult) -> np.ndarray:
    """Per-episode implementation shortfall (bps) of a backtest result, as a 1-D array."""
    return np.array([m["implementation_shortfall_bps"] for m in res.episode_metrics], dtype=float)


def _paired_pvalue(diff: np.ndarray) -> float:
    """Two-sided p-value of a paired difference series (paired t-test on the differences)."""
    d = np.asarray(diff, dtype=float)
    if d.size < 2:
        return float("nan")
    if float(d.std(ddof=1)) == 0.0:
        # Identical series -> no evidence of a difference (p = 1); a non-zero constant shift is
        # a degenerate "infinitely significant" case, reported as p = 0.
        return 1.0 if np.isclose(d.mean(), 0.0) else 0.0
    from scipy import stats

    return float(stats.ttest_1samp(d, 0.0).pvalue)


def paired_is_table(
    results: Dict[str, BacktestResult],
    benchmark: str = "TWAP",
    *,
    ci: bool = True,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    rng: Any = 0,
) -> pd.DataFrame:
    """Paired comparison of implementation shortfall against a benchmark strategy.

    Because every strategy is evaluated on the *same* sequence of seeds (common random
    numbers), the per-episode IS difference cancels the shared price-path risk, giving a
    far lower-variance estimate of skill than comparing absolute means.  Columns:

    * ``IS_bps``                 -- mean implementation shortfall (lower = better).
    * ``vs_<bench>``             -- mean IS improvement over the benchmark (negative = better).
    * ``win_rate_%``             -- fraction of episodes with lower IS than the benchmark.
    * ``t_stat``                 -- paired t-statistic of the improvement (negative & large = robust).
    * ``vs_<bench>_ci_low/high`` -- 95% paired-bootstrap CI of the improvement (when ``ci``).
    * ``p_value``                -- two-sided paired-t p-value of the improvement (when ``ci``).

    ``rng`` seeds the bootstrap so the interval columns are reproducible across runs.
    """
    from rl_execution.metrics.stats import paired_bootstrap_ci

    if benchmark not in results:
        raise KeyError(f"Benchmark '{benchmark}' not in results.")
    bench = episode_is(results[benchmark])
    gen = np.random.default_rng(rng)

    rows = []
    for name, res in results.items():
        arr = episode_is(res)
        n = min(len(arr), len(bench))
        diff: np.ndarray = arr[:n] - bench[:n]
        sd = diff.std(ddof=1) if n > 1 else 0.0
        is_bench = name == benchmark
        row: Dict[str, Any] = {
            "strategy": name,
            "IS_bps": float(arr.mean()),
            f"vs_{benchmark}": float(diff.mean()),
            "win_rate_%": (float("nan") if is_bench else float(100.0 * np.mean(diff < 0))),
            "t_stat": (
                float("nan") if is_bench else float(diff.mean() / (sd / np.sqrt(n) + 1e-12))
            ),
        }
        if ci:
            if is_bench or n < 2:
                lo = hi = pval = float("nan")
            else:
                lo, hi = paired_bootstrap_ci(
                    arr[:n], bench[:n], n_boot=n_boot, alpha=alpha, rng=gen
                )
                pval = _paired_pvalue(diff)
            row[f"vs_{benchmark}_ci_low"] = lo
            row[f"vs_{benchmark}_ci_high"] = hi
            row["p_value"] = pval
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy").sort_values("IS_bps")


def corrected_significance(
    results_by_regime: Dict[str, Dict[str, BacktestResult]],
    benchmark: str = "TWAP",
    method: str = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Multiple-testing-corrected paired significance across the regime x strategy grid.

    For every (regime, non-benchmark strategy) pair, computes the paired IS improvement vs
    ``benchmark`` and its two-sided p-value, then applies a family-wise / FDR correction
    (``method`` in {``holm``, ``bh``, ``bonferroni``}) across *all* of those tests jointly --
    so a regime is only flagged significant after paying for the many comparisons made.
    Returns a tidy frame with ``vs_<bench>``, raw ``p_value``, ``p_adjusted`` and ``reject_H0``.
    """
    from rl_execution.metrics.stats import adjust_pvalues

    rows = []
    for regime, by_strat in results_by_regime.items():
        if benchmark not in by_strat:
            continue
        bench = episode_is(by_strat[benchmark])
        for name, res in by_strat.items():
            if name == benchmark:
                continue
            arr = episode_is(res)
            n = min(len(arr), len(bench))
            diff: np.ndarray = arr[:n] - bench[:n]
            rows.append(
                {
                    "regime": regime,
                    "strategy": name,
                    f"vs_{benchmark}": float(diff.mean()),
                    "n_episodes": int(n),
                    "p_value": _paired_pvalue(diff),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    reject, p_adj = adjust_pvalues(df["p_value"].to_numpy(), method=method, alpha=alpha)
    df["p_adjusted"] = p_adj
    df["reject_H0"] = reject
    return df.sort_values(["regime", f"vs_{benchmark}"]).reset_index(drop=True)
