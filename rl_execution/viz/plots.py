"""Matplotlib/seaborn plotting helpers.

Every function returns a ``matplotlib.figure.Figure`` (and optionally saves it), so the
same code serves both the offline report and the Streamlit dashboard.  A non-interactive
Agg backend is used so plots render in headless / script contexts.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import seaborn as sns

    _HAS_SNS = True
except Exception:  # pragma: no cover
    _HAS_SNS = False


def set_style() -> None:
    if _HAS_SNS:
        sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"


def save_fig(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------- curves
def plot_inventory_decay(results: Dict[str, "object"], save: Optional[str] = None):
    """Mean remaining-inventory trajectory (inventory decay curve) per strategy."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        traj = res.mean_inventory_trajectory()
        if traj.size == 0:
            continue
        steps = np.arange(len(traj))
        ax.plot(steps, traj / traj[0], marker="o", ms=3, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Remaining inventory (fraction)")
    ax.set_title("Inventory decay curves")
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


def plot_execution_schedule(results: Dict[str, "object"], save: Optional[str] = None):
    """Mean shares executed per step (execution schedule) per strategy."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    width = 0.8 / max(len(names), 1)
    for i, (name, res) in enumerate(results.items()):
        sched = res.mean_schedule()
        if sched.size == 0:
            continue
        x = np.arange(len(sched))
        ax.bar(x + i * width, sched, width=width, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Shares executed (mean)")
    ax.set_title("Execution schedules")
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


def plot_reward_curve(results: Dict[str, "object"], save: Optional[str] = None):
    """Mean cumulative reward across the episode per strategy."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        curve = res.mean_reward_curve()
        if curve.size == 0:
            continue
        ax.plot(np.cumsum(curve), label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward (bps-scaled)")
    ax.set_title("Cumulative reward over the execution horizon")
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


def plot_training_curve(
    reward_log: Dict[str, Iterable[float]] | Iterable[float],
    window: int = 50,
    save: Optional[str] = None,
):
    """Smoothed training reward curve(s).  Accepts a single series or {name: series}."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not isinstance(reward_log, dict):
        reward_log = {"agent": reward_log}
    for name, rewards in reward_log.items():
        r = np.asarray(list(rewards), dtype=float)
        if r.size == 0:
            continue
        if r.size >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(r, kernel, mode="valid")
            ax.plot(np.arange(len(smooth)) + window, smooth, label=f"{name} (MA{window})")
        else:
            ax.plot(r, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training reward curves")
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


# --------------------------------------------------------------------------- comparisons
def plot_cost_comparison(table: pd.DataFrame, save: Optional[str] = None):
    """Grouped bar chart of IS / execution cost / market impact per strategy."""
    set_style()
    cols = [c for c in ["IS_bps", "ExecCost_bps", "MktImpact_bps"] if c in table.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    table = table.sort_values("IS_bps")
    x = np.arange(len(table))
    width = 0.8 / max(len(cols), 1)
    for i, col in enumerate(cols):
        ax.bar(x + i * width, table[col].to_numpy(), width=width, label=col)
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(table.index, rotation=30, ha="right")
    ax.set_ylabel("Cost (bps)")
    ax.set_title("Execution cost comparison (lower is better)")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


def plot_rl_vs_baselines(table: pd.DataFrame, rl_names: Iterable[str], save: Optional[str] = None):
    """Bar chart of mean IS with RL agents highlighted vs baselines."""
    set_style()
    table = table.sort_values("IS_bps")
    rl = set(rl_names)
    colors = ["#d1495b" if name in rl else "#5b8c85" for name in table.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(table.index, table["IS_bps"].to_numpy(), color=colors)
    ax.set_ylabel("Implementation shortfall (bps)")
    ax.set_title("RL agents vs baselines (lower is better)")
    ax.axhline(0, color="k", lw=0.6)
    plt.xticks(rotation=30, ha="right")
    if "IS_std" in table.columns:
        ax.errorbar(
            table.index,
            table["IS_bps"],
            yerr=table["IS_std"],
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            capsize=3,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d1495b"),
        plt.Rectangle((0, 0), 1, 1, color="#5b8c85"),
    ]
    ax.legend(handles, ["RL agent", "Baseline"], fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


def plot_is_distribution(results: Dict[str, "object"], save: Optional[str] = None):
    """Box/violin of per-episode implementation shortfall per strategy."""
    set_style()
    data, labels = [], []
    for name, res in results.items():
        df = res.metrics_frame()
        if "implementation_shortfall_bps" in df:
            data.append(df["implementation_shortfall_bps"].to_numpy())
            labels.append(name)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Implementation shortfall (bps)")
    ax.set_title("Per-episode IS distribution (lower is better)")
    plt.xticks(rotation=30, ha="right")
    if save:
        save_fig(fig, save)
    return fig


def plot_price_path(result, episode: int = 0, save: Optional[str] = None):
    """Mid-price path and fills for a single sample episode."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    if result.price_paths and episode < len(result.price_paths):
        path = result.price_paths[episode]
        ax.plot(path, label="Mid price", color="#33658a")
        sched = result.schedules[episode] if episode < len(result.schedules) else None
        if sched is not None:
            steps = np.arange(1, len(sched) + 1)
            sizes = 20 + 180 * sched / (sched.max() + 1e-9)
            ax.scatter(
                steps,
                path[1 : len(sched) + 1],
                s=sizes,
                color="#d1495b",
                alpha=0.7,
                label="Fills (size ∝ shares)",
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title(f"Sample execution path ({result.name})")
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


# --------------------------------------------------------------------------- brownian
def plot_brownian_paths(
    paths: np.ndarray, sigma: float, s0: float = 100.0, save: Optional[str] = None
):
    """Sample price paths with the theoretical ``+/- sigma sqrt(t)`` diffusion envelope.

    ``paths`` is ``(n_paths, n_steps + 1)``.  The envelope is what makes the square-root-of-
    time scaling of Brownian motion visible: the spread of the paths should track it.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(paths.shape[1])
    for row in paths[:40]:
        ax.plot(steps, row, lw=0.8, alpha=0.45, color="#33658a")
    envelope = sigma * s0 * np.sqrt(steps)
    ax.plot(steps, s0 + envelope, color="#d1495b", ls="--", lw=1.6, label=r"$\pm\sigma\sqrt{t}$")
    ax.plot(steps, s0 - envelope, color="#d1495b", ls="--", lw=1.6)
    ax.plot(steps, paths.mean(axis=0), color="#111111", lw=2.0, label="Mean path")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.set_title(f"Brownian price paths ({paths.shape[0]} simulations)")
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


def plot_return_qq(returns: np.ndarray, save: Optional[str] = None):
    """Normal QQ-plot of simulated returns -- a straight line means Gaussian increments."""
    set_style()
    from scipy import stats as _stats

    fig, ax = plt.subplots(figsize=(6, 6))
    _stats.probplot(np.asarray(returns, dtype=float).ravel(), dist="norm", plot=ax)
    ax.get_lines()[0].set(markersize=3, alpha=0.5, color="#33658a")
    ax.get_lines()[1].set(color="#d1495b", lw=1.6)
    ax.set_title("Normal QQ-plot of simulated returns")
    if save:
        save_fig(fig, save)
    return fig


def plot_vol_acf(
    returns_by_label: Dict[str, np.ndarray], max_lag: int = 20, save: Optional[str] = None
):
    """Autocorrelation of ``|returns|`` -- the standard fingerprint of volatility clustering.

    An i.i.d. volatility model sits flat at zero; an exponential-OU log-volatility process
    decays slowly, which is what real markets show and what gives the environment's
    ``volatility`` observation feature genuine forecast value.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rets in returns_by_label.items():
        x = np.abs(np.asarray(rets, dtype=float).ravel())
        x = x - x.mean()
        denom = float((x * x).mean())
        acf = [
            float((x[:-k] * x[k:]).mean() / denom) if denom > 0 else 0.0
            for k in range(1, max_lag + 1)
        ]
        ax.plot(range(1, max_lag + 1), acf, marker="o", ms=3, label=label)
    ax.axhline(0.0, color="#888888", lw=1.0)
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel(r"Autocorrelation of $|r_t|$")
    ax.set_title("Volatility clustering")
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


def plot_regime_heatmap(regime_df: pd.DataFrame, value: str = "IS_bps", save: Optional[str] = None):
    """Heatmap of a metric across (strategy x regime)."""
    set_style()
    pivot = regime_df.pivot(index="strategy", columns="regime", values=value)
    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 3, 0.6 * pivot.shape[0] + 2))
    if _HAS_SNS:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            center=0,
            ax=ax,
            cbar_kws={"label": value},
        )
    else:  # pragma: no cover
        im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        fig.colorbar(im, ax=ax, label=value)
    ax.set_title(f"{value} by strategy and regime (lower = better)")
    if save:
        save_fig(fig, save)
    return fig
