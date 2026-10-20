"""Matplotlib/seaborn plotting helpers.

Every function returns a ``matplotlib.figure.Figure`` (and optionally saves it).  A
non-interactive Agg backend is used so plots render in headless / script contexts (lab
machines, CI).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Optional

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
def plot_inventory_decay(results: Mapping[str, Any], save: Optional[str] = None):
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


def plot_execution_schedule(results: Mapping[str, Any], save: Optional[str] = None):
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


def plot_validation_curves(
    curves: Mapping[str, pd.DataFrame],
    references: Optional[Mapping[str, float]] = None,
    save: Optional[str] = None,
):
    """Validation objective against training steps, one line per run.

    ``curves`` maps a run name to a frame with ``step`` and ``objective_bps`` columns;
    ``references`` optionally adds horizontal lines (e.g. the TWAP and Almgren-Chriss
    objectives) to show what "good" looks like.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in curves.items():
        ax.plot(df["step"], df["objective_bps"], marker="o", ms=3, label=name)
    for name, value in (references or {}).items():
        ax.axhline(value, ls="--", lw=1, color="0.4")
        ax.annotate(name, (0, value), textcoords="offset points", xytext=(2, 3), fontsize=8)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Validation objective (bps, lower is better)")
    ax.set_title("Checkpoint selection on validation seeds")
    ax.legend(fontsize=8)
    if save:
        save_fig(fig, save)
    return fig


# --------------------------------------------------------------------------- comparisons
def plot_risk_vs_cost(summary: pd.DataFrame, save: Optional[str] = None):
    """Mean implementation shortfall against its standard deviation, one point per strategy.

    Execution is a mean-variance problem: a strategy up and to the right of another is
    dominated.  ``summary`` needs ``strategy``, ``IS_bps`` and ``IS_std`` columns.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in summary.iterrows():
        ax.scatter(row["IS_std"], row["IS_bps"], s=40)
        ax.annotate(
            str(row["strategy"]),
            (row["IS_std"], row["IS_bps"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    ax.set_xlabel("IS standard deviation (bps)")
    ax.set_ylabel("Mean implementation shortfall (bps)")
    ax.set_title("Cost vs risk (lower-left is better)")
    if save:
        save_fig(fig, save)
    return fig


def plot_is_distribution(results: Mapping[str, Any], save: Optional[str] = None):
    """Box/violin of per-episode implementation shortfall per strategy."""
    set_style()
    data, labels = [], []
    for name, res in results.items():
        df = res.metrics_frame()
        if "implementation_shortfall_bps" in df:
            data.append(df["implementation_shortfall_bps"].to_numpy())
            labels.append(name)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Not labels=/tick_labels=: the former was removed in matplotlib 3.11, the
    # latter only exists from 3.9, and the floor here is 3.7. Boxes sit at 1..n.
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Implementation shortfall (bps)")
    ax.set_title("Per-episode IS distribution (lower is better)")
    plt.xticks(rotation=30, ha="right")
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
