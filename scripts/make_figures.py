#!/usr/bin/env python
"""Figures for RESULTS.md, drawn from the committed results (a light and a dark version).

    python scripts/make_figures.py                     # all figures -> docs/figures/
    python scripts/make_figures.py --compute-frontier  # first re-trace AC's frontier (~1 min)

Everything is read from committed files under ``results/``: ``per_seed.csv`` and
``comparison.csv`` from ``rlx report``, ``schedules.csv`` from ``rlx evaluate``, the E2
``sweep.csv`` files, the lab's ``bench/throughput.csv``, and ``frontier/ac_frontier.csv``
(Almgren-Chriss at a range of risk aversions on the E1 test block, written by
``--compute-frontier``). Colour marks the learned agents only (SB3 PPO, VecPPO,
DoubleDQN); baselines are drawn in neutral ink with their own line styles, so a colour
always means the same agent across figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
FRONTIER = RESULTS / "frontier" / "ac_frontier.csv"

#: Chart surfaces and inks; the three agent colours validated together (colour-blind safe
#: in every pair, both modes).
THEMES: Dict[str, Dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "#f0efec",
        "PPO": "#2a78d6",
        "VecPPO": "#eb6834",
        "DoubleDQN": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "#262624",
        "PPO": "#3987e5",
        "VecPPO": "#d95926",
        "DoubleDQN": "#199e70",
    },
}


def setup(theme: Dict[str, str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.facecolor": theme["surface"],
            "figure.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "axes.edgecolor": theme["axis"],
            "axes.labelcolor": theme["ink2"],
            "axes.titlecolor": theme["ink"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": theme["grid"],
            "grid.linewidth": 0.6,
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "xtick.labelcolor": theme["ink2"],
            "ytick.labelcolor": theme["ink2"],
            "legend.frameon": False,
            "legend.labelcolor": theme["ink2"],
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
        }
    )


def save(fig, name: str, mode: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}{'' if mode == 'light' else '-dark'}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def read(name: str, file: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name / file)


def agent_row(name: str, agent: str) -> pd.Series:
    table = read(name, "comparison.csv")
    return table[table["strategy"] == agent].iloc[0]


# ----------------------------------------------------------------------------- E1
#: The E1 configurations, top to bottom: (label, results folder, agent, filled marker).
E1_ROWS = [
    ("DoubleDQN, 2M steps (E1)", "e1", "DoubleDQN", True),
    ("PPO, 2M steps (E1)", "e1", "PPO", False),
    ("PPO, 5M steps (E1b)", "e1b_5m_10seed", "PPO", True),
    ("VecPPO, 5M, 256 envs (GPU)", "e1_vec", "VecPPO", False),
    ("VecPPO tuned, 50M, 4,096 envs", "e1t_C", "VecPPO", True),
]


def fig_e1_gaps(t: Dict[str, str], mode: str) -> Path:
    """Per-seed gap to the optimum, with the across-seed mean and its 95% CI."""
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    for y, (label, name, agent, filled) in enumerate(E1_ROWS[::-1]):
        seeds = read(name, "per_seed.csv")
        gaps = seeds.loc[seeds["agent"] == agent, "vs_AlmgrenChriss"].to_numpy()
        row = agent_row(name, agent)
        colour = t[agent]
        jitter = np.linspace(-0.12, 0.12, gaps.size)
        ax.scatter(gaps, y + jitter, s=16, color=colour, alpha=0.55, linewidths=0, zorder=3)
        ax.plot([row["ci_low"], row["ci_high"]], [y + 0.3] * 2, color=colour, lw=2.2, zorder=4)
        ax.scatter(
            [row["vs_AlmgrenChriss"]],
            [y + 0.3],
            s=46,
            zorder=5,
            color=colour if filled else t["surface"],
            edgecolors=colour,
            linewidths=1.6,
        )
        ax.annotate(
            f"{row['vs_AlmgrenChriss']:+.2f}  [{row['ci_low']:.2f}, {row['ci_high']:.2f}]",
            (row["ci_high"], y + 0.3),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=t["ink2"],
        )
    ax.axvline(0.0, color=t["ink"], lw=1.2, zorder=2)
    ax.axvline(0.5, color=t["muted"], lw=1.0, ls=(0, (2, 2)), zorder=2)
    top = len(E1_ROWS) - 0.35
    ax.text(-0.02, top, "optimum", ha="right", color=t["ink2"], fontsize=8.5)
    ax.text(0.52, top, "E1 tolerance (0.5 bps)", ha="left", color=t["muted"], fontsize=8.5)
    ax.set_yticks(range(len(E1_ROWS)), [r[0] for r in E1_ROWS[::-1]])
    ax.set_ylim(-0.5, len(E1_ROWS) - 0.1)
    ax.set_xlim(-0.35, 3.0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("objective above the optimum, bps (test block; dots = training seeds)")
    ax.set_title("E1: how far each learner ends from the provable optimum")
    return save(fig, "e1_gap_to_optimum", mode)


def fig_e1_schedule(t: Dict[str, str], mode: str) -> Path:
    """Mean trade schedule (share of the order per step) against Almgren-Chriss and TWAP."""

    def mean_schedule(name: str, agent: str) -> np.ndarray:
        s = read(name, "schedules.csv")
        s = s[s["agent"] == agent]
        return s.groupby("step")["fraction"].mean().sort_index().to_numpy() * 100

    steps = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(steps, mean_schedule("e1", "TWAP"), color=t["muted"], ls=(0, (4, 3)), label="TWAP")
    ax.plot(steps, mean_schedule("e1", "AlmgrenChriss"), color=t["ink"], label="Almgren-Chriss")
    series = [
        ("PPO, 2M (E1)", "e1", "PPO", (0, (5, 2))),
        ("PPO, 5M (E1b)", "e1b_5m_10seed", "PPO", "-"),
        ("DoubleDQN, 2M (E1)", "e1", "DoubleDQN", "-"),
        ("VecPPO tuned", "e1t_C", "VecPPO", "-"),
    ]
    for label, name, agent, ls in series:
        ax.plot(steps, mean_schedule(name, agent), color=t[agent], ls=ls, label=label)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("step (20 decision steps)")
    ax.set_ylabel("share of the order traded, %")
    ax.legend(loc="upper right", fontsize=9, ncols=2)
    ax.set_title("E1: mean schedule across seeds, test block")
    return save(fig, "e1_schedule", mode)


def compute_frontier() -> Path:
    """Almgren-Chriss at several risk aversions on the E1 test block (5,000 antithetic)."""
    from rl_execution.backtest import compare_strategies
    from rl_execution.baselines import AlmgrenChriss
    from rl_execution.config import ExecutionConfig, ac_risk_aversion
    from rl_execution.envs import ExecutionEnv
    from rl_execution.experiments import get_regime

    market, execution = get_regime("abm_ac_optimal"), ExecutionConfig()
    matched = ac_risk_aversion(market, execution)
    multiples = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    results = compare_strategies(
        lambda: ExecutionEnv(market, execution),
        {k: AlmgrenChriss(risk_aversion=matched * k) for k in multiples},
        n_episodes=5000,
        base_seed=2_000_000,
        antithetic=True,
        progress=False,
    )
    rows = []
    for k, res in results.items():
        is_bps = np.array([m["implementation_shortfall_bps"] for m in res.episode_metrics])
        j_bps = np.array([m["objective_bps"] for m in res.episode_metrics])
        # the same estimators as the report's comparison table (sample sd, ddof=1)
        rows.append(
            {
                "lambda_multiple": k,
                "IS_bps": float(is_bps.mean()),
                "IS_std": float(is_bps.std(ddof=1)),
                "J_bps": float(j_bps.mean()),
            }
        )
    FRONTIER.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FRONTIER, index=False)
    return FRONTIER


def fig_frontier(t: Dict[str, str], mode: str) -> Path:
    """Mean cost against its spread: AC's frontier over risk aversion, and where agents sit."""
    front = pd.read_csv(FRONTIER)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(
        front["IS_std"],
        front["IS_bps"],
        color=t["ink"],
        lw=1.4,
        zorder=2,
        label="Almgren-Chriss at other risk aversions",
    )
    ax.scatter(front["IS_std"], front["IS_bps"], s=12, color=t["ink"], zorder=3)
    labels = {0.0: "TWAP (0)", 0.5: "0.5λ", 1.0: "λ: the optimum", 1.5: "1.5λ", 2.0: "2λ"}
    for _, r in front.iterrows():
        k = float(r["lambda_multiple"])
        if k in labels:
            ax.annotate(
                labels[k],
                (r["IS_std"], r["IS_bps"]),
                xytext=(-8, 8) if k == 0 else (-10, -4),
                textcoords="offset points",
                ha="right",
                va="top" if k else "bottom",
                fontsize=8.5,
                color=t["ink2"],
            )
    best = front[front["lambda_multiple"] == 1.0].iloc[0]
    ax.scatter([best["IS_std"]], [best["IS_bps"]], s=60, color=t["ink"], zorder=4)
    points = [
        ("DoubleDQN 2M", "e1", "DoubleDQN", True),
        ("PPO 2M", "e1", "PPO", False),
        ("PPO 5M", "e1b_5m_10seed", "PPO", True),
        ("VecPPO tuned", "e1t_C", "VecPPO", True),
    ]
    for label, name, agent, filled in points:
        row = agent_row(name, agent)
        ax.scatter(
            [row["IS_std"]],
            [row["IS_bps"]],
            s=52,
            zorder=5,
            color=t[agent] if filled else t["surface"],
            edgecolors=t["surface"] if filled else t[agent],
            linewidths=1.6,
            label=label,
        )
    ax.set_xlim(54, 101)
    ax.set_ylim(18.3, 26.6)
    ax.set_xlabel("spread of implementation shortfall: sd across episodes, bps")
    ax.set_ylabel("mean implementation shortfall, bps")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("E1: cost against risk (λ = the risk aversion the agents are rewarded with)")
    return save(fig, "e1_frontier", mode)


# ----------------------------------------------------------------------------- E2
def fig_e2(t: Dict[str, str], mode: str) -> Path:
    """Improvement over AC, and over SignalRule, against the signal's IC."""
    sb3, tuned = read("e2", "sweep.csv"), read("e2_tuned", "sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, bench in zip(axes, ["AlmgrenChriss", "SignalRule"]):
        ax.axvspan(0.05, 0.10, color=t["band"], zorder=0, lw=0)
        ax.axhline(0.0, color=t["ink"], lw=1.1, zorder=1)
        lines = [
            ("PPO (SB3, 5M)", sb3, "PPO", t["PPO"]),
            ("VecPPO tuned", tuned, "VecPPO", t["VecPPO"]),
        ]
        if bench == "AlmgrenChriss":
            lines.append(("SignalRule (2-line rule)", sb3, "SignalRule", t["ink2"]))
        for label, frame, agent, colour in lines:
            d = frame[(frame["benchmark"] == bench) & (frame["strategy"] == agent)]
            d = d.sort_values("ic")
            ls = (0, (1.5, 2)) if agent == "SignalRule" else "-"
            ax.errorbar(
                d["ic"],
                d["improvement"],
                yerr=[d["improvement"] - d["ci_low"], d["ci_high"] - d["improvement"]],
                color=colour,
                ls=ls,
                marker="o",
                ms=5,
                capsize=0,
                elinewidth=1.4,
                label=label,
                zorder=3,
            )
        ax.set_xticks([0, 0.025, 0.05, 0.10, 0.20], ["0", ".025", ".05", ".10", ".20"])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("information coefficient of the order-book signal")
    axes[0].set_ylabel("cheaper than Almgren-Chriss, bps")
    axes[0].set_title("E2: saving over Almgren-Chriss")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].set_ylabel("cheaper than SignalRule, bps")
    axes[1].set_title("E2: learned policy vs the two-line SignalRule")
    axes[1].set_ylim(-1.0, 1.6)
    axes[1].text(0.2, 0.06, "SignalRule", ha="right", fontsize=8.5, color=t["ink2"])
    axes[1].legend(loc="upper left", fontsize=9)
    for ax in axes:
        bottom = ax.get_ylim()[0]
        ax.text(
            0.075,
            bottom,
            "real signals\n",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=t["muted"],
        )
    fig.tight_layout(w_pad=3.0)
    return save(fig, "e2_signal_sweep", mode)


# ----------------------------------------------------------------------------- E3
def fig_e3(t: Dict[str, str], mode: str) -> Path:
    """Gap to AC in each regime, in-distribution and held out; off-scale values labelled."""
    from rl_execution.experiments import HELD_OUT, IN_DISTRIBUTION

    table = read("e3", "comparison.csv")
    regimes = IN_DISTRIBUTION + HELD_OUT
    lo, hi = -6.5, 10.5
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ys = {r: len(regimes) - 1 - i - (0.6 if r in HELD_OUT else 0.0) for i, r in enumerate(regimes)}
    styles = [
        ("TWAP", t["muted"], "o", t["surface"]),
        ("SignalRule", t["ink2"], "D", t["ink2"]),
        ("PPO", t["PPO"], "o", t["PPO"]),
    ]
    off_scale: Dict[str, list] = {}
    for agent, colour, marker, face in styles:
        for i, regime in enumerate(regimes):
            r = table[(table["regime"] == regime) & (table["strategy"] == agent)].iloc[0]
            y, x = ys[regime], float(r["vs_AlmgrenChriss"])
            if agent == "PPO" and r["ci_low"] < hi:
                ax.plot(
                    [max(r["ci_low"], lo), min(r["ci_high"], hi)], [y] * 2, color=colour, lw=2.2
                )
            label = (agent if agent != "PPO" else "PPO, 10 seeds (95% CI)") if i == 0 else None
            if x > hi:  # drawn as an arrow at the edge, its value written beside it
                ax.scatter([hi], [y], marker=">", s=40, color=colour, zorder=4, label=label)
                off_scale.setdefault(regime, []).append(f"{agent} {x:+.1f}")
                continue
            ax.scatter(
                [x],
                [y],
                marker=marker,
                s=34,
                color=face,
                edgecolors=colour,
                linewidths=1.4,
                zorder=4,
                label=label,
            )
    for regime, values in off_scale.items():
        ax.annotate(
            "off scale: " + ", ".join(values),
            (hi, ys[regime]),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            color=t["ink2"],
        )
    ax.axvline(0.0, color=t["ink"], lw=1.1, zorder=1)
    ax.axhline(len(regimes) - len(IN_DISTRIBUTION) - 0.8, color=t["axis"], lw=0.8)
    for group, name in ((IN_DISTRIBUTION, "inside the training family"), (HELD_OUT, "held out")):
        ax.text(lo + 0.1, ys[group[0]] + 0.5, name, va="bottom", fontsize=8.5, color=t["muted"])
    ax.set_yticks([ys[r] for r in regimes], [r.replace("_", " ") for r in regimes])
    ax.set_xlim(lo, hi)
    ax.set_ylim(ys[HELD_OUT[-1]] - 0.6, ys[IN_DISTRIBUTION[0]] + 1.1)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("objective vs Almgren-Chriss in that regime, bps (negative = cheaper)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("E3: one policy trained on a randomised family, scored inside and outside it")
    return save(fig, "e3_robustness", mode)


# ----------------------------------------------------------------------------- Phase 4
def fig_throughput(t: Dict[str, str], mode: str) -> Path:
    """GPU throughput against batch size, with the CPU baselines as reference lines.

    The batched env's and VecPPO's own CPU rows are left out: on the lab machine's shared
    CPU they changed by up to 37x between two runs (see RESULTS.md).
    """
    bench = read("bench", "throughput.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    series = [
        ("batched env, GPU", "batched env", t["DoubleDQN"]),
        ("VecPPO training, GPU", "VecPPO", t["VecPPO"]),
    ]
    for label, what, colour in series:
        d = bench[(bench["what"] == what) & (bench["device"] == "cuda")].sort_values("n_envs")
        ax.plot(d["n_envs"], d["steps_per_s"], color=colour, marker="o", ms=4, label=label)
    for what, colour, text in [
        ("numpy env", t["muted"], "numpy env, 1 core"),
        ("SB3 PPO", t["PPO"], "SB3 PPO training, 16 envs"),
    ]:
        v = float(bench.loc[bench["what"] == what, "steps_per_s"].iloc[0])
        ax.axhline(v, color=colour, lw=1.0, ls=(0, (1.5, 2)))
        ax.text(3e5, v * 1.15, text, ha="right", fontsize=8.5, color=t["ink2"])
    ax.axhline(1e6, color=t["ink"], lw=0.8, ls=(0, (6, 3)))
    ax.text(3e5, 1.15e6, "target: 1M env steps/s", ha="right", fontsize=8.5, color=t["ink2"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(150, 4e5)
    ax.set_xlabel("environments stepped together (N)")
    ax.set_ylabel("environment steps per second")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Phase 4: throughput on one GPU (lab machine, RTX 4000 Ada)")
    return save(fig, "throughput", mode)


FIGS = [fig_e1_gaps, fig_e1_schedule, fig_frontier, fig_e2, fig_e3, fig_throughput]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--compute-frontier", action="store_true")
    args = parser.parse_args()
    if args.compute_frontier or not FRONTIER.exists():
        print(f"wrote {compute_frontier().relative_to(ROOT)}")
    for mode, theme in THEMES.items():
        setup(theme)
        for fig in FIGS:
            print(f"wrote {fig(theme, mode).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
