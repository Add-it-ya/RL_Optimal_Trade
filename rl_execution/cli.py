"""Command-line interface: ``rlx train``, ``rlx evaluate`` and ``rlx report``.

Examples
--------
    rlx train --agent ppo --timesteps 200000 --seeds 0-4 --regime abm_ac_optimal --experiment e1
    rlx evaluate --experiment e1 --episodes 2000 --antithetic
    rlx report --experiment e1

``train``
    Trains one agent per seed into ``runs/<experiment>/<Agent>/seed_<k>/``.  A seed whose
    folder is already complete is skipped, so an interrupted job can simply be re-launched.
``evaluate``
    Scores every run of the experiment, plus the baselines, on a block of test seeds that
    training never sees, and writes ``results/<experiment>/`` (per-episode CSV, summary
    table, figures).
``report``
    Aggregates ``results/<experiment>/`` into ``REPORT.md``: one row per strategy with the
    mean objective (implementation shortfall + Almgren-Chriss risk penalty), the level,
    spread and tail of implementation shortfall, how many steps it trades in, and the paired
    gap to Almgren-Chriss with a 95% CI (across training seeds for RL agents) and
    Holm-adjusted p-values.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rl_execution.agents import (
    DISPLAY_NAMES,
    AgentStrategy,
    normalize_agent_name,
    required_action_type,
)
from rl_execution.config import ActionType, ExecutionConfig, RunConfig, Side
from rl_execution.utils.config_io import load_run_config
from rl_execution.utils.io import RESULTS_DIR, RUNS_DIR, ensure_dir, load_json, save_json
from rl_execution.utils.warnings import configure_cli_warnings

#: First seed of the test block.  Training resets draw from the env's own seeded RNG, so
#: they never land on these explicit evaluation seeds.
TEST_BASE_SEED = 2_000_000

#: A trained seed whose policy trades in fewer effective steps than this (inverse
#: Herfindahl of its schedule; TWAP = horizon, one block trade = 1) is flagged degenerate.
DEGENERATE_EFFECTIVE_STEPS = 5.0

#: The schedule ``L1_vs_AC`` is measured against: the closed-form optimum of the reward.
SCHEDULE_REFERENCE = "AlmgrenChriss"


def parse_seeds(spec: str) -> List[int]:
    """``"3"`` -> [3]; ``"0-4"`` -> [0, 1, 2, 3, 4]; ``"0,2,7"`` -> [0, 2, 7]."""
    seeds: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"No seeds in {spec!r}")
    return seeds


# --------------------------------------------------------------------------- train
def _train_config(args) -> RunConfig:
    """Merge a YAML RunConfig (if any) with the CLI flags; explicit flags win."""
    if args.config:
        rc = load_run_config(args.config, validate=False)
    else:
        rc = RunConfig(
            execution=ExecutionConfig(
                total_inventory=args.inventory, horizon=args.horizon, side=Side(args.side)
            )
        )
    if args.agent:
        rc.agent = normalize_agent_name(args.agent)
    if args.timesteps is not None:
        rc.timesteps = args.timesteps
    if args.regime is not None:
        rc.regime, rc.randomized = args.regime, False
    if args.randomized:
        rc.regime, rc.randomized = None, True
    if args.experiment is not None:
        rc.experiment = args.experiment
    for flag in ("n_envs", "eval_every", "val_episodes"):
        if getattr(args, flag) is not None:
            setattr(rc, flag, getattr(args, flag))
    rc.device = args.device or rc.device
    return rc.validate()


def cmd_train(args) -> None:
    from rl_execution.training import run_dir, save_run, train_run

    base = _train_config(args)
    market = "domain-randomised" if base.randomized else (base.regime or "config market")
    for seed in parse_seeds(args.seeds):
        rc = replace(base, seed=seed)
        out = run_dir(args.runs_dir, rc.experiment, rc.agent, seed)
        if (out / "provenance.json").exists() and not args.overwrite:
            print(f"skip   {out} (already trained; --overwrite to redo)")
            continue
        print(f"train  {DISPLAY_NAMES[rc.agent]} seed={seed} {rc.timesteps:,} steps on {market}")
        t0 = time.time()
        result = train_run(rc, progress=args.progress)
        save_run(result, rc, out)
        selected = result.selected()
        assert selected is not None  # train_run always selects on validation seeds
        print(
            f"       done in {time.time() - t0:.0f}s; validation objective "
            f"{result.validation[0]['objective_bps']:.2f} bps untrained -> "
            f"{selected['objective_bps']:.2f} bps at step {result.selected_step:,} (selected)"
        )
        print(f"       -> {out}")


# --------------------------------------------------------------------------- evaluate
def _load_runs(runs_root: Path):
    """``(strategies, action_types, runs_meta, exec_config, regimes)`` for every run."""
    from rl_execution.training import find_runs, load_run

    strategies: Dict[str, AgentStrategy] = {}
    action_types: Dict[str, ActionType] = {}
    meta: List[dict] = []
    exec_config: Optional[ExecutionConfig] = None
    regimes = set()
    for d in find_runs(runs_root):
        agent, rc = load_run(d)
        name = f"{DISPLAY_NAMES[normalize_agent_name(rc.agent)]}/s{rc.seed}"
        strategies[name] = AgentStrategy(agent, name)
        # train_run builds the env from the agent type; config.yaml keeps the CLI default
        action_types[name] = required_action_type(rc.agent)
        prov = load_json(d / "provenance.json")
        meta.append(
            {
                "strategy": name,
                "run_dir": str(d),
                "git_sha": prov.get("git_sha"),
                "git_dirty": prov.get("git_dirty"),
                "weights_sha256": prov.get("weights_sha256"),
            }
        )
        exec_config = exec_config or rc.execution
        regimes.add(rc.regime)
    return strategies, action_types, meta, exec_config, regimes


def _agent_of(strategy: str) -> str:
    """``"PPO/s3"`` -> ``"PPO"``; baselines map to themselves."""
    return strategy.split("/s")[0]


def _seed_of(strategy: str) -> float:
    return float(strategy.split("/s")[1]) if "/s" in strategy else float("nan")


def schedules_frame(results) -> pd.DataFrame:
    """Mean schedule of every (regime, strategy), as the fraction of the order per step.

    Episodes that finish early are zero-padded (see ``normalised_schedule``), so one
    strategy's fractions sum to the share of the order it executed: 1 for a full fill.
    """
    from rl_execution.backtest import normalised_schedule

    rows = []
    for regime, by_strat in results.items():
        for name, res in by_strat.items():
            for step, fraction in enumerate(normalised_schedule(res)):
                rows.append(
                    {
                        "regime": regime,
                        "strategy": name,
                        "agent": _agent_of(name),
                        "seed": _seed_of(name),
                        "step": step,
                        "fraction": float(fraction),
                    }
                )
    return pd.DataFrame(rows, columns=["regime", "strategy", "agent", "seed", "step", "fraction"])


def schedule_l1(
    schedules: pd.DataFrame, reference: str = SCHEDULE_REFERENCE
) -> Dict[Tuple[str, str], float]:
    """``{(regime, strategy): L1 distance of its mean schedule from the reference's}``."""
    from rl_execution.backtest import schedule_gap

    def fractions(frame: pd.DataFrame) -> np.ndarray:
        return frame.sort_values("step")["fraction"].to_numpy(dtype=float)

    out: Dict[Tuple[str, str], float] = {}
    for regime, reg in schedules.groupby("regime", sort=False):
        ref = reg[reg["strategy"] == reference]
        if ref.empty:
            continue
        for name, grp in reg.groupby("strategy", sort=False):
            out[(str(regime), str(name))] = schedule_gap(fractions(grp), fractions(ref))
    return out


def cmd_evaluate(args) -> None:
    from rl_execution import viz
    from rl_execution.backtest import pool_results
    from rl_execution.experiments import (
        HELD_OUT,
        IN_DISTRIBUTION,
        build_baselines,
        evaluate_across_regimes,
        list_regimes,
        regime_results_frame,
        resolve_regimes,
    )

    runs_root = Path(args.runs_dir) / args.experiment
    strategies, action_types, runs_meta, exec_config, trained_on = _load_runs(runs_root)
    if not strategies:
        print(f"No runs under {runs_root}; evaluating the baselines only.")
    exec_config = replace(exec_config or ExecutionConfig(), action_type=ActionType.CONTINUOUS)

    if args.regimes:
        regimes = resolve_regimes(args.regimes)
    elif strategies and trained_on == {None}:
        regimes = IN_DISTRIBUTION + HELD_OUT  # domain-randomised experiments: the E3 split
    elif strategies and None not in trained_on:
        regimes = sorted(trained_on)  # single-regime experiments: evaluate where trained
    else:
        regimes = list_regimes(include_brownian=args.include_brownian)

    all_strategies = {**build_baselines(), **strategies}
    print(
        f"Evaluating {len(all_strategies)} strategies x {len(regimes)} regimes, "
        f"{args.episodes} episodes each (test seeds from {args.base_seed:,}"
        f"{', antithetic' if args.antithetic else ''}) ..."
    )
    results = evaluate_across_regimes(
        all_strategies,
        exec_config,
        regimes=regimes,
        action_types=action_types,
        n_episodes=args.episodes,
        base_seed=args.base_seed,
        antithetic=args.antithetic,
        progress=False,
    )

    out = ensure_dir(Path(args.out) if args.out else RESULTS_DIR / args.experiment)
    rows = []
    for regime, by_strat in results.items():
        for name, res in by_strat.items():
            for i, m in enumerate(res.episode_metrics):
                rows.append(
                    {
                        "regime": regime,
                        "strategy": name,
                        "agent": _agent_of(name),
                        "seed": _seed_of(name),
                        "episode": i,
                        **m,
                    }
                )
    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
    schedules_frame(results).to_csv(out / "schedules.csv", index=False)
    regime_results_frame(results).to_csv(out / "summary.csv", index=False)
    save_json(
        {
            "experiment": args.experiment,
            "runs_dir": str(runs_root),
            "trained_on": sorted("randomized" if r is None else r for r in trained_on),
            "regimes": regimes,
            "episodes": args.episodes,
            "base_seed": args.base_seed,
            "antithetic": bool(args.antithetic),
            "runs": runs_meta,
            "params": {
                regime: {name: res.params for name, res in by_strat.items() if res.params}
                for regime, by_strat in results.items()
            },
        },
        out / "meta.json",
    )

    figures = ensure_dir(out / "figures")
    for regime, by_strat in results.items():
        pooled = {}
        for agent in dict.fromkeys(_agent_of(n) for n in by_strat):
            members = [r for n, r in by_strat.items() if _agent_of(n) == agent]
            pooled[agent] = members[0] if len(members) == 1 else pool_results(members, agent)
        summary = pd.DataFrame(
            [
                {
                    "strategy": name,
                    "IS_bps": res.aggregate["implementation_shortfall_bps_mean"],
                    "IS_std": res.aggregate["implementation_shortfall_bps_std"],
                }
                for name, res in pooled.items()
            ]
        )
        viz.plot_execution_schedule(pooled, save=str(figures / f"{regime}_schedule.png"))
        viz.plot_inventory_decay(pooled, save=str(figures / f"{regime}_inventory.png"))
        viz.plot_is_distribution(pooled, save=str(figures / f"{regime}_is_distribution.png"))
        viz.plot_risk_vs_cost(summary, save=str(figures / f"{regime}_risk_vs_cost.png"))
    print(f"Wrote {out / 'episodes.csv'}, schedules.csv, summary.csv, meta.json and figures/")
    print(f"Next: rlx report --experiment {args.experiment}")


# --------------------------------------------------------------------------- report
def _pvalue(diffs: np.ndarray) -> float:
    """Two-sided one-sample t-test p-value that ``diffs`` has mean zero."""
    from rl_execution.backtest.engine import TIE_TOLERANCE

    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return float("nan")
    if float(np.max(np.abs(d))) <= TIE_TOLERANCE:
        return 1.0
    if float(d.std(ddof=1)) == 0.0:
        return 1.0 if np.isclose(d.mean(), 0.0) else 0.0
    from scipy import stats

    return float(stats.ttest_1samp(d, 0.0).pvalue)


def comparison_table(
    episodes: pd.DataFrame,
    benchmark: str = "AlmgrenChriss",
    metric: str = "objective_bps",
    block: int = 1,
    n_boot: int = 5_000,
    schedules: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One row per (regime, agent): objective, IS level / spread / tail, and the paired gap.

    ``vs_<benchmark>`` is the mean of ``metric - metric_benchmark`` formed episode by
    episode -- every strategy faced the same test seeds (common random numbers), so the
    shared price risk cancels.  For an agent trained on several seeds the training seed is
    the unit of analysis: each seed contributes its mean paired gap, and the CI and p-value
    are taken across seeds.  A single-seed agent or a baseline falls back to the paired
    episode-level bootstrap (``block=2`` for antithetic pairs).  Columns missing from
    ``episodes`` are reported as NaN.

    ``L1_vs_AC`` is the L1 distance between the strategy's mean schedule and
    Almgren-Chriss's (see :func:`schedule_l1`; the median across seeds for a multi-seed
    agent), taken from ``schedules`` -- NaN when that is not given.  A strategy whose every
    paired difference is within ``TIE_TOLERANCE`` of zero is reported as an exact tie.
    """
    from rl_execution.backtest.engine import TIE_TOLERANCE
    from rl_execution.metrics.stats import adjust_pvalues, bootstrap_ci, paired_bootstrap_ci

    nan = float("nan")
    l1 = schedule_l1(schedules) if schedules is not None else {}

    def column(frame: pd.DataFrame, name: str) -> np.ndarray:
        if name not in frame:
            return np.full(len(frame), nan)
        return frame.sort_values("episode")[name].to_numpy(dtype=float)

    rows = []
    for regime, reg in episodes.groupby("regime", sort=False):
        bench = reg[reg["strategy"] == benchmark]
        if bench.empty:
            continue
        bench_m = column(bench, metric)
        for agent, grp in reg.groupby("agent", sort=False):
            per_seed_gap, per_seed_l1 = [], []
            n_degenerate = 0
            max_abs_diff = 0.0
            last_m: np.ndarray = np.empty(0)
            for name, run in grp.groupby("strategy", sort=False):
                m = column(run, metric)
                n = min(m.size, bench_m.size)
                per_seed_gap.append(float(np.mean(m[:n] - bench_m[:n])))
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(m[:n] - bench_m[:n]))))
                per_seed_l1.append(l1.get((str(regime), str(name)), nan))
                last_m = m[:n]
                n_degenerate += int(
                    np.nanmean(column(run, "effective_steps")) < DEGENERATE_EFFECTIVE_STEPS
                )
            is_bps = column(grp, "implementation_shortfall_bps")
            trained = bool(grp["seed"].notna().any())
            row = {
                "regime": regime,
                "strategy": agent,
                "n_seeds": len(per_seed_gap) if trained else "—",
                "n_degenerate": n_degenerate if trained else "—",
                "J_bps": float(np.mean(column(grp, "objective_bps"))),
                "IS_bps": float(is_bps.mean()),
                "IS_std": float(is_bps.std(ddof=1)) if is_bps.size > 1 else nan,
                "IS_p95": float(np.quantile(is_bps, 0.95)),
                "eff_steps": float(np.mean(column(grp, "effective_steps"))),
                "L1_vs_AC": (
                    float(np.nanmedian(per_seed_l1)) if not np.all(np.isnan(per_seed_l1)) else nan
                ),
                f"vs_{benchmark}": float(np.mean(per_seed_gap)),
                "ci_low": nan,
                "ci_high": nan,
                "p_value": nan,
            }
            if agent != benchmark:
                if max_abs_diff <= TIE_TOLERANCE:  # trades identically to the benchmark
                    row[f"vs_{benchmark}"], row["ci_low"], row["ci_high"] = 0.0, 0.0, 0.0
                    row["p_value"] = 1.0
                elif len(per_seed_gap) >= 2:
                    row["ci_low"], row["ci_high"] = bootstrap_ci(per_seed_gap, n_boot=n_boot, rng=0)
                    row["p_value"] = _pvalue(np.asarray(per_seed_gap))
                else:
                    n = last_m.size
                    row["ci_low"], row["ci_high"] = paired_bootstrap_ci(
                        last_m, bench_m[:n], n_boot=n_boot, rng=0, block=block
                    )
                    diff = last_m - bench_m[:n]
                    if block > 1 and diff.size // block >= 2:
                        diff = diff[: (diff.size // block) * block].reshape(-1, block).mean(axis=1)
                    row["p_value"] = _pvalue(diff)
            rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    tested = table["p_value"].notna() if not table.empty else pd.Series(dtype=bool)
    if tested.any():
        _, p_adj = adjust_pvalues(table.loc[tested, "p_value"].to_numpy(), method="holm")
        table.loc[tested, "p_holm"] = p_adj
    return table


def split_degradation(
    episodes: pd.DataFrame,
    splits: Mapping[str, Sequence[str]],
    benchmark: str = "AlmgrenChriss",
    metric: str = "objective_bps",
    n_boot: int = 5_000,
) -> pd.DataFrame:
    """How much each strategy's gap to the benchmark grows out of distribution (E3).

    For every run, the mean paired gap ``metric - metric_benchmark`` of each regime is
    averaged, one weight per regime, over ``splits["in_distribution"]`` and over
    ``splits["held_out"]``; ``degradation`` is held-out minus in-distribution, in the
    metric's units.  For an agent trained on several seeds the columns are means across
    seeds and the 95% CI of ``degradation`` is taken across seeds; a baseline (one run)
    gets point values.  Only regimes present in ``episodes`` count.
    """
    from rl_execution.metrics.stats import bootstrap_ci

    nan = float("nan")
    gaps: Dict[str, Dict[str, float]] = {}  # strategy -> {regime: mean paired gap}
    agent_of, trained = {}, {}
    for regime, reg in episodes.groupby("regime", sort=False):
        bench = reg[reg["strategy"] == benchmark].sort_values("episode")[metric]
        b = bench.to_numpy(dtype=float)
        if b.size == 0:
            continue
        for name, run in reg.groupby("strategy", sort=False):
            m = run.sort_values("episode")[metric].to_numpy(dtype=float)
            n = min(m.size, b.size)
            gaps.setdefault(str(name), {})[str(regime)] = float(np.mean(m[:n] - b[:n]))
            agent_of[str(name)] = str(run["agent"].iloc[0])
            trained[str(name)] = bool(run["seed"].notna().any())

    per_agent: Dict[str, List[Tuple[float, float]]] = {}
    for name, by_regime in gaps.items():
        inside = [g for r, g in by_regime.items() if r in splits["in_distribution"]]
        outside = [g for r, g in by_regime.items() if r in splits["held_out"]]
        if inside and outside:
            per_agent.setdefault(agent_of[name], []).append(
                (float(np.mean(inside)), float(np.mean(outside)))
            )

    rows = []
    for agent, pairs in per_agent.items():
        if agent == benchmark:
            continue
        ins, outs = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        degradation = outs - ins
        is_trained = any(trained[n] for n in gaps if agent_of[n] == agent)
        ci = bootstrap_ci(degradation, n_boot=n_boot, rng=0) if len(pairs) >= 2 else (nan, nan)
        rows.append(
            {
                "strategy": agent,
                "n_seeds": len(pairs) if is_trained else "—",
                f"vs_{benchmark}_in": float(ins.mean()),
                f"vs_{benchmark}_held_out": float(outs.mean()),
                "degradation": float(degradation.mean()),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, decimals: Optional[Dict[str, int]] = None) -> str:
    """Markdown table; floats get one decimal unless ``decimals`` names the column."""
    decimals = decimals or {}

    def fmt(v, col):
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            if col in decimals:
                return f"{v:.{decimals[col]}f}"
            return f"{v:.3g}" if abs(v) < 1e-2 and v != 0 else f"{v:.1f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += [
        "| " + " | ".join(fmt(v, c) for v, c in zip(row, df.columns)) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def cmd_report(args) -> None:
    from rl_execution import viz
    from rl_execution.experiments import SPLITS, get_regime, outside_randomized_support

    res_dir = Path(args.results) if args.results else RESULTS_DIR / args.experiment
    episodes = pd.read_csv(res_dir / "episodes.csv")
    meta = load_json(res_dir / "meta.json")
    block = 2 if meta.get("antithetic") else 1
    sched_path = res_dir / "schedules.csv"
    schedules = pd.read_csv(sched_path) if sched_path.exists() else None
    table = comparison_table(
        episodes, benchmark=args.benchmark, metric=args.metric, block=block, schedules=schedules
    )
    table.to_csv(res_dir / "comparison.csv", index=False)
    randomized = meta.get("trained_on") == ["randomized"]
    robustness = (
        split_degradation(episodes, SPLITS, benchmark=args.benchmark, metric=args.metric)
        if randomized
        else pd.DataFrame()
    )

    # validation curves and the checkpoint each run kept (validation seeds, not test seeds)
    curves, selection = {}, []
    for run in meta.get("runs", []):
        run_path = Path(run["run_dir"])
        if (run_path / "val_curve.csv").exists():
            curves[run["strategy"]] = pd.read_csv(run_path / "val_curve.csv")
        prov = load_json(run_path / "provenance.json") if run_path.exists() else {}
        if "selected_step" in prov:
            selection.append(
                {
                    "run": run["strategy"],
                    "J_untrained": prov["initial_val_objective_bps"],
                    "J_selected": prov["selected_val_objective_bps"],
                    "selected_step": f"{prov['selected_step']:,}",
                    "J_final": prov["final_val_objective_bps"],
                }
            )
    if curves:
        ensure_dir(res_dir / "figures")
        viz.plot_validation_curves(curves, save=str(res_dir / "figures" / "validation_curves.png"))

    shas = sorted({r["git_sha"] for r in meta.get("runs", []) if r.get("git_sha")})
    n_dirty = sum(bool(r.get("git_dirty")) for r in meta.get("runs", []))
    parts = [
        f"# Results: `{meta['experiment']}`",
        "",
        f"- Test episodes per strategy: {meta['episodes']} (seeds from {meta['base_seed']:,}"
        f"{', antithetic pairs' if block == 2 else ''})",
        f"- Trained runs: {len(meta.get('runs', []))}; code version(s): "
        + (", ".join(s[:10] for s in shas) or "n/a")
        + (f" ({n_dirty} trained with uncommitted changes)" if n_dirty else ""),
        "- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus "
        "the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how "
        "many steps the order is effectively spread over (20 = TWAP, 1 = one block trade).",
        "- `L1_vs_AC` is the L1 distance between the strategy's mean schedule (fraction of the "
        "order traded at each step) and Almgren-Chriss's: 0 = the same shape, 2 = disjoint. "
        "For multi-seed agents it is the median across seeds.",
        f"- `vs_{args.benchmark}` is the mean paired difference in `{args.metric}` (negative = "
        f"better than {args.benchmark}). For multi-seed agents the 95% CI and p-value are "
        "across training seeds; `p_holm` is Holm-adjusted over every row of every table.",
        f"- `n_degenerate` counts trained seeds whose policy trades in fewer than "
        f"{DEGENERATE_EFFECTIVE_STEPS:g} effective steps -- a block trade, not an execution "
        "schedule. They are reported, never dropped.",
        "",
    ]
    cols = ["strategy", "n_seeds", "n_degenerate", "J_bps", "IS_bps", "IS_std", "IS_p95"]
    cols += ["eff_steps", "L1_vs_AC"]
    cols += [f"vs_{args.benchmark}", "ci_low", "ci_high", "p_holm"]
    # E1 is judged on sub-bps gaps and an L1 bar of 0.10: print enough digits to read them
    precise = {c: 2 for c in ("J_bps", f"vs_{args.benchmark}", "ci_low", "ci_high")}
    precise["L1_vs_AC"] = 3
    fitted = meta.get("params", {})
    regimes = list(dict.fromkeys(table["regime"])) if not table.empty else []
    if randomized:  # E3: in-distribution and held-out regimes reported separately
        rest = [r for r in regimes if r not in SPLITS["in_distribution"] + SPLITS["held_out"]]
        sections = [
            ("In distribution", [r for r in regimes if r in SPLITS["in_distribution"]]),
            ("Held out", [r for r in regimes if r in SPLITS["held_out"]]),
            ("Other regimes", rest),
        ]
    else:
        sections = [("", regimes)]
    for label, members in sections:
        if not members:
            continue
        heading = "###" if label else "##"
        if label:
            parts += [f"## {label}", ""]
        for regime in members:
            sub = table[table["regime"] == regime]
            title = regime
            axes = outside_randomized_support(get_regime(regime)) if randomized else []
            if axes:
                title += f" (outside the training family on: {', '.join(axes)})"
            table_md = _markdown_table(sub[cols].sort_values("J_bps"), decimals=precise)
            parts += [f"{heading} {title}", "", table_md]
            parts.append("")
            beta = fitted.get(regime, {}).get("SignalRule", {}).get("beta")
            if beta is not None:
                parts += [f"SignalRule coefficient fitted on validation seeds: beta = {beta}.", ""]
            for fig in ("schedule", "risk_vs_cost", "is_distribution"):
                if (res_dir / "figures" / f"{regime}_{fig}.png").exists():
                    parts.append(f"![{regime} {fig}](figures/{regime}_{fig}.png)")
            parts.append("")
    if not robustness.empty:
        parts += [
            "## Robustness: in distribution vs held out",
            "",
            f"Mean paired gap to {args.benchmark} in `{args.metric}`, one weight per regime, over "
            "the in-distribution and the held-out regimes above; `degradation` is held out "
            "minus in distribution (positive = worse out of distribution). For multi-seed "
            "agents the 95% CI is across training seeds.",
            "",
            _markdown_table(robustness.sort_values("degradation"), decimals={"degradation": 2}),
            "",
        ]
    if selection:
        parts += [
            "## Checkpoint selection",
            "",
            "Mean objective (bps) on the validation seeds before training, at the checkpoint "
            "each run kept, and at the end of training. The test seeds above were never used "
            "for selection.",
            "",
            _markdown_table(pd.DataFrame(selection)),
            "",
        ]
    if curves:
        parts += ["![validation curves](figures/validation_curves.png)", ""]
    (res_dir / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {res_dir / 'REPORT.md'} and comparison.csv")


# --------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlx", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train one agent for one or more seeds")
    t.add_argument("--agent", choices=["ppo", "dqn", "doubledqn"], default=None)
    t.add_argument("--config", default=None, help="YAML RunConfig; explicit flags override it")
    t.add_argument("--timesteps", type=int, default=None)
    t.add_argument("--regime", default=None, help="train on one named regime")
    t.add_argument("--randomized", action="store_true", help="train on the randomised family")
    t.add_argument("--seeds", default="0", help='e.g. "0", "0-9" or "0,3,5"')
    t.add_argument("--experiment", default=None, help="groups runs under runs/<experiment>/")
    t.add_argument("--runs-dir", default=str(RUNS_DIR))
    t.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    t.add_argument("--n-envs", type=int, default=None, help="PPO environments (default 16)")
    t.add_argument(
        "--eval-every", type=int, default=None, help="validation scoring interval in env steps"
    )
    t.add_argument("--val-episodes", type=int, default=None, help="episodes per validation score")
    t.add_argument("--side", default="sell", choices=["buy", "sell"])
    t.add_argument("--inventory", type=float, default=10_000.0)
    t.add_argument("--horizon", type=int, default=20)
    t.add_argument("--overwrite", action="store_true", help="retrain seeds that already exist")
    t.add_argument("--progress", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="score every run + baselines on held-out test seeds")
    e.add_argument("--experiment", required=True)
    e.add_argument("--runs-dir", default=str(RUNS_DIR))
    e.add_argument("--out", default=None, help="default: results/<experiment>")
    e.add_argument("--regimes", nargs="+", default=None)
    e.add_argument("--include-brownian", action="store_true")
    e.add_argument("--episodes", type=int, default=400)
    e.add_argument("--base-seed", type=int, default=TEST_BASE_SEED)
    e.add_argument("--antithetic", action="store_true", help="evaluate in Z / -Z pairs")
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("report", help="aggregate evaluation results into REPORT.md")
    r.add_argument("--experiment", required=True)
    r.add_argument("--results", default=None, help="default: results/<experiment>")
    r.add_argument("--benchmark", default="AlmgrenChriss")
    r.add_argument(
        "--metric",
        default="objective_bps",
        choices=["objective_bps", "implementation_shortfall_bps"],
        help="what the paired comparison is made on",
    )
    r.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    configure_cli_warnings()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
