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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rl_execution.agents import DISPLAY_NAMES, AgentStrategy, normalize_agent_name
from rl_execution.config import ActionType, ExecutionConfig, RunConfig, Side
from rl_execution.utils.config_io import load_run_config
from rl_execution.utils.io import RESULTS_DIR, RUNS_DIR, ensure_dir, load_json, save_json
from rl_execution.utils.warnings import configure_cli_warnings

#: First seed of the test block.  Training resets draw from the env's own seeded RNG, so
#: they never land on these explicit evaluation seeds.
TEST_BASE_SEED = 2_000_000


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
        agent, log = train_run(rc, progress=args.progress)
        save_run(agent, rc, out, log)
        tail = float(np.mean(log[-100:])) if log else float("nan")
        print(f"       done in {time.time() - t0:.0f}s, mean return (last 100 eps) {tail:.2f}")
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
        action_types[name] = rc.execution.action_type
        prov = load_json(d / "provenance.json")
        meta.append(
            {
                "strategy": name,
                "run_dir": str(d),
                "git_sha": prov.get("git_sha"),
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


def cmd_evaluate(args) -> None:
    from rl_execution import viz
    from rl_execution.backtest import pool_results
    from rl_execution.experiments import (
        build_baselines,
        evaluate_across_regimes,
        list_regimes,
        regime_results_frame,
    )

    runs_root = Path(args.runs_dir) / args.experiment
    strategies, action_types, runs_meta, exec_config, trained_on = _load_runs(runs_root)
    if not strategies:
        print(f"No runs under {runs_root}; evaluating the baselines only.")
    exec_config = replace(exec_config or ExecutionConfig(), action_type=ActionType.CONTINUOUS)

    if args.regimes:
        regimes = list(args.regimes)
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
    regime_results_frame(results).to_csv(out / "summary.csv", index=False)
    save_json(
        {
            "experiment": args.experiment,
            "runs_dir": str(runs_root),
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
    print(f"Wrote {out / 'episodes.csv'}, summary.csv, meta.json and figures/")
    print(f"Next: rlx report --experiment {args.experiment}")


# --------------------------------------------------------------------------- report
def _pvalue(diffs: np.ndarray) -> float:
    """Two-sided one-sample t-test p-value that ``diffs`` has mean zero."""
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return float("nan")
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
) -> pd.DataFrame:
    """One row per (regime, agent): objective, IS level / spread / tail, and the paired gap.

    ``vs_<benchmark>`` is the mean of ``metric - metric_benchmark`` formed episode by
    episode -- every strategy faced the same test seeds (common random numbers), so the
    shared price risk cancels.  For an agent trained on several seeds the training seed is
    the unit of analysis: each seed contributes its mean paired gap, and the CI and p-value
    are taken across seeds.  A single-seed agent or a baseline falls back to the paired
    episode-level bootstrap (``block=2`` for antithetic pairs).  Columns missing from
    ``episodes`` are reported as NaN.
    """
    from rl_execution.metrics.stats import adjust_pvalues, bootstrap_ci, paired_bootstrap_ci

    nan = float("nan")

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
            per_seed_gap = []
            last_m: np.ndarray = np.empty(0)
            for _, run in grp.groupby("strategy", sort=False):
                m = column(run, metric)
                n = min(m.size, bench_m.size)
                per_seed_gap.append(float(np.mean(m[:n] - bench_m[:n])))
                last_m = m[:n]
            is_bps = column(grp, "implementation_shortfall_bps")
            trained = bool(grp["seed"].notna().any())
            row = {
                "regime": regime,
                "strategy": agent,
                "n_seeds": len(per_seed_gap) if trained else "—",
                "J_bps": float(np.mean(column(grp, "objective_bps"))),
                "IS_bps": float(is_bps.mean()),
                "IS_std": float(is_bps.std(ddof=1)) if is_bps.size > 1 else nan,
                "IS_p95": float(np.quantile(is_bps, 0.95)),
                "eff_steps": float(np.mean(column(grp, "effective_steps"))),
                f"vs_{benchmark}": float(np.mean(per_seed_gap)),
                "ci_low": nan,
                "ci_high": nan,
                "p_value": nan,
            }
            if agent != benchmark:
                if len(per_seed_gap) >= 2:
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


def _markdown_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "—"
            return f"{v:.3g}" if abs(v) < 1e-2 and v != 0 else f"{v:.1f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def cmd_report(args) -> None:
    from rl_execution import viz

    res_dir = Path(args.results) if args.results else RESULTS_DIR / args.experiment
    episodes = pd.read_csv(res_dir / "episodes.csv")
    meta = load_json(res_dir / "meta.json")
    block = 2 if meta.get("antithetic") else 1
    table = comparison_table(episodes, benchmark=args.benchmark, metric=args.metric, block=block)
    table.to_csv(res_dir / "comparison.csv", index=False)

    # training curves, one smoothed line per (agent, seed)
    curves = {}
    for log in sorted(Path(meta["runs_dir"]).glob("*/seed_*/train_log.csv")):
        curves[f"{log.parent.parent.name}/{log.parent.name}"] = pd.read_csv(log)["return"]
    if curves:
        viz.plot_training_curve(curves, save=str(res_dir / "figures" / "training_curves.png"))

    shas = sorted({r["git_sha"] for r in meta.get("runs", []) if r.get("git_sha")})
    parts = [
        f"# Results: `{meta['experiment']}`",
        "",
        f"- Test episodes per strategy: {meta['episodes']} (seeds from {meta['base_seed']:,}"
        f"{', antithetic pairs' if block == 2 else ''})",
        f"- Trained runs: {len(meta.get('runs', []))}; code version(s): "
        + (", ".join(s[:10] for s in shas) or "n/a"),
        "- `J_bps` is the objective the agents are rewarded on: implementation shortfall plus "
        "the Almgren-Chriss penalty on carried variance (lower is better). `eff_steps` is how "
        "many steps the order is effectively spread over (20 = TWAP, 1 = one block trade).",
        f"- `vs_{args.benchmark}` is the mean paired difference in `{args.metric}` (negative = "
        f"better than {args.benchmark}). For multi-seed agents the 95% CI and p-value are "
        "across training seeds; `p_holm` is Holm-adjusted over every row of every table.",
        "",
    ]
    cols = ["strategy", "n_seeds", "J_bps", "IS_bps", "IS_std", "IS_p95", "eff_steps"]
    cols += [f"vs_{args.benchmark}", "ci_low", "ci_high", "p_holm"]
    fitted = meta.get("params", {})
    for regime, sub in table.groupby("regime", sort=False):
        parts += [f"## {regime}", "", _markdown_table(sub[cols].sort_values("J_bps")), ""]
        beta = fitted.get(regime, {}).get("SignalRule", {}).get("beta")
        if beta is not None:
            parts += [f"SignalRule coefficient fitted on validation seeds: beta = {beta}.", ""]
        for fig in ("schedule", "risk_vs_cost", "is_distribution"):
            if (res_dir / "figures" / f"{regime}_{fig}.png").exists():
                parts.append(f"![{regime} {fig}](figures/{regime}_{fig}.png)")
        parts.append("")
    if curves:
        parts += ["## Training curves", "", "![training curves](figures/training_curves.png)", ""]
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
