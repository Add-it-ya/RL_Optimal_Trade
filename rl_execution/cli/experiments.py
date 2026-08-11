"""End-to-end experiment pipeline: train agents, evaluate across regimes, make figures.

Trains the RL agents (domain-randomised so one model is robust across regimes), evaluates
them together with all baselines across every market regime using paired seeds, then writes
a results table, a results pickle (for the dashboard), training-reward logs and the full set
of figures into ``results/`` and ``reports/figures/``.

Examples
--------
    rlx-experiments --quick                 # fast smoke pipeline (sandboxed under results/quick)
    rlx-experiments --timesteps 120000 --episodes 200
    rlx-experiments --no-train              # reuse saved models
    rlx-experiments --out-dir results/exp1 --models-dir models/exp1

``--quick`` writes to ``results/quick`` and ``models/quick`` unless ``--out-dir`` /
``--models-dir`` say otherwise, so a smoke run can never overwrite published artefacts.
"""

from __future__ import annotations

import argparse
import time

from rl_execution import viz
from rl_execution.agents import AgentStrategy, required_action_type
from rl_execution.config import ExecutionConfig, RunConfig, Side
from rl_execution.experiments import (
    build_baselines,
    evaluate_across_regimes,
    list_regimes,
    regime_results_frame,
)
from rl_execution.tracking import get_tracker
from rl_execution.training import (
    load_agent,
    make_env,
    model_artifact_path,
    save_agent,
    train_agent,
)
from rl_execution.utils.io import (
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dir,
    resolve_artifact_dirs,
    save_json,
    save_pickle,
)
from rl_execution.utils.provenance import config_hash, git_sha
from rl_execution.utils.seeding import set_global_seeds
from rl_execution.utils.warnings import configure_cli_warnings

DISPLAY = {"dqn": "DQN", "doubledqn": "DoubleDQN", "ppo": "PPO", "a2c": "A2C", "sac": "SAC"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", nargs="+", default=["dqn", "doubledqn", "ppo", "a2c", "sac"])
    p.add_argument("--regimes", nargs="+", default=None, help="default: all regimes")
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--side", default="sell", choices=["buy", "sell"])
    p.add_argument("--inventory", type=float, default=10_000.0)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="cpu is recommended for these small MLP policies",
    )
    p.add_argument("--no-train", action="store_true", help="reuse saved models")
    p.add_argument("--quick", action="store_true", help="tiny budget for a fast demo")
    p.add_argument(
        "--out-dir",
        default=None,
        help="where to write tables/pickles (default: results/, or results/quick/ under --quick)",
    )
    p.add_argument(
        "--models-dir",
        default=None,
        help="where to read/write models (default: models/, or models/quick/ under --quick)",
    )
    p.add_argument(
        "--tracker",
        default="auto",
        choices=["auto", "mlflow", "wandb", "null"],
        help="experiment tracker backend (default: auto -> MLflow-local unless WANDB_API_KEY)",
    )
    p.add_argument("--experiment", default="rl-execution", help="experiment / W&B project name")
    p.add_argument("--no-register", action="store_true", help="skip model-registry versioning")
    return p.parse_args()


def main():
    configure_cli_warnings()
    args = parse_args()
    if args.quick:
        args.timesteps = min(args.timesteps, 8_000)
        args.episodes = min(args.episodes, 40)

    set_global_seeds(args.seed)
    regimes = args.regimes or list_regimes()
    exec_config = ExecutionConfig(
        total_inventory=args.inventory, horizon=args.horizon, side=Side(args.side)
    )
    # A --quick smoke run writes to results/quick and models/quick unless the user says
    # otherwise, so tiny-budget numbers can never overwrite the published artefacts.
    results_dir, models_dir = resolve_artifact_dirs(args.out_dir, args.models_dir, quick=args.quick)
    # Figures follow the results dir: a sandboxed run must not overwrite the published
    # reports/figures/*.png that REPORT.md refers to either.
    figures_dir = FIGURES_DIR if results_dir == RESULTS_DIR else results_dir / "figures"
    ensure_dir(results_dir)
    ensure_dir(models_dir)
    ensure_dir(figures_dir)
    print(f"Writing results -> {results_dir}  models -> {models_dir}  figures -> {figures_dir}")

    # ---- experiment tracking ----------------------------------------------------
    tracker = get_tracker(args.tracker, experiment=args.experiment)
    register = not args.no_register
    tracker.start_run(
        name="experiments",
        config={
            "agents": args.agents,
            "regimes": regimes,
            "timesteps": args.timesteps,
            "episodes": args.episodes,
            "seed": args.seed,
            "device": args.device,
            "out_dir": str(results_dir),
            "models_dir": str(models_dir),
            "execution": exec_config.to_dict(),
        },
        tags={"agents": ",".join(args.agents), "seed": args.seed, "git_sha": git_sha()},
    )

    # ---- train (or load) agents -------------------------------------------------
    agent_strategies = {}
    action_types = {}
    reward_logs = {}
    for name in args.agents:
        disp = DISPLAY.get(name, name.upper())
        action_types[disp] = required_action_type(name)
        if args.no_train:
            print(f"Loading saved agent '{name}' ...")
            env = make_env(name, exec_config, randomized=True, seed=args.seed)
            agent = load_agent(name, env, models_dir=models_dir)
        else:
            t0 = time.time()
            print(f"Training {disp} for {args.timesteps:,} steps ...")
            agent, reward_logs[disp] = train_agent(
                name,
                total_timesteps=args.timesteps,
                exec_config=exec_config,
                randomized=True,
                seed=args.seed,
                device=args.device,
            )
            rc = RunConfig(
                agent=name,
                timesteps=args.timesteps,
                seed=args.seed,
                randomized=True,
                eval_episodes=args.episodes,
                device=args.device,
                execution=exec_config,
                tracker=args.tracker,
                experiment=args.experiment,
            )
            save_agent(
                agent,
                name,
                tag=name,
                meta={"episode_rewards": reward_logs[disp], "exec_config": exec_config.to_dict()},
                seed=args.seed,
                config=rc,
                models_dir=models_dir,
            )
            for i, r in enumerate(reward_logs[disp]):
                tracker.log_metrics({f"train/{name}/episode_reward": r}, step=i)
            if register:
                tracker.log_model(
                    str(model_artifact_path(name, name, models_dir=models_dir)),
                    name=name,
                    metadata={
                        "seed": args.seed,
                        "agent": name,
                        "git_sha": git_sha(),
                        "config_hash": config_hash(rc),
                        "timesteps": args.timesteps,
                    },
                )
            print(f"  done in {time.time() - t0:.1f}s")
        agent_strategies[disp] = AgentStrategy(agent, disp)

    # ---- assemble strategy set --------------------------------------------------
    strategies = {**build_baselines(), **agent_strategies}

    # ---- evaluate across regimes ------------------------------------------------
    print(
        f"\nEvaluating {len(strategies)} strategies across {len(regimes)} regimes "
        f"({args.episodes} episodes each) ..."
    )
    results = evaluate_across_regimes(
        strategies,
        exec_config,
        regimes=regimes,
        action_types=action_types,
        n_episodes=args.episodes,
        base_seed=10_000,
        progress=False,
    )
    df = regime_results_frame(results)
    df.to_csv(results_dir / "regime_results.csv", index=False)
    save_pickle(results, results_dir / "results.pkl")
    if reward_logs:
        save_json(reward_logs, results_dir / "reward_logs.json")
    print(f"\nSaved results table -> {results_dir / 'regime_results.csv'}")

    # ---- figures ----------------------------------------------------------------
    rep = "normal_liquidity" if "normal_liquidity" in regimes else regimes[0]
    rep_res = results[rep]
    rl_names = list(agent_strategies.keys())
    from rl_execution.backtest import results_table

    rep_table = results_table(rep_res)
    viz.plot_inventory_decay(rep_res, save=str(figures_dir / "inventory_decay.png"))
    viz.plot_execution_schedule(rep_res, save=str(figures_dir / "execution_schedule.png"))
    viz.plot_reward_curve(rep_res, save=str(figures_dir / "reward_curve.png"))
    viz.plot_cost_comparison(rep_table, save=str(figures_dir / "cost_comparison.png"))
    viz.plot_rl_vs_baselines(rep_table, rl_names, save=str(figures_dir / "rl_vs_baselines.png"))
    viz.plot_is_distribution(rep_res, save=str(figures_dir / "is_distribution.png"))
    if rl_names and rl_names[0] in rep_res:
        viz.plot_price_path(rep_res[rl_names[0]], save=str(figures_dir / "sample_path.png"))
    viz.plot_regime_heatmap(df, value="IS_bps", save=str(figures_dir / "regime_heatmap.png"))
    if reward_logs:
        viz.plot_training_curve(reward_logs, save=str(figures_dir / "training_curves.png"))
    print(f"Saved figures -> {figures_dir}")

    # ---- paired comparison + multiple-testing-corrected significance (for the report) ----
    from rl_execution.backtest import corrected_significance, paired_is_table

    paired = paired_is_table(rep_res, benchmark="TWAP")
    paired.to_csv(results_dir / "paired_vs_twap.csv")
    # Holm-correct the paired tests across the full regime x strategy grid so the report does
    # not cherry-pick a significant regime out of many. Cheap: just t-tests on existing results.
    corrected = corrected_significance(results, benchmark="TWAP", method="holm")
    corrected.to_csv(results_dir / "corrected_significance.csv", index=False)

    # ---- headline summary -------------------------------------------------------
    _print_headline(df, rl_names, rep, results=results)
    summary = _summary_dict(df, rl_names)
    save_json(summary, results_dir / "summary.json")

    # ---- log results to the tracker --------------------------------------------
    for artifact in (
        results_dir / "regime_results.csv",
        results_dir / "summary.json",
        results_dir / "paired_vs_twap.csv",
        results_dir / "corrected_significance.csv",
    ):
        tracker.log_artifact(str(artifact))
    for fig in sorted(figures_dir.glob("*.png")):
        tracker.log_artifact(str(fig))
    if "win_rates" in summary:
        tracker.log_metrics({f"win_rate/{k}": v for k, v in summary["win_rates"].items()})
    tracker.log_metrics(
        {f"mean_IS_bps/{k}": float(v) for k, v in summary.get("mean_IS_bps", {}).items()}
    )
    tracker.finish("ok")

    print("\nNext: rlx-report")


def _print_headline(df, rl_names, rep, results=None):
    from rl_execution.backtest import paired_is_table

    print("\n" + "=" * 72)
    print(f"HEADLINE (representative regime = {rep})")
    print("=" * 72)
    sub = df[df["regime"] == rep][["strategy", "IS_bps", "ExecCost_bps", "IS_Sharpe"]]
    print(sub.sort_values("IS_bps").to_string(index=False))

    if results is not None and rep in results and "TWAP" in results[rep]:
        print("\nPaired comparison vs TWAP (common random numbers; vs_TWAP<0 = better):")
        print(paired_is_table(results[rep], benchmark="TWAP").round(2).to_string())

    # win-rate of best RL agent vs key baselines averaged across regimes
    baselines = ["TWAP", "VWAP", "Random"]
    pivot = df.pivot(index="regime", columns="strategy", values="IS_bps")
    if rl_names:
        best_rl = pivot[rl_names].mean().idxmin()
        print(f"\nBest RL agent by mean IS across regimes: {best_rl}")
        for b in baselines:
            if b in pivot.columns:
                wins = (pivot[best_rl] < pivot[b]).sum()
                print(
                    f"  {best_rl} beats {b:<7} in {wins}/{len(pivot)} regimes "
                    f"(mean IS {pivot[best_rl].mean():.1f} vs {pivot[b].mean():.1f} bps)"
                )


def _summary_dict(df, rl_names):
    pivot = df.pivot(index="regime", columns="strategy", values="IS_bps")
    out = {"mean_IS_bps": pivot.mean().to_dict()}
    if rl_names:
        best_rl = pivot[rl_names].mean().idxmin()
        out["best_rl"] = best_rl
        out["win_rates"] = {
            b: int((pivot[best_rl] < pivot[b]).sum())
            for b in ["TWAP", "VWAP", "Random", "POV", "AlmgrenChriss"]
            if b in pivot.columns
        }
        out["n_regimes"] = int(len(pivot))
    return out


if __name__ == "__main__":
    main()
