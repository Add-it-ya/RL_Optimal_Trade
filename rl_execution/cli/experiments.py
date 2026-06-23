"""End-to-end experiment pipeline: train agents, evaluate across regimes, make figures.

Trains the RL agents (domain-randomised so one model is robust across regimes), evaluates
them together with all baselines across every market regime using paired seeds, then writes
a results table, a results pickle (for the dashboard), training-reward logs and the full set
of figures into ``results/`` and ``reports/figures/``.

Examples
--------
    rlx-experiments --quick                 # fast smoke pipeline
    rlx-experiments --timesteps 120000 --episodes 200
    rlx-experiments --no-train              # reuse saved models
"""

from __future__ import annotations

import argparse
import time

from rl_execution import viz
from rl_execution.agents import AgentStrategy, required_action_type
from rl_execution.config import ExecutionConfig, Side
from rl_execution.experiments import (
    build_baselines,
    evaluate_across_regimes,
    list_regimes,
    regime_results_frame,
)
from rl_execution.training import load_agent, make_env, save_agent, train_agent
from rl_execution.utils.io import (
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dir,
    save_json,
    save_pickle,
)
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
    return p.parse_args()


def main():
    configure_cli_warnings()
    args = parse_args()
    if args.quick:
        args.timesteps = min(args.timesteps, 8_000)
        args.episodes = min(args.episodes, 40)

    regimes = args.regimes or list_regimes()
    exec_config = ExecutionConfig(
        total_inventory=args.inventory, horizon=args.horizon, side=Side(args.side)
    )
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)

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
            agent = load_agent(name, env)
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
            save_agent(
                agent,
                name,
                tag=name,
                meta={"episode_rewards": reward_logs[disp], "exec_config": exec_config.to_dict()},
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
    df.to_csv(RESULTS_DIR / "regime_results.csv", index=False)
    save_pickle(results, RESULTS_DIR / "results.pkl")
    if reward_logs:
        save_json(reward_logs, RESULTS_DIR / "reward_logs.json")
    print("\nSaved results table -> results/regime_results.csv")

    # ---- figures ----------------------------------------------------------------
    rep = "normal_liquidity" if "normal_liquidity" in regimes else regimes[0]
    rep_res = results[rep]
    rl_names = list(agent_strategies.keys())
    from rl_execution.backtest import results_table

    rep_table = results_table(rep_res)
    viz.plot_inventory_decay(rep_res, save=str(FIGURES_DIR / "inventory_decay.png"))
    viz.plot_execution_schedule(rep_res, save=str(FIGURES_DIR / "execution_schedule.png"))
    viz.plot_reward_curve(rep_res, save=str(FIGURES_DIR / "reward_curve.png"))
    viz.plot_cost_comparison(rep_table, save=str(FIGURES_DIR / "cost_comparison.png"))
    viz.plot_rl_vs_baselines(rep_table, rl_names, save=str(FIGURES_DIR / "rl_vs_baselines.png"))
    viz.plot_is_distribution(rep_res, save=str(FIGURES_DIR / "is_distribution.png"))
    if rl_names and rl_names[0] in rep_res:
        viz.plot_price_path(rep_res[rl_names[0]], save=str(FIGURES_DIR / "sample_path.png"))
    viz.plot_regime_heatmap(df, value="IS_bps", save=str(FIGURES_DIR / "regime_heatmap.png"))
    if reward_logs:
        viz.plot_training_curve(reward_logs, save=str(FIGURES_DIR / "training_curves.png"))
    print("Saved figures -> reports/figures/")

    # ---- paired comparison table (saved for the report) ------------------------
    from rl_execution.backtest import paired_is_table

    paired = paired_is_table(rep_res, benchmark="TWAP")
    paired.to_csv(RESULTS_DIR / "paired_vs_twap.csv")

    # ---- headline summary -------------------------------------------------------
    _print_headline(df, rl_names, rep, results=results)
    save_json(_summary_dict(df, rl_names), RESULTS_DIR / "summary.json")
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
