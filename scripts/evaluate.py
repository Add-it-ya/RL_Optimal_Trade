#!/usr/bin/env python
"""Evaluate baselines (and any saved RL agents) on a single market regime.

Examples
--------
    python scripts/evaluate.py --regime bear --episodes 200
    python scripts/evaluate.py --regime high_vol --agents ppo sac
    python scripts/evaluate.py --baselines-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")

from rl_execution.agents import AgentStrategy, required_action_type
from rl_execution.backtest import compare_strategies, results_table
from rl_execution.config import ActionType, ExecutionConfig, Side
from rl_execution.experiments import build_baselines, get_regime
from rl_execution.envs import ExecutionEnv
from rl_execution.training import load_agent, make_env
from rl_execution.utils.io import FIGURES_DIR, MODELS_DIR, ensure_dir
from rl_execution import viz

DISPLAY = {"dqn": "DQN", "doubledqn": "DoubleDQN", "ppo": "PPO", "a2c": "A2C", "sac": "SAC"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regime", default="normal_liquidity")
    p.add_argument("--agents", nargs="*", default=["dqn", "doubledqn", "ppo", "a2c", "sac"])
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--side", default="sell", choices=["buy", "sell"])
    p.add_argument("--inventory", type=float, default=10_000.0)
    p.add_argument("--horizon", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    exec_config = ExecutionConfig(
        total_inventory=args.inventory, horizon=args.horizon, side=Side(args.side))
    market_config = get_regime(args.regime)

    strategies = dict(build_baselines())
    action_types = {}
    rl_names = []
    if not args.baselines_only:
        for name in args.agents:
            tag_json = MODELS_DIR / f"{name}.json"
            if not tag_json.exists():
                print(f"  (skip {name}: no saved model)")
                continue
            disp = DISPLAY.get(name, name.upper())
            env = make_env(name, exec_config, regime=args.regime)
            strategies[disp] = AgentStrategy(load_agent(name, env), disp)
            action_types[disp] = required_action_type(name)
            rl_names.append(disp)

    def factory_for(at):
        from dataclasses import replace
        ec = replace(exec_config, action_type=at)
        return lambda: ExecutionEnv(market_config, ec)

    # evaluate each strategy with its compatible action space, paired seeds
    from rl_execution.backtest import evaluate
    results = {}
    for nm, strat in strategies.items():
        at = action_types.get(nm, ActionType.CONTINUOUS)
        results[nm] = evaluate(factory_for(at), strat, n_episodes=args.episodes,
                               base_seed=10_000, name=nm, progress=False)

    table = results_table(results)
    print(f"\n=== {args.regime} ({args.side}, {args.episodes} episodes) ===")
    print(table.round(3).to_string())

    ensure_dir(FIGURES_DIR)
    tag = args.regime
    viz.plot_inventory_decay(results, save=str(FIGURES_DIR / f"{tag}_inventory.png"))
    viz.plot_cost_comparison(table, save=str(FIGURES_DIR / f"{tag}_cost.png"))
    if rl_names:
        viz.plot_rl_vs_baselines(table, rl_names, save=str(FIGURES_DIR / f"{tag}_rl_vs_base.png"))
    viz.plot_is_distribution(results, save=str(FIGURES_DIR / f"{tag}_is_dist.png"))
    print(f"\nSaved figures -> reports/figures/{tag}_*.png")


if __name__ == "__main__":
    main()
