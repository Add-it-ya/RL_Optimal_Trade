#!/usr/bin/env python
"""Train a single RL agent for optimal execution.

Examples
--------
    python scripts/train.py --agent ppo --timesteps 100000 --randomized
    python scripts/train.py --agent doubledqn --timesteps 80000 --regime bear
    python scripts/train.py --agent sac --timesteps 100000 --side buy --tag sac_buy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")

from rl_execution.agents import AgentStrategy
from rl_execution.backtest import evaluate, results_table
from rl_execution.config import ExecutionConfig, Side
from rl_execution.training import make_env_factory, save_agent, train_agent
from rl_execution.utils.io import save_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", required=True,
                   choices=["dqn", "doubledqn", "ppo", "a2c", "sac", "sb3dqn"])
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--regime", default=None, help="fixed regime name (else randomized)")
    p.add_argument("--randomized", action="store_true",
                   help="train on the domain-randomized regime distribution")
    p.add_argument("--side", default="sell", choices=["buy", "sell"])
    p.add_argument("--inventory", type=float, default=10_000.0)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="cpu is recommended for these small MLP policies")
    p.add_argument("--progress", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    randomized = args.randomized or (args.regime is None)
    exec_config = ExecutionConfig(
        total_inventory=args.inventory, horizon=args.horizon, side=Side(args.side),
    )

    print(f"Training {args.agent} for {args.timesteps:,} steps "
          f"({'randomized' if randomized else args.regime}) ...")
    agent, reward_log = train_agent(
        args.agent, total_timesteps=args.timesteps, exec_config=exec_config,
        regime=args.regime, randomized=randomized, seed=args.seed,
        progress=args.progress, device=args.device,
    )

    tag = args.tag or args.agent
    meta = {
        "timesteps": args.timesteps,
        "regime": args.regime,
        "randomized": randomized,
        "side": args.side,
        "exec_config": exec_config.to_dict(),
        "episode_rewards": reward_log,
    }
    save_agent(agent, args.agent, tag=tag, meta=meta)
    print(f"Saved model + sidecar under models/{tag}.*")

    # quick evaluation on the (fixed or randomized) regime
    factory = make_env_factory(args.agent, exec_config, regime=args.regime,
                               randomized=randomized)
    res = evaluate(factory, AgentStrategy(agent, args.agent),
                   n_episodes=args.eval_episodes, base_seed=10_000, progress=False)
    row = res.summary_row()
    print("\nQuick evaluation:")
    for k, v in row.items():
        print(f"  {k:>14}: {v:.3f}" if isinstance(v, float) else f"  {k:>14}: {v}")


if __name__ == "__main__":
    main()
