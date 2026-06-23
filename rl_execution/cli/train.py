"""Train a single RL agent for optimal execution -- tracked, seeded and versioned.

Every run sets global seeds, is logged to an experiment tracker (MLflow-local by default,
W&B opt-in, Null for CI), stamps provenance into the model sidecar, and registers a versioned
model in the registry.  A run is fully described by a :class:`RunConfig`, so a YAML config plus
a seed reproduces it.

Examples
--------
    rlx-train --agent ppo --timesteps 100000 --randomized
    rlx-train --agent doubledqn --timesteps 80000 --regime bear
    rlx-train --config run.yaml --seed 3            # reproduce from a serialised config
    rlx-train --agent sac --tracker null --no-register   # quick local run, nothing logged
"""

from __future__ import annotations

import argparse

from rl_execution.agents import AgentStrategy
from rl_execution.backtest import evaluate
from rl_execution.config import ExecutionConfig, RunConfig, Side
from rl_execution.tracking import get_tracker
from rl_execution.training import (
    make_env_factory,
    model_artifact_path,
    save_agent,
    train_agent,
)
from rl_execution.utils.config_io import dump_run_config, load_run_config
from rl_execution.utils.provenance import config_hash, git_sha
from rl_execution.utils.seeding import set_global_seeds
from rl_execution.utils.warnings import configure_cli_warnings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="YAML RunConfig; CLI flags below override it")
    # None defaults so we can tell "explicitly set" from "left at default" when merging a config.
    p.add_argument(
        "--agent", default=None, choices=["dqn", "doubledqn", "ppo", "a2c", "sac", "sb3dqn"]
    )
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--regime", default=None, help="fixed regime name (else randomized)")
    p.add_argument(
        "--randomized", action="store_true", help="train on the domain-randomized distribution"
    )
    p.add_argument("--side", default="sell", choices=["buy", "sell"])
    p.add_argument("--inventory", type=float, default=10_000.0)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "auto"], help="cpu recommended"
    )
    p.add_argument(
        "--tracker",
        default=None,
        choices=["auto", "mlflow", "wandb", "null"],
        help="experiment tracker backend (default: auto -> MLflow-local unless WANDB_API_KEY)",
    )
    p.add_argument("--experiment", default=None, help="experiment / W&B project name")
    p.add_argument("--no-register", action="store_true", help="skip model-registry versioning")
    p.add_argument("--save-config", default=None, help="write the resolved RunConfig to this YAML")
    p.add_argument("--progress", action="store_true")
    return p.parse_args()


def build_run_config(args) -> RunConfig:
    """Merge a YAML config (if any) with CLI flags into a validated :class:`RunConfig`."""
    if args.config:
        rc = load_run_config(args.config, validate=False)
    else:
        if not args.agent:
            raise SystemExit("error: --agent is required unless --config is given")
        rc = RunConfig(
            agent=args.agent,
            timesteps=args.timesteps if args.timesteps is not None else 80_000,
            regime=args.regime,
            randomized=args.randomized or (args.regime is None),
            device=args.device,
            eval_episodes=args.eval_episodes,
            tag=args.tag,
            execution=ExecutionConfig(
                total_inventory=args.inventory, horizon=args.horizon, side=Side(args.side)
            ),
        )

    # Flags that override even when a --config is supplied (only if explicitly provided).
    if args.agent is not None:
        rc.agent = args.agent
    if args.timesteps is not None:
        rc.timesteps = args.timesteps
    if args.seed is not None:
        rc.seed = args.seed
    if args.tracker is not None:
        rc.tracker = args.tracker
    if args.experiment is not None:
        rc.experiment = args.experiment
    return rc.validate()


def main():
    configure_cli_warnings()
    args = parse_args()
    rc = build_run_config(args)
    set_global_seeds(rc.seed)

    print(
        f"Training {rc.agent} for {rc.timesteps:,} steps "
        f"({'randomized' if rc.randomized else rc.regime}), seed={rc.seed} "
        f"[tracker={rc.tracker}] ..."
    )

    tracker = get_tracker(rc.tracker, experiment=rc.experiment)
    register = not args.no_register
    with tracker:
        tracker.start_run(
            name=rc.run_tag,
            config=rc.to_dict(),
            tags={"agent": rc.agent, "seed": rc.seed, "git_sha": git_sha()},
        )

        agent, reward_log = train_agent(
            rc.agent,
            total_timesteps=rc.timesteps,
            exec_config=rc.execution,
            regime=rc.regime,
            randomized=rc.randomized,
            seed=rc.seed,
            progress=args.progress,
            device=rc.device,
            agent_kwargs=rc.agent_kwargs or None,
        )
        for i, r in enumerate(reward_log):
            tracker.log_metrics({"train/episode_reward": r}, step=i)

        meta = {
            "timesteps": rc.timesteps,
            "regime": rc.regime,
            "randomized": rc.randomized,
            "side": rc.execution.side.value,
            "exec_config": rc.execution.to_dict(),
            "episode_rewards": reward_log,
        }
        save_agent(agent, rc.agent, tag=rc.run_tag, meta=meta, seed=rc.seed, config=rc)
        print(f"Saved model + sidecar under models/{rc.run_tag}.*")

        # quick evaluation on the (fixed or randomized) regime
        factory = make_env_factory(
            rc.agent, rc.execution, regime=rc.regime, randomized=rc.randomized
        )
        res = evaluate(
            factory,
            AgentStrategy(agent, rc.agent),
            n_episodes=rc.eval_episodes,
            base_seed=rc.eval_base_seed,
            progress=False,
        )
        row = res.summary_row()
        tracker.log_metrics({f"eval/{k}": v for k, v in row.items() if isinstance(v, (int, float))})

        # register a versioned, lineage-tagged model (no-op for the Null tracker)
        if register:
            model_file = str(model_artifact_path(rc.agent, rc.run_tag))
            version = tracker.log_model(
                model_file,
                name=rc.run_tag,
                metadata={
                    "seed": rc.seed,
                    "agent": rc.agent,
                    "git_sha": git_sha(),
                    "config_hash": config_hash(rc),
                    "timesteps": rc.timesteps,
                    "eval": {k: v for k, v in row.items() if isinstance(v, (int, float))},
                },
            )
            if version:
                print(f"Registered model '{rc.run_tag}' version {version}")

        if args.save_config:
            dump_run_config(rc, args.save_config)
            print(f"Wrote resolved config -> {args.save_config}")

    print("\nQuick evaluation:")
    for k, v in row.items():
        print(f"  {k:>14}: {v:.3f}" if isinstance(v, float) else f"  {k:>14}: {v}")


if __name__ == "__main__":
    main()
