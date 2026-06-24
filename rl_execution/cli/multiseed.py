"""Across-seed robustness study: train each agent over many seeds and report across-seed CIs.

Writes ``results/multiseed_across_seed.csv`` (headline: mean ``vs_TWAP`` with a bootstrap CI over
*training seeds*) and ``results/multiseed_within_run.csv`` (episode-level CI for a single seed);
both feed §4.2d of ``rlx-report``. Trained ``{config, seed}`` pairs are cached in the model
registry, so re-running (or resuming) the study does not retrain.

Examples
--------
    rlx-multiseed --agents doubledqn --seeds 0 1 2 3 4 --quick
    rlx-multiseed --agents doubledqn dqn --seeds 0 1 2 3 4 5 6 7 8 9 --timesteps 80000 --episodes 200
"""

from __future__ import annotations

import argparse

import pandas as pd

from rl_execution.config import ExecutionConfig, RunConfig, Side
from rl_execution.experiments import list_regimes, run_multiseed
from rl_execution.tracking import get_tracker
from rl_execution.utils.io import RESULTS_DIR, ensure_dir
from rl_execution.utils.provenance import git_sha
from rl_execution.utils.seeding import set_global_seeds
from rl_execution.utils.warnings import configure_cli_warnings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", nargs="+", default=["doubledqn"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--regimes", nargs="+", default=None, help="default: all regimes")
    p.add_argument("--benchmark", default="TWAP")
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--side", default="sell", choices=["buy", "sell"])
    p.add_argument("--inventory", type=float, default=10_000.0)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--n-jobs", type=int, default=1, help="parallel seeds (joblib); 1 = serial")
    p.add_argument("--no-register", action="store_true", help="skip registry caching of seeds")
    p.add_argument("--quick", action="store_true", help="tiny budget / few seeds for a smoke run")
    p.add_argument("--tracker", default="auto", choices=["auto", "mlflow", "wandb", "null"])
    p.add_argument("--experiment", default="rl-execution")
    return p.parse_args()


def main():  # pragma: no cover - trains agents end-to-end; exercised only in full runs
    configure_cli_warnings()
    args = parse_args()
    if args.quick:
        args.timesteps = min(args.timesteps, 8_000)
        args.episodes = min(args.episodes, 40)
        args.seeds = args.seeds[:3]

    regimes = args.regimes or list_regimes()
    exec_config = ExecutionConfig(
        total_inventory=args.inventory, horizon=args.horizon, side=Side(args.side)
    )
    ensure_dir(RESULTS_DIR)

    tracker = get_tracker(args.tracker, experiment=args.experiment)
    tracker.start_run(
        name="multiseed",
        config={
            "agents": args.agents,
            "seeds": args.seeds,
            "regimes": regimes,
            "timesteps": args.timesteps,
            "episodes": args.episodes,
        },
        tags={"git_sha": git_sha(), "seeds": ",".join(map(str, args.seeds))},
    )

    across_frames, within_frames = [], []
    for name in args.agents:
        set_global_seeds(args.seeds[0])
        rc = RunConfig(
            agent=name,
            timesteps=args.timesteps,
            randomized=True,
            eval_episodes=args.episodes,
            device=args.device,
            execution=exec_config,
            tracker=args.tracker,
            experiment=args.experiment,
        )
        print(f"Multi-seed {name}: seeds={args.seeds}, {len(regimes)} regimes ...")
        result = run_multiseed(
            name,
            rc,
            seeds=args.seeds,
            regimes=regimes,
            benchmark=args.benchmark,
            episodes=args.episodes,
            register=not args.no_register,
            n_jobs=args.n_jobs,
        )
        across_frames.append(result.across_seed_frame())
        within_frames.append(result.within_run_frame())

    across = pd.concat(across_frames, ignore_index=True)
    within = pd.concat(within_frames, ignore_index=True)
    across.to_csv(RESULTS_DIR / "multiseed_across_seed.csv", index=False)
    within.to_csv(RESULTS_DIR / "multiseed_within_run.csv", index=False)

    print(f"\nAcross-seed (headline) vs {args.benchmark}:")
    print(across.round(2).to_string(index=False))
    for art in (
        RESULTS_DIR / "multiseed_across_seed.csv",
        RESULTS_DIR / "multiseed_within_run.csv",
    ):
        tracker.log_artifact(str(art))
    tracker.finish("ok")
    print("\nSaved -> results/multiseed_across_seed.csv. Next: rlx-report")


if __name__ == "__main__":
    main()
