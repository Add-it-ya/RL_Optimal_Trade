#!/usr/bin/env python
"""Environment and training throughput: numpy + SB3 vs the batched env on CPU and GPU.

    python scripts/bench_throughput.py                  # everything this machine can run
    python scripts/bench_throughput.py --skip-sb3       # batched env only
    python scripts/bench_throughput.py --out results/bench

Rows (steps are environment steps, i.e. decisions):

* ``numpy env``        -- the reference ``ExecutionEnv``, one episode at a time;
* ``SB3 PPO``          -- PPO training on ``DummyVecEnv`` x 16 (the E1/E2/E3 setup);
* ``batched env``      -- ``BatchedExecutionEnv`` stepping N episodes with fixed actions;
* ``VecPPO``           -- PPO training on the batched env (rollout + update).

The Phase 4 target is >= 1M environment steps per second on one GPU for the batched env.
Writes ``throughput.csv`` (and prints a table) when ``--out`` is given.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import rl_execution.experiments  # noqa: F401  (import order: avoid the training import cycle)
from rl_execution.config import ExecutionConfig
from rl_execution.envs import ExecutionEnv
from rl_execution.experiments import get_regime
from rl_execution.vec import BatchedExecutionEnv
from rl_execution.vec.ppo import VecPPOConfig, train_vec_ppo


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_numpy_env(market, ec, episodes: int = 300) -> float:
    env = ExecutionEnv(market, ec)
    steps, t0 = 0, time.perf_counter()
    for i in range(episodes):
        env.reset(seed=i)
        done = False
        while not done:
            _, _, done, _, _ = env.step(np.zeros(1, dtype=np.float32))
            steps += 1
    return steps / (time.perf_counter() - t0)


def bench_sb3(market, ec, timesteps: int = 100_000) -> float:
    """One torch thread, as the real runs use (one process per seed); many threads only
    contend over SB3's tiny matrix products.  A short warm-up run keeps the (slow) SB3 and
    TensorBoard imports and model construction out of the timing."""
    from rl_execution.training import train_agent

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kwargs = dict(market_config=market, randomized=False, seed=0, n_envs=16)
        train_agent("ppo", 4096, ec, **kwargs)  # warm-up
        t0 = time.perf_counter()
        train_agent("ppo", timesteps, ec, **kwargs)
        return timesteps / (time.perf_counter() - t0)
    finally:
        torch.set_num_threads(threads)


def bench_batched_env(market, ec, n: int, device: torch.device, repeats: int = 5) -> float:
    env = BatchedExecutionEnv(n, market, ec, device=device, seed=0)
    action = torch.zeros(n, device=device)
    env.reset()  # warm-up (allocations, kernel compilation)
    for _ in range(ec.horizon):
        env.step(action)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        env.reset()
        for _ in range(ec.horizon):
            env.step(action)
    _sync(device)
    return repeats * n * ec.horizon / (time.perf_counter() - t0)


def bench_vecppo(market, ec, n: int, device: torch.device, timesteps: int) -> float:
    env = BatchedExecutionEnv(n, market, ec, device=device, seed=0)
    cfg = VecPPOConfig(n_envs=n, minibatch_size=max(1024, n * ec.horizon // 8))
    train_vec_ppo(env, n * ec.horizon, cfg)  # warm-up rollout
    _sync(device)
    t0 = time.perf_counter()
    out = train_vec_ppo(env, timesteps, cfg)
    _sync(device)
    real = out["episodes"][-1]["step"]
    return real / (time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--regime", default="abm_ac_optimal")
    parser.add_argument("--skip-sb3", action="store_true", help="skip the slow SB3 row")
    parser.add_argument("--out", default=None, help="folder for throughput.csv")
    args = parser.parse_args()

    market, ec = get_regime(args.regime), ExecutionConfig()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    rows = []

    def add(name, device, n, rate):
        rows.append({"what": name, "device": str(device), "n_envs": n, "steps_per_s": rate})
        print(f"{name:<12} {str(device):<6} N={n:<7} {rate:>14,.0f} steps/s", flush=True)

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    add("numpy env", "cpu", 1, bench_numpy_env(market, ec))
    if not args.skip_sb3:
        add("SB3 PPO", "cpu", 16, bench_sb3(market, ec))
    for device in devices:
        sizes = [256, 4096, 16384] + ([65536, 262144] if device.type == "cuda" else [])
        for n in sizes:
            add("batched env", device, n, bench_batched_env(market, ec, n, device))
        for n in [256, 4096] + ([16384] if device.type == "cuda" else []):
            budget = max(200_000, 4 * n * ec.horizon)
            add("VecPPO", device, n, bench_vecppo(market, ec, n, device, budget))

    table = pd.DataFrame(rows)
    best_gpu = table[(table.device == "cuda") & (table.what == "batched env")].steps_per_s
    if len(best_gpu):
        verdict = "met" if best_gpu.max() >= 1e6 else "NOT met"
        print(f"\nPhase 4 target (>= 1M env steps/s on one GPU): {verdict} ({best_gpu.max():,.0f})")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        table.assign(host=platform.node(), torch=torch.__version__).to_csv(
            out / "throughput.csv", index=False
        )
        print(f"Wrote {out / 'throughput.csv'}")


if __name__ == "__main__":
    main()
