#!/usr/bin/env python
"""Lab bring-up check: is this machine ready for Phase 4?

    python scripts/lab/check_gpu.py

1. Prints Python, torch, CUDA and GPU details.
2. Runs the batched environment on the GPU on markets copied from the numpy reference
   simulator, and checks it against the reference: float64 must agree to ~1e-9, float32
   to float32 resolution (the same tolerances as ``tests/test_vec_env.py``, which checks the
   CPU path).
3. Measures batched-env throughput on the GPU (target >= 1M steps/s).

Exits with status 1 if anything fails, so a setup script can stop on it.
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np


def main() -> int:
    import torch

    ok = True
    print(f"python   {sys.version.split()[0]} on {platform.platform()}")
    print(f"torch    {torch.__version__} (built for CUDA {torch.version.cuda})")
    cuda = torch.cuda.is_available()
    print(f"CUDA     available: {cuda}")
    if not cuda:
        print(
            "\nNo usable GPU. Check `nvidia-smi`, then reinstall torch with the CUDA build that "
            "matches the driver (see scripts/lab/README.md). The CPU path still works."
        )
        return 1
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}    {props.name}, {props.total_memory / 2**30:.1f} GiB")

    import rl_execution.experiments  # noqa: F401  (import order)
    from rl_execution.config import ExecutionConfig
    from rl_execution.envs import ExecutionEnv
    from rl_execution.experiments import get_regime
    from rl_execution.vec import BatchedExecutionEnv, reference_paths

    dev = torch.device("cuda")
    ec = ExecutionConfig()
    print("\nparity on the GPU against the numpy reference (64 episodes per regime):")
    for regime in ("abm_ac_optimal", "sideways", "gbm_medium", "clustered_vol", "thin"):
        mc = get_regime(regime)
        rng = np.random.default_rng(0)
        actions = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(ec.horizon, 64)))
        results = {}
        for name, device, dtype in [
            ("cpu64", "cpu", torch.float64),
            ("gpu64", dev, torch.float64),
            ("gpu32", dev, torch.float32),
        ]:
            configs, paths = reference_paths(
                lambda: ExecutionEnv(mc, ec), 64, 9_000, antithetic=True, device=device, dtype=dtype
            )
            env = BatchedExecutionEnv(64, configs, ec, device=device, dtype=dtype)
            env.reset(paths)
            rewards = []
            for t in range(ec.horizon):
                rewards.append(env.step(actions[t].to(device))[1].double().cpu())
            results[name] = (
                torch.stack(rewards),
                env.episode_summary()["objective_bps"].double().cpu(),
            )
        ref_r, ref_j = results["cpu64"]
        gap64 = max(
            (results["gpu64"][0] - ref_r).abs().max(), (results["gpu64"][1] - ref_j).abs().max()
        )
        gap32_r = (results["gpu32"][0] - ref_r).abs().max()
        gap32_j = (results["gpu32"][1] - ref_j).abs().max()
        good = gap64 < 1e-9 and gap32_r < 1e-4 and gap32_j < 1e-2
        ok &= bool(good)
        print(
            f"  {'ok  ' if good else 'FAIL'} {regime:<15} float64 {float(gap64):.1e}  "
            f"float32 reward {float(gap32_r):.1e}, objective {float(gap32_j):.1e} bps"
        )  # fmt: skip

    print("\nbatched env throughput on the GPU:")
    best = 0.0
    for n in (4_096, 65_536, 262_144):
        env = BatchedExecutionEnv(n, get_regime("abm_ac_optimal"), ec, device=dev, seed=0)
        action = torch.zeros(n, device=dev)
        for repeat in range(2):  # the first pass warms up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            env.reset()
            for _ in range(ec.horizon):
                env.step(action)
            torch.cuda.synchronize()
        rate = n * ec.horizon / (time.perf_counter() - t0)
        best = max(best, rate)
        print(f"  N={n:<8} {rate:>14,.0f} steps/s")
    target = best >= 1e6
    print(f"  target >= 1M steps/s: {'met' if target else 'NOT met'}")

    print("\nREADY" if ok else "\nNOT READY: a parity check failed (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
