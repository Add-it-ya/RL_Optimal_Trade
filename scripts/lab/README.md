# Running Phase 4 on the lab GPU

Phase 4 moves training onto a GPU. The pieces are in the package already and are tested on
the CPU:

- `rl_execution/vec/torch_env.py`: `BatchedExecutionEnv` steps thousands of episodes at once
  as tensor operations. It is checked against the numpy reference environment step by step
  on identical markets (`tests/test_vec_env.py`).
- `rl_execution/vec/ppo.py`: a compact PPO on that environment (`rlx train --agent vecppo`).
  Its defaults mirror the SB3 PPO used for E1–E3. Checkpoints are selected on the numpy
  simulator's own validation episodes, and `rlx evaluate` scores the trained actor on the
  numpy environment, so every reported number still comes from the reference env.
- `scripts/bench_throughput.py`: numpy + SB3 vs the batched environment on CPU and GPU.

What the lab run has to show is fixed below, before any GPU result exists.

## Before the visit

Check: the GPU (`nvidia-smi` should list it, and its top line gives the driver's CUDA
version), Python 3.10 or newer, internet access for `pip`, a few GB of disk space, and
whether you can leave a job running for about an hour (sleep or wipe-on-logout policies).

## 1. Clone this branch

```bash
git clone -b phase-4-gpu https://github.com/Add-it-ya/RL_Optimal_Trade.git
```
```bash
cd RL_Optimal_Trade
```

## 2. Set up (once)

Linux (the PyPI torch wheels include CUDA):

```bash
bash scripts/lab/setup.sh
```

Windows (PyPI torch is CPU-only, so pass the CUDA build; pick the tag that matches
`nvidia-smi` on https://pytorch.org/get-started/locally/):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\lab\setup.ps1 -CudaTag cu126
```

Setup ends by running `scripts/lab/check_gpu.py`. It prints the GPU, checks the batched
environment **on the GPU** against the numpy reference, and measures throughput. It must end
with `READY`. If it says `CUDA available: False`, the torch build does not match the driver:
rerun setup with another tag (Linux: `bash scripts/lab/setup.sh cu121`, for example).

With conda instead of venv: create an env with Python 3.11, install torch from the
pytorch.org selector, then `pip install -e ".[rl,dev]"` and `python scripts/lab/check_gpu.py`.

## 3. Smoke run (about 2 minutes)

Activate the environment (`source .venv/bin/activate`, or `.\.venv\Scripts\Activate.ps1`),
then do a short run under a throwaway experiment name, so it can't be mistaken for the real
one:

```bash
EXPERIMENT=smoke SEEDS=0 STEPS=200000 EPISODES=200 bash scripts/lab/run_phase4.sh
```
```powershell
powershell -ExecutionPolicy Bypass -File scripts\lab\run_phase4.ps1 -Experiment smoke -Seeds 0 -Steps 200000 -Episodes 200
```

## 4. The real run (under an hour on a recent GPU)

```bash
nohup bash scripts/lab/run_phase4.sh > /dev/null 2>&1 &
```
```powershell
powershell -ExecutionPolicy Bypass -File scripts\lab\run_phase4.ps1
```

It runs, in order:

1. the bring-up check;
2. the CPU parity tests;
3. the throughput benchmark, written to `results/bench/throughput.csv`;
4. VecPPO on the E1 market: 10 seeds × 5M steps on the GPU, evaluated on 5,000 antithetic
   test episodes on the numpy environment, with the report in `results/e1_vec/REPORT.md`;
5. one archive, `lab_results_<host>_<date>.tar.gz` (`.zip` on Windows).

Everything is logged to `results/lab_logs/`. Re-running resumes: finished seeds are skipped.
On Linux, `nohup` keeps it running after you log out; watch it with
`tail -f results/lab_logs/phase4_*.log`.

## 5. Pass criteria (pre-registered)

| Check | Pass |
|---|---|
| GPU parity | float64 on the GPU equals the numpy reference to 1e-9; float32 within 1e-4 per scaled reward and 0.01 bps per episode objective (`check_gpu.py`) |
| CPU parity tests | `tests/test_vec_env.py` passes |
| Throughput | the batched environment reaches ≥ 1M steps/s on one GPU |
| Parity test 3: VecPPO reproduces the SB3 E1 result | Across 10 seeds at 5M steps, the mean gap to Almgren-Chriss is within ±0.30 bps of SB3's 5M result (+0.20 bps, 95% CI 0.07–0.35), i.e. in [−0.10, +0.50]. Median schedule L1 to AC < 0.10. No degenerate seed. |

The SB3 reference is E1b (10 seeds × 5M steps, same test seeds, same evaluation): gap
+0.20 bps (0.07–0.35), median L1 0.065, 8 of 10 seeds under 0.10. If VecPPO misses, report
it as it is. Its hyperparameters were not tuned for the GPU, and that is a finding too.

## 6. Bring the results back

Copy the archive to the laptop (USB, drive, email) and unpack it in the repository root:

```bash
tar xzf lab_results_<host>_<date>.tar.gz
```

`runs/` and `results/` are not tracked by git, so the archive is the only way they travel.
**Don't commit or push from the lab machine.** The commit checks live in local git hooks that
a fresh clone doesn't have, so changes to the branch are made on the laptop only.

## Troubleshooting

- `CUDA available: False`: the torch build doesn't match the driver; see step 2.
- Out of memory in `check_gpu.py` or the benchmark: the largest batch sizes (262,144
  episodes) need a few GB of GPU memory, and everything else is far smaller. Lower them in
  `scripts/lab/check_gpu.py` or `scripts/bench_throughput.py` if needed.
- `ModuleNotFoundError: rl_execution`: the environment isn't activated.
- PowerShell refuses to run scripts: use `-ExecutionPolicy Bypass` as shown.
