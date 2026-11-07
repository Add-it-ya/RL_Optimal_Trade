# Lab logs

Console logs of the three GPU runs behind the Phase 4 numbers in `RESULTS.md`, from the lab
machine (Ubuntu, NVIDIA RTX 4000 Ada, 20 GB, shared with other users; Python 3.12, torch
2.15 + CUDA 13.2). The only edit: the clone's absolute path is replaced by `<clone>`.

| log | run | what it records |
|---|---|---|
| `phase4_20261030_110835.log` | `scripts/lab/run_phase4.sh` at `8ca9303` | GPU parity check (`check_gpu.py`), CPU parity tests, throughput benchmark, VecPPO E1 training (`runs/e1_vec`), its evaluation |
| `tuning_20261030_122940.log` | `scripts/lab/run_tuning.sh` at `20f2f4a` | training configs C, D, B (A reused from `e1_vec`), scoring on the tuning block, the pick, the winner's test |
| `e2_tuned_20261030_145110.log` | `scripts/lab/run_e2_tuned.sh` at `20f2f4a` | E2 retrained with the winning config, evaluation, sweep |

Training times quoted in `RESULTS.md` are the `done in ...s` lines here.
