#!/usr/bin/env bash
# Phase 4 on a GPU machine, start to finish (see scripts/lab/README.md).
#
#   source .venv/bin/activate
#   bash scripts/lab/run_phase4.sh
#
# 1. bring-up check (GPU parity against the numpy reference, GPU throughput);
# 2. the CPU parity tests (tests/test_vec_env.py);
# 3. the throughput benchmark -> results/bench/throughput.csv;
# 4. parity test 3: VecPPO on the E1 market, 10 seeds x 5M steps, scored on the numpy env
#    -> runs/e1_vec, results/e1_vec/REPORT.md;
# 5. one archive with everything to bring back.
#
# Override with environment variables, e.g. SEEDS=0-2 STEPS=1000000 bash scripts/lab/run_phase4.sh
# Re-running is safe: finished seeds are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS="${SEEDS:-0-9}"
STEPS="${STEPS:-5000000}"
DEVICE="${DEVICE:-cuda}"
EPISODES="${EPISODES:-5000}"
EXPERIMENT="${EXPERIMENT:-e1_vec}"

mkdir -p results/lab_logs
LOG="results/lab_logs/phase4_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== Phase 4 on $(hostname), $(date), commit $(git rev-parse --short HEAD)"

echo "== 1/5 bring-up check"
python scripts/lab/check_gpu.py

echo "== 2/5 CPU parity tests"
python -m pytest -q -p no:cacheprovider tests/test_vec_env.py

echo "== 3/5 throughput benchmark"
python scripts/bench_throughput.py --out results/bench

echo "== 4/5 VecPPO on the E1 market ($SEEDS, $STEPS steps, $DEVICE)"
python -m rl_execution.cli train --agent vecppo --device "$DEVICE" --seeds "$SEEDS" \
  --timesteps "$STEPS" --regime abm_ac_optimal --experiment "$EXPERIMENT" --progress
python -m rl_execution.cli evaluate --experiment "$EXPERIMENT" --episodes "$EPISODES" --antithetic
python -m rl_execution.cli report --experiment "$EXPERIMENT"

echo "== 5/5 archive"
bash scripts/lab/pack_results.sh "$EXPERIMENT" bench
echo "== done; log: $LOG"
