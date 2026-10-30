#!/usr/bin/env bash
# Tuning study on E1, on the lab GPU (see scripts/lab/README.md, "Tuning rerun").
#
#   source .venv/bin/activate
#   nohup bash scripts/lab/run_tuning.sh > /dev/null 2>&1 &
#   QUICK=1 bash scripts/lab/run_tuning.sh          # a few-minute dry run (quick_* names)
#
# 1. train configs A-D (scripts/lab/configs/), 10 seeds each, on the E1 market;
#    A is the Phase 4 run (runs/e1_vec), so its seeds are skipped if already there;
# 2. score every config on the TUNING block (seeds from 1,500,000), never the test block;
# 3. pick the winner with scripts/lab/pick_config.py (rule fixed in advance);
# 4. score only the winner on the test block (seeds from 2,000,000) and write its report;
# 5. one archive with everything.
# Then run scripts/lab/run_e2_tuned.sh to redo E2 with the winner.
# Re-running is safe: finished seeds are skipped; evaluations are redone.
set -euo pipefail
cd "$(dirname "$0")/../.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0-9}"
TUNE_EPISODES="${TUNE_EPISODES:-2000}"
TEST_EPISODES="${TEST_EPISODES:-5000}"
QUICK="${QUICK:-0}"
TUNING_SEED=1500000

PREFIX=""
EXTRA=()
if [ "$QUICK" = "1" ]; then
  PREFIX="quick_"
  EXTRA=(--timesteps 4000 --eval-every 2000 --n-envs 64 --val-episodes 32)
  TUNE_EPISODES=40
  TEST_EPISODES=40
fi

# cheapest first (expected GPU minutes for 10 seeds: A ~20, C ~21, D ~23, B ~82)
CONFIGS=(A C D B)
declare -A FILE=([A]=A_n256_5m [B]=B_n256_20m [C]=C_n4096_50m [D]=D_n16384_200m)
experiment() {  # A is the Phase 4 run itself
  if [ "$1" = "A" ] && [ "$QUICK" != "1" ]; then echo "e1_vec"; else echo "${PREFIX}e1t_$1"; fi
}

mkdir -p results/lab_logs
LOG="results/lab_logs/${PREFIX}tuning_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== tuning study on $(hostname), $(date), commit $(git rev-parse --short HEAD), QUICK=$QUICK"

echo "== 1/5 train"
for c in "${CONFIGS[@]}"; do
  echo "-- config $c (${FILE[$c]}) -> runs/$(experiment "$c")"
  python -m rl_execution.cli train --config "scripts/lab/configs/${FILE[$c]}.yaml" \
    --device "$DEVICE" --seeds "$SEEDS" --regime abm_ac_optimal \
    --experiment "$(experiment "$c")" --progress ${EXTRA[@]+"${EXTRA[@]}"}
done

echo "== 2/5 score every config on the tuning block (seeds from $TUNING_SEED)"
candidates=()
for c in "${CONFIGS[@]}"; do
  exp="$(experiment "$c")"
  python -m rl_execution.cli evaluate --experiment "$exp" --base-seed "$TUNING_SEED" \
    --episodes "$TUNE_EPISODES" --antithetic --out "results/${exp}_tuning"
  python -m rl_execution.cli report --experiment "$exp" --results "results/${exp}_tuning"
  candidates+=("$c=results/${exp}_tuning")
done

echo "== 3/5 pick the winner"
python scripts/lab/pick_config.py --out "results/${PREFIX}tuning" "${candidates[@]}"
WINNER="$(python -c "import json; print(json.load(open('results/${PREFIX}tuning/choice.json'))['winner'])")"
WIN_EXP="$(experiment "$WINNER")"

echo "== 4/5 score the winner ($WINNER, runs/$WIN_EXP) on the test block"
python -m rl_execution.cli evaluate --experiment "$WIN_EXP" --episodes "$TEST_EPISODES" --antithetic
python -m rl_execution.cli report --experiment "$WIN_EXP"

echo "== 5/5 archive"
names=("${PREFIX}tuning")
for c in "${CONFIGS[@]}"; do
  exp="$(experiment "$c")"
  names+=("$exp" "${exp}_tuning")
done
bash scripts/lab/pack_results.sh "${names[@]}"
echo "== done: winner $WINNER; test report results/$WIN_EXP/REPORT.md; log $LOG"
echo "== next: bash scripts/lab/run_e2_tuned.sh"
