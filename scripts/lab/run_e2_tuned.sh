#!/usr/bin/env bash
# E2 again with the tuning study's winner, on the lab GPU (see scripts/lab/README.md).
#
#   nohup bash scripts/lab/run_e2_tuned.sh > /dev/null 2>&1 &
#   QUICK=1 bash scripts/lab/run_e2_tuned.sh        # dry run after QUICK=1 run_tuning.sh
#
# Reads the winner from results/tuning/choice.json (written by run_tuning.sh), trains VecPPO
# with that config on each signal strength (10 seeds each, IC 0 trained fresh), scores each
# where it was trained on the test block, then builds the sweep (improvement over AC and
# over SignalRule against IC) in results/e2_tuned/, and packs everything.
# Re-running is safe: finished seeds are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0-9}"
EPISODES="${EPISODES:-5000}"
QUICK="${QUICK:-0}"

PREFIX=""
EXTRA=()
if [ "$QUICK" = "1" ]; then
  PREFIX="quick_"
  EXTRA=(--timesteps 4000 --eval-every 2000 --n-envs 64 --val-episodes 32)
  EPISODES=40
fi

CHOICE="results/${PREFIX}tuning/choice.json"
if [ ! -f "$CHOICE" ]; then
  echo "no $CHOICE: run scripts/lab/run_tuning.sh first" >&2
  exit 1
fi
WINNER="$(python -c "import json; print(json.load(open('$CHOICE'))['winner'])")"
CONFIG="$(python -c "import json; print(json.load(open('$CHOICE'))['config'])")"

mkdir -p results/lab_logs
LOG="results/lab_logs/${PREFIX}e2_tuned_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== E2 with config $WINNER ($CONFIG) on $(hostname), $(date), commit $(git rev-parse --short HEAD)"

experiments=()
for ic in ic000 ic025 ic050 ic100 ic200; do
  exp="${PREFIX}e2t_${ic}"
  experiments+=("$exp")
  echo "== signal_${ic} -> runs/$exp"
  python -m rl_execution.cli train --config "$CONFIG" --device "$DEVICE" --seeds "$SEEDS" \
    --regime "signal_${ic}" --experiment "$exp" --progress ${EXTRA[@]+"${EXTRA[@]}"}
  python -m rl_execution.cli evaluate --experiment "$exp" --episodes "$EPISODES" --antithetic
  python -m rl_execution.cli report --experiment "$exp"
done

echo "== sweep"
python -m rl_execution.cli sweep --experiments "${experiments[@]}" --out "results/${PREFIX}e2_tuned"

echo "== archive"
bash scripts/lab/pack_results.sh "${experiments[@]}" "${PREFIX}e2_tuned" "${PREFIX}tuning"
echo "== done: results/${PREFIX}e2_tuned/SWEEP.md; log $LOG"
