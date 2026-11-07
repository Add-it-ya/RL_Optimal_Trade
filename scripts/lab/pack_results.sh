#!/usr/bin/env bash
# Bundle lab output into one archive to bring back (runs/ and results/ are not in git).
#
#   bash scripts/lab/pack_results.sh e1_vec bench      # experiments to include
#
# Includes runs/<exp> and results/<exp> for each name given, plus results/lab_logs.
set -euo pipefail
cd "$(dirname "$0")/../.."

items=()
for exp in "$@"; do
  for d in "runs/$exp" "results/$exp"; do
    if [ -e "$d" ]; then items+=("$d"); fi
  done
done
if [ -d results/lab_logs ]; then items+=("results/lab_logs"); fi
if [ ${#items[@]} -eq 0 ]; then
  echo "nothing to pack" >&2
  exit 1
fi
out="lab_results_$(hostname)_$(date +%Y%m%d_%H%M).tar.gz"
tar czf "$out" "${items[@]}"
echo "Wrote $out ($(du -h "$out" | cut -f1)) with: ${items[*]}"
