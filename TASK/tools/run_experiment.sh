#!/usr/bin/env bash
# Run one benchmark and keep everything needed to reproduce and report it.
#
#   TASK/tools/run_experiment.sh <name> <model.zip> <seed_file> [workers] [baseline_name]
#
# Examples:
#   TASK/tools/run_experiment.sh champion_v40   TASK/champion/submission.zip   village_subset_seeds.json
#   TASK/tools/run_experiment.sh village_low_5  Submission/village_low_5.zip   village_subset_seeds.json 6 champion_v40
#
# Writes results/<name>/:
#   benchmark.log      full console output (also printed live)
#   results.json       per-seed results (--summary-json-out)
#   seeds.json         exact seed file used
#   meta.txt           model path, sha256, seed file, workers, date, git commit
#   patch.diff         drone_agent.py diff against the champion
#   compare_vs_<baseline>.txt   paired comparison (only if baseline_name is given)
set -euo pipefail

NAME=${1:?name}; MODEL=${2:?model zip}; SEEDS=${3:?seed file}
WORKERS=${4:-6}; BASE=${5:-}
CHAMPION=TASK/champion/submission.zip
OUT=results/$NAME
mkdir -p "$OUT"

cp "$SEEDS" "$OUT/seeds.json"
{
  echo "name:     $NAME"
  echo "model:    $MODEL"
  echo "sha256:   $(sha256sum "$MODEL" | cut -d' ' -f1)"
  echo "seeds:    $SEEDS"
  echo "workers:  $WORKERS"
  echo "date:     $(date -Is)"
  echo "commit:   $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
} | tee "$OUT/meta.txt"

diff <(unzip -p "$CHAMPION" drone_agent.py) <(unzip -p "$MODEL" drone_agent.py) \
  > "$OUT/patch.diff" || true
echo "patch.diff: $(grep -c '^[<>]' "$OUT/patch.diff" || true) changed lines vs champion"

swarm benchmark --model "$MODEL" --family-id cf_search_and_rescue \
  --seed-file "$SEEDS" --workers "$WORKERS" --relax-timeouts \
  --summary-json-out "$OUT/results.json" 2>&1 | tee "$OUT/benchmark.log"

if [[ -n "$BASE" ]]; then
  python TASK/tools/compare_runs.py "results/$BASE/results.json" "$OUT/results.json" \
    | tee "$OUT/compare_vs_${BASE}.txt"
fi
echo "Saved in $OUT/"
