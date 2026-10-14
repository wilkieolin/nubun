#!/bin/bash
# Weekend run: validated B=128 recipe (A/B 2026-10-10: recon 4.42 vs m2 4.58 at
# 3.2M samples), one wafer, TOTAL steps with one cosine over the whole run,
# segments under the 23 h job cap chained with --resume, retries on failure.
#   setsid nohup cerebras/runs/long_b128.sh > logs/long_b128.out 2>&1 &
#   SEED=2 setsid nohup cerebras/runs/long_b128.sh > logs/long_b128_seed2.out 2>&1 &
# SEED (default 42, the Oct 10-12 run) seeds init, dropout and the data stream;
# a second seed is Phase 9 A4's converged cross-seed test.
set -u
cd "$(dirname "$0")/../.."
TOTAL=${TOTAL:-500000}          # 64M samples (~20x m2), ~45 h at ~400 samples/s
SEG=${SEG:-240000}              # steps per job (~21 h)
SEED=${SEED:-42}
TAG=long-b128; [[ "$SEED" != 42 ]] && TAG=long-b128-seed$SEED
OUT=${OUT:-$PWD/model_dir/$TAG-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
ARGS=(--batch-size 128 --lr 6e-4 --steps "$TOTAL" --warmup-steps 500
      --ckpt-every 10000 --keep-ckpts 3 --keep-every 50000 --log-every 1000
      --seed "$SEED" --out-dir "$OUT" --job-label "name=nubun-$TAG")
latest() { ls "$OUT"/nubun_cs_step*.mdl 2>/dev/null \
           | sed -E 's/.*step([0-9]+)\.mdl$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
step_of() { sed -E 's/.*step([0-9]+)\.mdl$/\1/' <<<"$1"; }
echo "=== $(date -u) start TOTAL=$TOTAL SEG=$SEG SEED=$SEED OUT=$OUT"
fails=0
while :; do
  CK=$(latest); START=0; [[ -n "$CK" ]] && START=$(step_of "$CK")
  (( START >= TOTAL )) && break
  END=$(( START + SEG )); (( END > TOTAL )) && END=$TOTAL
  RES=(); [[ -n "$CK" ]] && RES=(--resume "$CK")
  echo "=== $(date -u) segment $START -> $END ${CK:+(resume $(basename "$CK"))}"
  bash cerebras/submit_alcf.sh m2 "${ARGS[@]}" --max-steps "$END" "${RES[@]}"; rc=$?
  NEW=$(latest); NS=0; [[ -n "$NEW" ]] && NS=$(step_of "$NEW")
  echo "=== $(date -u) segment rc=$rc latest=${NEW:-none}"
  if (( rc != 0 )); then
    fails=$(( fails + 1 ))
    if (( NS <= START )); then   # no progress at all in this attempt
      (( fails >= 3 )) && { echo "=== 3 failures without progress — giving up"; exit 1; }
    fi
    echo "=== retrying from ${NEW:-scratch} in 120 s (failure $fails)"; sleep 120
  fi
done
echo "=== $(date -u) DONE: $(latest)"
