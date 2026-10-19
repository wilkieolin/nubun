#!/usr/bin/env bash
# seed2_gb10.sh — PHASE9 §8 step (3): the GB10 follow-ups on the seed-2 500k model.
#
# Runs, in order, skipping any step whose output already exists:
#   1. round-trip meaning, full code and --rvq-depth 2/4   (roundtrip_eval.py)
#   2. compositionality suite                              (compositionality_metrics.py)
#   3. opus100 code dump + code prior at depth 2/4/8       (dump_codes.py, code_prior.py)
#   4. level-0 decode of the seed-2 inventory, under the memory watchdog
#      (analyze_codebook.py); the seed-42 decode too if it is missing
#   5. side-by-side report of the cross-seed-stable level-0 pairs
#      (scripts/stable_pairs_report.py on results/a4/pairs_seed2_seed42_500k.json)
#
# Outputs are tagged with the checkpoint's step (from its filename), so a run on
# the wrong checkpoint cannot overwrite or pre-empt the right one. The script
# refuses any file other than the converged 500k seed-2 .eval.pt unless FORCE=1.
#
# Usage: bash scripts/seed2_gb10.sh [seed2_ckpt] [seed42_ckpt]
# Env:   OUT (results/a4/seed2_<step>), DUMP (data/codes_cs3_seed2_<step>), TAU (0.5),
#        SKIP="1 3" to skip steps, FLOOR_GB (20, watchdog floor), FORCE=1
set -euo pipefail
cd "$(dirname "$0")/.."

CK=${1:-model_dir/long-b128-seed2-20261014T160630Z/nubun_cs_step500000.eval.pt}
CK42=${2:-model_dir/long-b128-20261010T042150Z/nubun_cs_step500000.eval.pt}
STEP=$(basename "$CK" | grep -o 'step[0-9]*' | tr -d 'step' || true)
STEP=${STEP:-unknown}
OUT=${OUT:-results/a4/seed2_$STEP}
DUMP=${DUMP:-data/codes_cs3_seed2_$STEP}
TAU=${TAU:-0.5}
SKIP=${SKIP:-}
FLOOR_GB=${FLOOR_GB:-20}
L0_42=${L0_42:-results/cs3_500k_codebook_L0.txt}
L0_2=$OUT/codebook_L0_seed2_$STEP.txt
PAIRS=results/a4/pairs_seed2_seed42_500k.json
LOG=$OUT/seed2_gb10.log

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -u +%FT%TZ) seed2_gb10.sh start"

[[ -f "$CK" ]] || { echo "missing seed-2 checkpoint: $CK"; exit 1; }
size=$(stat -c %s "$CK")
if [[ "$size" != 924642378 || "$STEP" != 500000 ]]; then
    echo "$CK: step $STEP, $size bytes. The converged seed-2 model is"
    echo "  model_dir/long-b128-seed2-20261014T160630Z/nubun_cs_step500000.eval.pt (924642378 bytes),"
    echo "  on the ALCF node; model_dir/seed2-b128-20261013T221048Z/ holds only the 50k seed-pair run."
    if [[ "${FORCE:-0}" != 1 ]]; then echo "Refusing to run (FORCE=1 to override; outputs go to $OUT)."; exit 1; fi
    echo "FORCE=1: continuing; outputs tagged _$STEP in $OUT"
fi
[[ -f "$PAIRS" ]] || { echo "missing $PAIRS (git pull origin main)"; exit 1; }

skip() { [[ " $SKIP " == *" $1 "* ]]; }
step() { echo; echo "--- $(date -u +%FT%TZ) step $*"; }
done_if() { [[ -s "$1" ]] && { echo "exists, skipping: $1"; return 0; } || return 1; }

# 1. round-trip -------------------------------------------------------------
if ! skip 1; then
    step 1 "round-trip meaning (seed 42: 0.582 full, 0.346 depth 2, 0.478 depth 4)"
    f=$OUT/roundtrip_seed2_$STEP.txt
    done_if "$f" || python roundtrip_eval.py --checkpoint "$CK" --langs en,zh,es,fr,ar,ru \
        --combine-splits --max-gen-len 48 | tee "$f"
    for d in 2 4; do
        f=$OUT/roundtrip_seed2_${STEP}_depth$d.txt
        done_if "$f" || python roundtrip_eval.py --checkpoint "$CK" --langs en,zh,es,fr,ar,ru \
            --combine-splits --max-gen-len 48 --rvq-depth $d | tee "$f"
    done
fi

# 2. compositionality ---------------------------------------------------------
if ! skip 2; then
    step 2 "compositionality suite (compare PHASE9 §7 A3, CS-3 @500k column)"
    f=$OUT/compositionality_cs3_seed2_$STEP.txt
    done_if "$f" || python compositionality_metrics.py --checkpoint "$CK" \
        --name cs3_seed2_$STEP --outdir "$OUT" | tee "$OUT/compositionality_seed2_$STEP.stdout"
fi

# 3. code dump + prior ---------------------------------------------------------
if ! skip 3; then
    step 3 "opus100 dump with the seed-2 encoder, then code prior at depth 2/4/8"
    done_if "$DUMP/codes.npy" || python dump_codes.py --checkpoint "$CK" --out "$DUMP"
    for d in 2 4 8; do
        f=$OUT/prior_seed2_${STEP}_depth$d.txt
        done_if "$f" || python code_prior.py --data "$DUMP" --mode codes --depth $d \
            --outdir "$OUT" | tee "$f"
    done
    echo "(text baseline unchanged: same sampling seed, 164.7 opus / 378.8 FLORES bits)"
fi

# 4. level-0 decodes -----------------------------------------------------------
decode_l0() {  # ckpt out
    python analyze_codebook.py --checkpoint "$1" --rvq-level 0 --output "$2" &
    pid=$!
    bash scripts/mem_watchdog.sh $pid "$FLOOR_GB" &
    wd=$!
    wait $pid; rc=$?
    kill $wd 2>/dev/null || true
    return $rc
}
if ! skip 4; then
    step 4 "level-0 decode of seed 2 (and seed 42 if $L0_42 is missing)"
    done_if "$L0_2" || decode_l0 "$CK" "$L0_2"
    if ! done_if "$L0_42"; then
        [[ -f "$CK42" ]] || { echo "missing seed-42 checkpoint $CK42 and no $L0_42"; exit 1; }
        decode_l0 "$CK42" "$L0_42"
    fi
fi

# 5. stable pairs report -------------------------------------------------------
if ! skip 5 && [[ "$STEP" == 500000 ]]; then
    step 5 "cross-seed-stable level-0 pairs, decoded in both models (J >= $TAU)"
    python scripts/stable_pairs_report.py --pairs "$PAIRS" --a "$L0_2" --b "$L0_42" \
        --tau "$TAU" | tee "$OUT/stable_pairs_seed2_seed42_500k.txt"
else
    echo "step 5 skipped: the pair list is for the 500k models"
fi

echo
echo "=== $(date -u +%FT%TZ) done. Outputs in $OUT (gitignored: git add -f $OUT/<file>)."
