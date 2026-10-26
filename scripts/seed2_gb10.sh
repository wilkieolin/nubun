#!/usr/bin/env bash
# seed2_gb10.sh — PHASE9 §8: the GB10 follow-ups on a converged 500k CS-3 model
# (written for seed 2; takes any seed-N checkpoint of the same recipe).
#
# Runs, in order, skipping any step whose output already exists:
#   1. round-trip meaning, full code and --rvq-depth 2/4   (roundtrip_eval.py)
#   2. compositionality suite                              (compositionality_metrics.py)
#   3. opus100 code dump + code prior at depth 2/4/8       (dump_codes.py, code_prior.py)
#   4. level-0 decode of this model's inventory, under the memory watchdog
#      (analyze_codebook.py); the seed-42 decode too if it is missing
#   5. side-by-side report of the cross-seed-stable level-0 pairs for every
#      results/a4/pairs_<thisseed>_<other>_500k.json present
#      (scripts/stable_pairs_report.py), against the other model's level-0 decode
#
# The seed tag is read from the checkpoint path ("seedN" in it, else seed42), the
# step from the filename; outputs go to results/a4/<seed>_<step>/ so runs on
# different models cannot collide. Refuses anything but a 500k .eval.pt of this
# recipe (924642378 bytes) unless FORCE=1.
#
# Usage: bash scripts/seed2_gb10.sh [ckpt] [seed42_ckpt]
#   e.g. bash scripts/seed2_gb10.sh model_dir/long-b128-seed3-20261019T210243Z/nubun_cs_step500000.eval.pt
# Env:   OUT (results/a4/<seed>_<step>), DUMP (data/codes_cs3_<seed>_<step>), TAU (0.5),
#        SKIP="1 3" to skip steps, FLOOR_GB (20, watchdog floor), FORCE=1
set -euo pipefail
cd "$(dirname "$0")/.."

CK=${1:-model_dir/long-b128-seed2-20261014T160630Z/nubun_cs_step500000.eval.pt}
CK42=${2:-model_dir/long-b128-20261010T042150Z/nubun_cs_step500000.eval.pt}
STEP=$(basename "$CK" | grep -o 'step[0-9]*' | tr -d 'step' || true)
STEP=${STEP:-unknown}
SEED=$(grep -o 'seed[0-9]*' <<<"$CK" | head -1 || true)
SEED=${SEED:-seed42}
OUT=${OUT:-results/a4/${SEED}_$STEP}
DUMP=${DUMP:-data/codes_cs3_${SEED}_$STEP}
TAU=${TAU:-0.5}
SKIP=${SKIP:-}
FLOOR_GB=${FLOOR_GB:-20}
L0_42=${L0_42:-results/cs3_500k_codebook_L0.txt}
L0_ME=$OUT/codebook_L0_${SEED}_$STEP.txt
LOG=$OUT/seed2_gb10.log

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -u +%FT%TZ) seed2_gb10.sh start: $CK ($SEED, step $STEP) -> $OUT"

[[ -f "$CK" ]] || { echo "missing checkpoint: $CK"; exit 1; }
size=$(stat -c %s "$CK")
if [[ "$size" != 924642378 || "$STEP" != 500000 ]]; then
    echo "$CK: step $STEP, $size bytes. A converged 500k .eval.pt of this recipe is 924642378 bytes"
    echo "  (model_dir/long-b128*-<stamp>/nubun_cs_step500000.eval.pt on the ALCF node;"
    echo "  model_dir/seed2-b128-20261013T221048Z/ holds only the 50k seed-pair run)."
    if [[ "${FORCE:-0}" != 1 ]]; then echo "Refusing to run (FORCE=1 to override; outputs go to $OUT)."; exit 1; fi
    echo "FORCE=1: continuing; outputs tagged _$STEP in $OUT"
fi

skip() { [[ " $SKIP " == *" $1 "* ]]; }
step() { echo; echo "--- $(date -u +%FT%TZ) step $*"; }
done_if() { [[ -s "$1" ]] && { echo "exists, skipping: $1"; return 0; } || return 1; }

# 1. round-trip -------------------------------------------------------------
if ! skip 1; then
    step 1 "round-trip meaning (seed 42: 0.583 full, 0.346 depth 2, 0.478 depth 4; seed 2: 0.651 / 0.348 / 0.496)"
    f=$OUT/roundtrip_${SEED}_$STEP.txt
    done_if "$f" || python roundtrip_eval.py --checkpoint "$CK" --langs en,zh,es,fr,ar,ru \
        --combine-splits --max-gen-len 48 | tee "$f"
    for d in 2 4; do
        f=$OUT/roundtrip_${SEED}_${STEP}_depth$d.txt
        done_if "$f" || python roundtrip_eval.py --checkpoint "$CK" --langs en,zh,es,fr,ar,ru \
            --combine-splits --max-gen-len 48 --rvq-depth $d | tee "$f"
    done
fi

# 2. compositionality ---------------------------------------------------------
if ! skip 2; then
    step 2 "compositionality suite (compare PHASE9 §7 A3, the 500k columns)"
    f=$OUT/compositionality_cs3_${SEED}_$STEP.txt
    done_if "$f" || python compositionality_metrics.py --checkpoint "$CK" \
        --name cs3_${SEED}_$STEP --outdir "$OUT" | tee "$OUT/compositionality_${SEED}_$STEP.stdout"
fi

# 3. code dump + prior ---------------------------------------------------------
if ! skip 3; then
    step 3 "opus100 dump with the $SEED encoder, then code prior at depth 2/4/8"
    done_if "$DUMP/codes.npy" || python dump_codes.py --checkpoint "$CK" --out "$DUMP"
    for d in 2 4 8; do
        f=$OUT/prior_${SEED}_${STEP}_depth$d.txt
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
    step 4 "level-0 decode of $SEED (and seed 42 if $L0_42 is missing)"
    done_if "$L0_ME" || decode_l0 "$CK" "$L0_ME"
    if ! done_if "$L0_42"; then
        [[ -f "$CK42" ]] || { echo "missing seed-42 checkpoint $CK42 and no $L0_42"; exit 1; }
        decode_l0 "$CK42" "$L0_42"
    fi
fi

# 5. stable pairs reports ------------------------------------------------------
l0_of() {  # seed tag -> that model's level-0 decode
    if [[ "$1" == seed42 ]]; then echo "$L0_42"; else echo "results/a4/${1}_500000/codebook_L0_${1}_500000.txt"; fi
}
if ! skip 5 && [[ "$STEP" == 500000 ]]; then
    step 5 "cross-seed-stable level-0 pairs of $SEED, decoded in both models (J >= $TAU)"
    n=0
    for PAIRS in results/a4/pairs_${SEED}_seed*_500k.json; do
        [[ -f "$PAIRS" ]] || continue
        other=$(basename "$PAIRS" .json); other=${other#pairs_${SEED}_}; other=${other%_500k}
        L0_OTHER=$(l0_of "$other")
        if [[ ! -s "$L0_OTHER" ]]; then echo "no level-0 decode for $other at $L0_OTHER; skipping $PAIRS"; continue; fi
        python scripts/stable_pairs_report.py --pairs "$PAIRS" --a "$L0_ME" --b "$L0_OTHER" \
            --tau "$TAU" | tee "$OUT/stable_pairs_${SEED}_${other}_500k.txt"
        n=$((n + 1))
    done
    (( n > 0 )) || echo "no results/a4/pairs_${SEED}_seed*_500k.json found (run codebook_stability.py --out on the ALCF node, git pull)"
else
    echo "step 5 skipped: the pair lists are for the 500k models"
fi

echo
echo "=== $(date -u +%FT%TZ) done. Outputs in $OUT (gitignored: git add -f $OUT/<file>)."
