#!/bin/bash
# Phase 9 branch 2, first wafer test of minimal pairs (PHASE9 §8): two runs on
# the seed_pair.sh recipe (B=128, lr 6e-4, 50k steps, seed 1), so each compares
# with that script's seed-1 run, which differs only in the data:
#   mix      PAIRS pairs in the first 2*PAIRS rows of every batch, no new loss.
#            Same graph as the baseline (the compile cache should hit).
#   penalty  the same, plus LAMBDA * relu(level-0 slot Hamming - BUDGET) per
#            pair (vqvae/pairs.py). New graph: run `compile` first.
# Data, once, on the user node (minpairs from `augment/minpairs.py package
# --max-len 64`, copied to $DATA_DIR/minpairs; data/minpairs_medium64 already
# carries its .sem.npz, so skip the precompute for it):
#   python cerebras/precompute_opus_semantic.py --opus-dir $DATA_DIR/minpairs
#   bash cerebras/unpack_opus.sh $DATA_DIR/minpairs $DATA_DIR/minpairs_npy
# Usage:
#   bash cerebras/runs/minpairs_pair.sh compile       # compile-only check of the penalty graph
#   bash cerebras/runs/minpairs_pair.sh               # both runs, one wafer each
#   RUNS=mix bash cerebras/runs/minpairs_pair.sh
# Afterwards: export with cerebras/export_ckpt.py, then scripts/pair_hamming.py
# and the usual round-trip / codebook_stability readouts.
set -u
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-$PWD/data}"   # as cerebras/submit_alcf.sh
MINPAIRS="${MINPAIRS:-$DATA_DIR/minpairs_npy}"
STEPS=${STEPS:-50000}
SEED=${SEED:-1}
PAIRS=${PAIRS:-8}               # 16 of 128 rows = 12.5% minimal pairs
LAMBDA=${LAMBDA:-1.0}             # the GB10 A/B value (PHASE9 §8)
BUDGET=${BUDGET:-2}
RUNS=${RUNS:-"mix penalty"}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
COMMON=(--batch-size 128 --lr 6e-4 --steps "$STEPS" --warmup-steps 500
        --ckpt-every 10000 --keep-ckpts 1 --log-every 1000 --seed "$SEED"
        --minpairs-dir "$MINPAIRS" --pairs-per-batch "$PAIRS")
# --micro-batch-size 128: with the penalty's partner pass (partner_src_ids) every
# micro-batch 32/64 floorplan aborts the compile ("Memory allocation was
# unsuccessful ... at the top level", also with --min-micro-batch 64); the full
# batch has no accumulation and the same gradient.
PEN=(--lambda-pair "$LAMBDA" --pair-budget "$BUDGET" --pair-levels 1 --micro-batch-size 128)
[[ -d "$MINPAIRS" ]] || { echo "ERROR: $MINPAIRS missing (see header)" >&2; exit 2; }
mkdir -p logs
if [[ "${1:-}" == compile ]]; then
  exec bash cerebras/submit_alcf.sh m2 "${COMMON[@]}" "${PEN[@]}" --compile-only \
      --out-dir "$PWD/model_dir/minpairs-compile-$STAMP" --job-label name=nubun-mp-compile
fi
for r in $RUNS; do
  OUT=$PWD/model_dir/minpairs-$r-seed$SEED-$STAMP
  mkdir -p "$OUT"
  EXTRA=(); [[ "$r" == penalty ]] && EXTRA=("${PEN[@]}")
  echo "=== $(date -u) launch $r seed=$SEED steps=$STEPS out=$OUT log=logs/minpairs-$r-$STAMP.out"
  setsid nohup bash cerebras/submit_alcf.sh m2 "${COMMON[@]}" "${EXTRA[@]}" \
      --out-dir "$OUT" --job-label "name=nubun-mp-$r" \
      > "logs/minpairs-$r-$STAMP.out" 2>&1 &
  sleep 5
done
echo "=== monitor: csctl get jobs | grep nubun-mp ; tail -f logs/minpairs-*-$STAMP.out"
