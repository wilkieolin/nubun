#!/bin/bash
# Phase 9 A4 — cross-seed stability. Two runs that differ ONLY in --seed
# (init + dropout stream + data order), the validated B=128 recipe, a short
# cosine (STEPS total), one wafer each, launched together so both finish in
# ~5 h. Afterwards: export each nubun_cs_step${STEPS}.mdl with
# cerebras/export_ckpt.py and compare with codebook_stability.py.
#   bash cerebras/runs/seed_pair.sh            # seeds 1 and 2
#   SEEDS="3 4" STEPS=50000 bash cerebras/runs/seed_pair.sh
set -u
cd "$(dirname "$0")/../.."
STEPS=${STEPS:-50000}
SEEDS=${SEEDS:-"1 2"}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
for s in $SEEDS; do
  OUT=$PWD/model_dir/seed$s-b128-$STAMP
  mkdir -p "$OUT"
  echo "=== $(date -u) launch seed=$s steps=$STEPS out=$OUT log=logs/seed$s-$STAMP.out"
  setsid nohup bash cerebras/submit_alcf.sh m2 \
      --batch-size 128 --lr 6e-4 --steps "$STEPS" --warmup-steps 500 \
      --ckpt-every 10000 --log-every 1000 --seed "$s" \
      --out-dir "$OUT" --job-label "name=nubun-seed$s" \
      > "logs/seed$s-$STAMP.out" 2>&1 &
  sleep 5
done
echo "=== monitor: csctl get jobs | grep nubun-seed ; tail -f logs/seed*-$STAMP.out"
