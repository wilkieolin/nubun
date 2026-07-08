#!/usr/bin/env bash
# run_phase5c.sh — Phase 5c: word-dropout sweep to break posterior collapse.
#
# The Step-0 ablation proved the decoder ignores the codes (swapping in a
# different sentence's codes cost only ~2-3% content-acc). Word dropout removes
# the teacher-forced prefix crutch so the decoder MUST read the codes. We sweep
# the dropout rate {0.15, 0.30, 0.50} and track the shuffle-gap (real - shuffle
# content-acc) as the KPI — it should GROW as dropout forces code reliance.
#
# Also adds the length-predictor head (--use-length-head) as the stop fix, and
# evaluates WITH the hard cap (fixed in evaluate_vqvae.py) so avg_bn is meaningful.
#
# Builds on Phase 5b config: weighted CE + lambda_sem=5.
# Override:  RATES="0.15 0.30 0.50"  STEPS=15000  ./run_phase5c.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f /tmp/hf_env.sh ]]; then source /tmp/hf_env.sh; fi
CONDA_HOOKS=(
    "/home/wilkie/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
)
for hook in "${CONDA_HOOKS[@]}"; do
    if [[ -f "$hook" ]]; then source "$hook"; break; fi
done
command -v conda &>/dev/null || { echo "ERROR: conda not found" >&2; exit 4; }
conda activate nubun
python -c "import vqvae" 2>/dev/null || { echo "ERROR: vqvae not importable" >&2; exit 5; }

RATES="${RATES:-0.15 0.30 0.50}"
STEPS="${STEPS:-15000}"
LAMBDA_SEM="${LAMBDA_SEM:-5.0}"
TOKEN_WEIGHTS="data/token_weights.pt"
mkdir -p logs results

[[ -f "$TOKEN_WEIGHTS" ]] || python build_token_weights.py --output "$TOKEN_WEIGHTS"

for RATE in $RATES; do
    TAG=$(python -c "print(f'{int(float(\"$RATE\")*100):02d}')")
    CONFIG_NAME="phase5c_wd${TAG}"
    echo "==================================================="
    echo "  Phase 5c  |  word_dropout=$RATE  |  $CONFIG_NAME"
    echo "  steps=$STEPS  lambda_sem=$LAMBDA_SEM  length-head=on  weighted-CE=on"
    echo "  Started: $(date -Iseconds)"
    echo "==================================================="

    PYTHONUNBUFFERED=1 python train_vqvae.py \
        --checkpoint-name "$CONFIG_NAME" \
        --corpus opus100 --steps "$STEPS" --batch-size 32 --bf16 \
        --k 128 --m-max 64 \
        --compression-ratio 0.7 --length-slack 4 --target-avg-len 24 --lambda-len-lr 0.001 \
        --use-ema --lambda-use 0.01 --reset-dead-every 250 --use-stop-mask \
        --combine-splits --src-langs all --tgt-langs all \
        --log-every 500 --eval-every 1000 --ckpt-every 5000 \
        --use-semantic-head --lambda-sem "$LAMBDA_SEM" \
        --use-token-weights --token-weights "$TOKEN_WEIGHTS" \
        --word-dropout "$RATE" --use-length-head --lambda-lenpred 0.1

    LATEST=$(ls -t data/${CONFIG_NAME}_step*.pt 2>/dev/null | head -1)
    [[ -z "$LATEST" ]] && { echo "ERROR: no checkpoint for $CONFIG_NAME" >&2; exit 3; }
    echo "checkpoint: $LATEST"

    echo "--- evaluating (capped) ---"
    python evaluate_vqvae.py --checkpoint "$LATEST" --combine-splits \
        --token-weights "$TOKEN_WEIGHTS" --output "results/${CONFIG_NAME}_eval.txt"

    echo "--- KPI: ablation / shuffle-gap ---"
    python diagnose_ablation.py --checkpoint "$LATEST" --combine-splits \
        2>&1 | tee "results/${CONFIG_NAME}_ablation.txt"

    echo "--- codebook semantics ---"
    python analyze_codebook.py --checkpoint "$LATEST" --active-only \
        --decode-chunk 256 --output "results/${CONFIG_NAME}_semantics.txt"

    echo "Done: $CONFIG_NAME (wd=$RATE)"
done

echo ""
echo "ALLDONE — phase5c sweep complete. Shuffle-gap KPI per config:"
grep -H "real - shuffle" results/phase5c_wd*_ablation.txt 2>/dev/null || true
