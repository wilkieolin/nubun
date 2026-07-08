#!/usr/bin/env bash
# run_phase5b.sh — Phase 5b: semantic loss (stronger) + frequency-weighted CE.
#
# Builds on the Phase 5 pilot. Two changes aimed at DISPLACING (not just nudging)
# the residual punctuation/function-word content in the codes:
#   1. --lambda-sem 5   (up from 2.0): stronger meaning gradient into bottleneck
#   2. --use-token-weights: downweight frequent punctuation/glue in the recon CE
#      so token-CE and the semantic loss pull the SAME direction instead of
#      fighting.
#
# Override via env:  LAMBDA_SEM=5  STEPS=15000  CONFIG_NAME=phase5b  ./run_phase5b.sh

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
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found via known hook paths" >&2
    exit 4
fi
conda activate nubun
if ! python -c "import vqvae" &>/dev/null 2>&1; then
    echo "ERROR: vqvae not importable after conda activate (CWD=$(pwd))" >&2
    exit 5
fi

CONFIG_NAME="${CONFIG_NAME:-phase5b}"
LAMBDA_SEM="${LAMBDA_SEM:-5.0}"
STEPS="${STEPS:-15000}"
TOKEN_WEIGHTS="data/token_weights.pt"

mkdir -p logs results

# Build the per-token weight vector if it doesn't exist yet.
if [[ ! -f "$TOKEN_WEIGHTS" ]]; then
    echo "--- building token weights ($TOKEN_WEIGHTS not found) ---"
    python build_token_weights.py --output "$TOKEN_WEIGHTS"
fi

echo "==================================================="
echo "  Phase 5b: strong semantic loss + weighted CE"
echo "  Config:     $CONFIG_NAME"
echo "  lambda_sem: $LAMBDA_SEM"
echo "  steps:      $STEPS"
echo "  Host:       $(hostname)"
echo "  Started:    $(date -Iseconds)"
echo "==================================================="

# 1. Train
PYTHONUNBUFFERED=1 python train_vqvae.py \
    --checkpoint-name "$CONFIG_NAME" \
    --corpus opus100 \
    --steps "$STEPS" \
    --batch-size 32 \
    --bf16 \
    --k 128 \
    --m-max 64 \
    --compression-ratio 0.7 \
    --length-slack 4 \
    --target-avg-len 24 \
    --lambda-len-lr 0.001 \
    --use-ema \
    --lambda-use 0.01 \
    --reset-dead-every 250 \
    --use-stop-mask \
    --combine-splits \
    --src-langs all \
    --tgt-langs all \
    --log-every 500 \
    --eval-every 1000 \
    --ckpt-every 5000 \
    --use-semantic-head \
    --lambda-sem "$LAMBDA_SEM" \
    --use-token-weights \
    --token-weights "$TOKEN_WEIGHTS"

LATEST=$(ls -t data/${CONFIG_NAME}_step*.pt 2>/dev/null | head -1)
if [[ -z "$LATEST" ]]; then
    echo "ERROR: no checkpoint matching data/${CONFIG_NAME}_step*.pt" >&2
    exit 3
fi
echo ""
echo "Using checkpoint: $LATEST"

# 2. Eval
echo ""
echo "--- evaluating ---"
python evaluate_vqvae.py --checkpoint "$LATEST" --combine-splits \
    --output "results/${CONFIG_NAME}_eval.txt"

# 3. Codebook semantics
echo ""
echo "--- analyzing codebook ---"
python analyze_codebook.py --checkpoint "$LATEST" --active-only \
    --decode-chunk 256 \
    --output "results/${CONFIG_NAME}_semantics.txt"

echo ""
echo "Done: $CONFIG_NAME"
echo "  checkpoint: $LATEST"
echo "  eval:       results/${CONFIG_NAME}_eval.txt"
echo "  semantics:  results/${CONFIG_NAME}_semantics.txt"
