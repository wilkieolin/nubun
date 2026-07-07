#!/usr/bin/env bash
# run_phase5.sh — Phase 5: semantic-target loss (Option 3).
#
# Adds an auxiliary cosine loss pulling the pooled quantized bottleneck toward
# the frozen MiniLM sentence embedding of the source, so codes encode meaning
# instead of high-frequency token/punctuation glue. The AR decoder + token
# reconstruction stay intact (needed for translation eval + codebook discovery).
#
# Defaults to a short PILOT run. Override via env:
#   LAMBDA_SEM=2.0   STEPS=15000   CONFIG_NAME=phase5_pilot   ./run_phase5.sh
# For the A/B baseline, run with LAMBDA_SEM=0 (semantic loss disabled but head
# present) or just re-use the Phase 4 checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f /tmp/hf_env.sh ]]; then source /tmp/hf_env.sh; fi

# Activate the nubun conda env (non-interactive SSH lacks conda in PATH).
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

# Tunables
CONFIG_NAME="${CONFIG_NAME:-phase5_pilot}"
LAMBDA_SEM="${LAMBDA_SEM:-2.0}"
STEPS="${STEPS:-15000}"

mkdir -p logs results

echo "==================================================="
echo "  Phase 5: semantic-target loss"
echo "  Config:     $CONFIG_NAME"
echo "  lambda_sem: $LAMBDA_SEM"
echo "  steps:      $STEPS"
echo "  Host:       $(hostname)"
echo "  Started:    $(date -Iseconds)"
echo "==================================================="

# 1. Train — same backbone config as the Phase 4 final run, plus semantic head.
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
    --lambda-sem "$LAMBDA_SEM"

# Find latest checkpoint matching the name
LATEST=$(ls -t data/${CONFIG_NAME}_step*.pt 2>/dev/null | head -1)
if [[ -z "$LATEST" ]]; then
    echo "ERROR: no checkpoint matching data/${CONFIG_NAME}_step*.pt" >&2
    exit 3
fi
echo ""
echo "Using checkpoint: $LATEST"

# 2. Eval — translation accuracy across all (src,tgt) language pairs
echo ""
echo "--- evaluating ---"
python evaluate_vqvae.py --checkpoint "$LATEST" --combine-splits \
    --output "results/${CONFIG_NAME}_eval.txt"

# 3. Codebook semantics — did codes shift from punctuation to content words?
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
