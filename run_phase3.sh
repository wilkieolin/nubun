#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional HF auth (sources /tmp/hf_env.sh if present, used for gated datasets)
if [[ -f /tmp/hf_env.sh ]]; then
    source /tmp/hf_env.sh
fi

# Activate conda env if not already active
if ! command -v python &>/dev/null || ! python -c "import torch" &>/dev/null; then
    if command -v conda &>/dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate nubun
    fi
fi

CHECKPOINT_NAME="${CHECKPOINT_NAME:-vqvae_phase3}"
STEPS="${STEPS:-50000}"

echo "=========================================="
echo "Phase 3: Sentence-Level VQ-VAE"
echo "checkpoint name: $CHECKPOINT_NAME"
echo "training steps:  $STEPS"
echo "=========================================="

if [[ ! -f data/parallel_corpus.npz ]]; then
    echo ""; echo "Step 1/5: Building parallel corpus from FLORES-200..."
    python build_parallel_corpus.py
else
    echo ""; echo "Step 1/5: Skipping corpus build (data/parallel_corpus.npz exists)"
fi

if [[ ! -f data/embedding_table.pt ]]; then
    echo ""; echo "Step 2/5: Caching MiniLM embedding table..."
    python -m vqvae.embedding_cache
else
    echo ""; echo "Step 2/5: Skipping embedding cache (data/embedding_table.pt exists)"
fi

echo ""; echo "Step 3/5: Training VQ-VAE for $STEPS steps..."
PYTHONUNBUFFERED=1 python train_vqvae.py \
    --steps "$STEPS" \
    --batch-size 32 \
    --bf16 \
    --use-ema --lambda-use 0.01 --reset-dead-every 250 \
    --use-stop-mask --lambda-len 0.05 \
    --src-langs all --tgt-langs all \
    --combine-splits \
    --log-every 500 --eval-every 1000 --ckpt-every 5000 \
    --checkpoint-name "$CHECKPOINT_NAME"

LATEST_CKPT="data/${CHECKPOINT_NAME}_step${STEPS}.pt"

echo ""; echo "Step 4/5: Held-out evaluation..."
python evaluate_vqvae.py \
    --checkpoint "$LATEST_CKPT" \
    --combine-splits

echo ""; echo "Step 5/5: Output-side semantic discovery..."
python analyze_codebook.py \
    --checkpoint "$LATEST_CKPT" \
    --n-samples 256

echo ""; echo "=========================================="
echo "Phase 3 done."
echo "  eval:        results/vqvae_eval.txt"
echo "  semantics:   results/codebook_semantics.txt"
echo "  checkpoint:  $LATEST_CKPT"
echo "=========================================="
