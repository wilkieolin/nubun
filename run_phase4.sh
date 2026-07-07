#!/usr/bin/env bash
# run_phase4.sh — single-config Phase 4 training run.
# Args after --config-name=NAME are passed to train_vqvae.py verbatim.
# Used both directly and as the per-host command launched by launch_sweep.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f /tmp/hf_env.sh ]]; then source /tmp/hf_env.sh; fi

# Activate the nubun conda env. Non-interactive SSH usually doesn't have
# conda in PATH, so source the profile.d hook by absolute path explicitly.
CONDA_HOOKS=(
    "/home/wilkie/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
)
for hook in "${CONDA_HOOKS[@]}"; do
    if [[ -f "$hook" ]]; then
        source "$hook"
        break
    fi
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

CONFIG_NAME=""
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --config-name=*) CONFIG_NAME="${arg#*=}" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done
if [[ -z "$CONFIG_NAME" ]]; then
    echo "ERROR: --config-name=NAME is required" >&2
    exit 2
fi

mkdir -p logs results

echo "==================================================="
echo "  Config:    $CONFIG_NAME"
echo "  Host:      $(hostname)"
echo "  Started:   $(date -Iseconds)"
echo "  Args:      ${EXTRA_ARGS[*]}"
echo "==================================================="

# 1. Train
PYTHONUNBUFFERED=1 python train_vqvae.py \
    --checkpoint-name "$CONFIG_NAME" \
    "${EXTRA_ARGS[@]}"

# Find latest checkpoint matching the name
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

# 3. Codebook semantics (active codes only)
# Use smaller decode-chunk to keep memory bounded — full 1024 was OOM-killing
# the box during the analyze step (logits at chunk*T*V bf16 = ~7.5 GB per chunk).
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
