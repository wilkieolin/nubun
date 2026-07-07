#!/usr/bin/env bash
# launch_single.sh — single-machine M6 training on this box (no DDP, no spark2).
#
# Usage:
#   bash scripts/launch_single.sh <checkpoint-name> [extra train_vqvae.py args...]
#
# Used in place of launch_ddp.sh when we want to keep the work on one box —
# either to avoid DDP coordination bugs or to leave spark2 free for other work.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT_NAME="${1:?usage: launch_single.sh <checkpoint-name> [args...]}"
shift

mkdir -p logs/single
LOG="logs/single/${CHECKPOINT_NAME}.log"

echo "==========================================="
echo "  single-machine launch"
echo "  checkpoint: $CHECKPOINT_NAME"
echo "  host:       $(hostname)"
echo "  log:        $LOG"
echo "  args:       $*"
echo "==========================================="

if [[ -f /tmp/hf_env.sh ]]; then source /tmp/hf_env.sh; fi

source /home/wilkie/miniconda3/etc/profile.d/conda.sh
conda activate nubun

PYTHONUNBUFFERED=1 python train_vqvae.py \
    --checkpoint-name="$CHECKPOINT_NAME" \
    "$@" 2>&1 | tee "$LOG"
