#!/usr/bin/env bash
# bootstrap_spark2.sh — One-time setup of the second GB10 box for parallel sweeps.
# Idempotent: re-running is safe; existing pieces are skipped.
#
# Usage: bash scripts/bootstrap_spark2.sh [--with-opus100]
#
# Steps:
#   1. rsync code (excluding heavy data and caches)
#   2. create the conda env if missing
#   3. rsync small data files (FLORES corpus + frozen embedding table)
#   4. (optional) rsync the larger opus100 corpus

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-spark2}"
REMOTE_DIR="${REMOTE_DIR:-/home/wilkie/code/nubun}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
WITH_OPUS100=0
for arg in "$@"; do
  case "$arg" in
    --with-opus100) WITH_OPUS100=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

echo "==================================================="
echo "  bootstrapping $REMOTE_HOST:$REMOTE_DIR"
echo "  local source:   $LOCAL_DIR"
echo "  with opus100:   $([[ $WITH_OPUS100 -eq 1 ]] && echo yes || echo no)"
echo "==================================================="

echo ""
echo "Step 1/4 — rsync code (no data, no checkpoints)"
rsync -av --exclude=data --exclude=__pycache__ --exclude='*.pt' \
    --exclude='*.npz' --exclude='.git' --exclude='results' \
    "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo ""
echo "Step 2/4 — ensure nubun conda env exists on $REMOTE_HOST"
ENV_EXISTS=$(ssh -o BatchMode=yes "$REMOTE_HOST" \
    "ls -d /home/wilkie/miniconda3/envs/nubun 2>/dev/null && echo YES || echo NO" \
    | tail -1)
if [[ "$ENV_EXISTS" == "YES" ]]; then
    echo "  env already exists; skipping create"
else
    echo "  creating conda env (this takes ~10 min)..."
    ssh "$REMOTE_HOST" "source /home/wilkie/miniconda3/etc/profile.d/conda.sh && \
        cd $REMOTE_DIR && conda env create -f environment.yml"
fi

echo ""
echo "Step 3/4 — rsync FLORES corpus + embedding table (~390 MB)"
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR/data"
rsync -av --info=progress2 \
    "$LOCAL_DIR/data/parallel_corpus.npz" \
    "$LOCAL_DIR/data/embedding_table.pt" \
    "$REMOTE_HOST:$REMOTE_DIR/data/"

if [[ $WITH_OPUS100 -eq 1 ]]; then
    echo ""
    echo "Step 4/4 — rsync opus100 corpus shards (~2 GB)"
    if [[ -d "$LOCAL_DIR/data/opus100" ]]; then
        rsync -av --info=progress2 "$LOCAL_DIR/data/opus100/" \
            "$REMOTE_HOST:$REMOTE_DIR/data/opus100/"
    else
        echo "  WARNING: $LOCAL_DIR/data/opus100/ does not exist; skipping"
    fi
else
    echo ""
    echo "Step 4/4 — skipped (use --with-opus100 to also sync the big corpus)"
fi

echo ""
echo "==================================================="
echo "Verifying remote env..."
ssh "$REMOTE_HOST" "source /home/wilkie/miniconda3/etc/profile.d/conda.sh && \
    conda activate nubun && cd $REMOTE_DIR && \
    python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())' && \
    python -c 'from vqvae.model import VQVAE; print(\"VQVAE import OK\")' && \
    ls data/ | head"
echo "==================================================="
echo "Bootstrap done."
