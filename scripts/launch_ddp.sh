#!/usr/bin/env bash
# launch_ddp.sh — Launch a 2-machine DDP training run across localhost + spark2.
#
# Usage:
#   bash scripts/launch_ddp.sh <config-name> [extra train_vqvae.py args...]
#
# Example (Phase 4 M6 final run with K=128 + loose-compression):
#   bash scripts/launch_ddp.sh vqvae_phase4_final \
#       --steps 100000 --batch-size 32 --bf16 --use-ema --lambda-use 0.01 \
#       --reset-dead-every 250 --use-stop-mask --combine-splits \
#       --src-langs all --tgt-langs all --log-every 500 --eval-every 1000 \
#       --ckpt-every 5000 --corpus opus100 --k 128 --m-max 64 \
#       --compression-ratio 0.7 --length-slack 4 --target-avg-len 24 \
#       --lambda-len-lr 0.001
#
# What it does:
#   - Picks a free TCP port on localhost
#   - Launches torchrun on localhost as node_rank=0 (master)
#   - Launches torchrun on spark2 (over SSH) as node_rank=1
#   - Streams both stdouts to local log files
#   - Waits for both to finish

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT_NAME="${1:?usage: launch_ddp.sh <checkpoint-name> [args...]}"
shift

REMOTE_HOST="${REMOTE_HOST:-spark2}"
REMOTE_DIR="${REMOTE_DIR:-/home/wilkie/code/nubun}"
# Use the 200GbE direct cross-connect (enp1s0f0np0) — firewalled on public IPs,
# but this link is open and gives sub-millisecond latency between the two GB10s.
MASTER_HOST="${MASTER_HOST:-169.254.102.104}"
MASTER_PORT="${MASTER_PORT:-29500}"
# Force gloo to bind to the cross-connect interface on both sides
GLOO_IFACE="${GLOO_IFACE:-enp1s0f0np0}"

mkdir -p logs/ddp
LOG_LOCAL="logs/ddp/${CHECKPOINT_NAME}_rank0.log"
LOG_REMOTE="logs/ddp/${CHECKPOINT_NAME}_rank1.log"

echo "==========================================="
echo "  DDP launch"
echo "  checkpoint: $CHECKPOINT_NAME"
echo "  master:     $MASTER_HOST:$MASTER_PORT"
echo "  rank0:      localhost  -> $LOG_LOCAL"
echo "  rank1:      $REMOTE_HOST -> $LOG_REMOTE"
echo "  args:       $*"
echo "==========================================="

# Source HF token if present (for any HF reads)
if [[ -f /tmp/hf_env.sh ]]; then source /tmp/hf_env.sh; fi

# Build the torchrun command line shared by both ranks
TR_ARGS=(
    --nnodes=2
    --nproc_per_node=1
    --master_addr="$MASTER_HOST"
    --master_port="$MASTER_PORT"
    train_vqvae.py
    --checkpoint-name="$CHECKPOINT_NAME"
    "$@"
)

# Activate conda for the local rank and launch in background
(
    source /home/wilkie/miniconda3/etc/profile.d/conda.sh
    conda activate nubun
    PYTHONUNBUFFERED=1 GLOO_SOCKET_IFNAME="$GLOO_IFACE" torchrun --node_rank=0 "${TR_ARGS[@]}"
) > "$LOG_LOCAL" 2>&1 &
LOCAL_PID=$!
echo "[localhost] torchrun PID $LOCAL_PID"

# Launch the remote rank over SSH. We need to source conda explicitly because
# non-interactive SSH doesn't run .bashrc; use the same absolute hook path.
ssh -o BatchMode=yes "$REMOTE_HOST" "
    set -e
    source /home/wilkie/miniconda3/etc/profile.d/conda.sh
    conda activate nubun
    cd $REMOTE_DIR
    PYTHONUNBUFFERED=1 GLOO_SOCKET_IFNAME=${GLOO_IFACE} torchrun --node_rank=1 ${TR_ARGS[*]}
" > "$LOG_REMOTE" 2>&1 &
REMOTE_PID=$!
echo "[$REMOTE_HOST] ssh PID $REMOTE_PID"

echo ""
echo "Both ranks launched. Tailing rank0 log; rank1 also at $LOG_REMOTE."
echo "(If either rank dies, the other will eventually time out at the rendezvous.)"
echo ""

# Wait for both
wait $LOCAL_PID
LOCAL_RC=$?
wait $REMOTE_PID
REMOTE_RC=$?

echo ""
echo "==========================================="
echo "rank0 exit: $LOCAL_RC"
echo "rank1 exit: $REMOTE_RC"
echo "==========================================="
exit $(( LOCAL_RC | REMOTE_RC ))
