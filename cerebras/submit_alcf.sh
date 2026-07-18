#!/usr/bin/env bash
# Submit a Nubun VQ-VAE training run to the ALCF Cerebras CS-3.
#
# Run this FROM AN ALCF USER NODE (cer-usn-01 / cer-usn-02), inside tmux/screen
# — a real run streams to stdout for hours and dies if your ssh drops.
#
#   ssh <you>@cerebras.alcf.anl.gov && ssh cer-usn-01
#   tmux new -s nubun
#   git clone <this repo> ~/nubun && cd ~/nubun
#   # stage data/ (embedding_table.pt, token_weights.pt, opus100/*.npz) here
#   bash cerebras/submit_alcf.sh compile   # dry compile first (no wafer time)
#   bash cerebras/submit_alcf.sh m1        # milestone 1: recon + RVQ only
#   bash cerebras/submit_alcf.sh m2        # milestone 2: full recipe
#
# Monitor from a SECOND console:  csctl get jobs | grep name=nubun
# Cancel:                         csctl cancel job <wsjob-id>
set -euo pipefail

# --- ALCF environment (R_2.10.0 matches our cerebras-pytorch 2.10.0) ----------
VENV="${VENV:-$HOME/R_2.10.0/venv_cerebras_pt/bin/activate}"
if [[ -f "$VENV" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
else
    echo "WARN: venv not found at $VENV — assuming cerebras env is already active." >&2
fi
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

# --- paths the appliance workers must see -------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO/data}"          # override to a shared ALCF data dir
MOUNT_DIRS="${MOUNT_DIRS:-$REPO,$DATA_DIR}"
PYTHON_PATHS="${PYTHON_PATHS:-$REPO}"
NUM_CSX="${NUM_CSX:-1}"
JOB_TIME_SEC="${JOB_TIME_SEC:-82800}"        # 23h; ALCF hard cap is 24h

common=(--num-csx "$NUM_CSX" --job-time-sec "$JOB_TIME_SEC"
        --mount-dirs "$MOUNT_DIRS" --python-paths "$PYTHON_PATHS")

cd "$REPO"
mode="${1:-smoke}"
case "$mode" in
  minprobe) # canonical MLP standalone loop — does ANY raw cstorch loop compile
            # on this cluster, or only cszoo fit? No data, nothing of ours.
    python cerebras/minimal_probe.py |& tee minprobe.log ;;
  minprobe-raw) # same MLP, but raw-generator input (train_cstorch style).
            # If minprobe works but this fails empty -> input pipeline is it.
    python cerebras/minimal_probe.py --raw-input |& tee minprobe_raw.log ;;
  minprobe-adamw) # same MLP, but AdamW + warmup-from-0 cosine (train_cstorch).
            # Tests whether LR=0 on the compiled first step empties the graph.
    python cerebras/minimal_probe.py --opt adamw |& tee minprobe_adamw.log ;;
  minprobe-attn) # bare nn.TransformerEncoder — does torch's built-in
            # transformer/attention lower on cstorch at all?
    python cerebras/minimal_probe.py --model attn |& tee minprobe_attn.log ;;
  minprobe-vqvae) # OUR model (small synthetic) through the known-good loop.
            # Add --no-rvq / --no-tie after the target to bisect components.
    python cerebras/minimal_probe.py --model vqvae "${@:2}" |& tee minprobe_vqvae.log ;;
  smoke)    # EXECUTE mode, synthetic data, tiny — isolates compile of OUR graph
            # from data + from --compile-only. No data/ files needed. This is the
            # exact M1 graph; if it compiles+executes, the empty-CIRH issue was
            # the compile-only path, and m1 (real data) should follow.
    python cerebras/train_cstorch.py --no-semantic --synthetic-data --steps 10 \
      --job-label name=nubun-smoke "${common[@]}" |& tee smoke.log ;;
  compile)  # trace + compile only, no wafer execution
    python cerebras/train_cstorch.py --no-semantic --steps 2000 --compile-only \
      --job-label name=nubun-compile "${common[@]}" |& tee compile.log ;;
  m1)       # recon + RVQ losses only — prove the model trains on the WSE
    python cerebras/train_cstorch.py --no-semantic --steps 2000 \
      --job-label name=nubun-m1 "${common[@]}" |& tee m1.log ;;
  m2)       # full recipe (precompute semantic targets first — see PORT_CS3 §7)
    python cerebras/train_cstorch.py --steps 100000 \
      --job-label name=nubun-m2 "${common[@]}" |& tee m2.log ;;
  *)
    echo "usage: $0 {minprobe|minprobe-raw|minprobe-adamw|minprobe-attn|minprobe-vqvae|smoke|compile|m1|m2}" >&2; exit 2 ;;
esac
