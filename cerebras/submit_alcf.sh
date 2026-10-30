#!/usr/bin/env bash
# Submit a Nubun VQ-VAE run to the ALCF Cerebras CS-3.
#
# Run FROM AN ALCF USER NODE (cer-usn-01 / cer-usn-02), inside tmux/screen —
# a real run streams to stdout for hours and dies if your ssh drops.
#
#   source ~/R_2.10.0/venv_cerebras_pt/bin/activate   # (the script does this)
#   bash cerebras/submit_alcf.sh minprobe-attn2  # cluster health canary (execute)
#   bash cerebras/submit_alcf.sh smoke           # the real M1 graph, synthetic data, execute
#   bash cerebras/submit_alcf.sh smoke-m2        # the real M2 graph (semantic head on)
#   bash cerebras/submit_alcf.sh m1              # recon + RVQ, REAL opus data
#   bash cerebras/submit_alcf.sh m2              # full recipe
#
# Every run gets its own log (logs/<mode>-<stamp>.log), client artifact dir
# (cerebras_logs/<mode>-<stamp>/: run_meta.json, cirh.mlir, track.json) and
# model dir (model_dir/<mode>-<stamp>/). On exit the wsjob ids are printed with
# the export command; on FAILURE the logs are exported automatically.
#
# Monitor from a SECOND console:  csctl get jobs -a | grep name=nubun
# Cancel:                         csctl cancel job <wsjob-id>
set -euo pipefail

# --- ALCF environment (R_2.10.0 == our cerebras-pytorch 2.10.0) --------------
VENV="${VENV:-$HOME/R_2.10.0/venv_cerebras_pt/bin/activate}"
if [[ -f "$VENV" ]]; then
    # shellcheck disable=SC1090
    source "$VENV"
else
    echo "WARN: venv not found at $VENV — assuming cerebras env is already active." >&2
fi
# The ALCF proxy (set in ~/.bashrc for pip/git) must NOT apply to the appliance
# client: gRPC C-core honours http(s)_proxy, and the cluster endpoints
# (10.125.8.2:443, *.cerebras.internal) are internal. We both exempt them and
# run python with the proxy variables unset.
export no_proxy="10.125.8.2,.cerebras.internal,localhost,127.0.0.1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
NOPROXY_ENV=(env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy)

# --- paths the appliance workers must see -------------------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO/data}"   # staged data (not in git; data/ is ignored)
PYTHON_PATHS="${PYTHON_PATHS:-$REPO}"
NUM_CSX="${NUM_CSX:-1}"

mode="${1:-smoke}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="${mode}-${STAMP}"
LOG="$REPO/logs/$RUN.log"
ART="$REPO/cerebras_logs/$RUN"
OUT="$REPO/model_dir/$RUN"
mkdir -p "$REPO/logs"
cd "$REPO"

# --- post-run: surface the wsjob ids + export logs on failure -----------------
on_exit() {
    local rc=$1
    [[ -n "$PY_RC" ]] && rc=$PY_RC          # python's own status, not the trap's
    echo
    echo "=== run $RUN finished rc=$rc  log=$LOG"
    local jobs
    jobs="$(grep -oE 'wsjob-[a-z0-9]+' "$LOG" 2>/dev/null | sort -u || true)"
    if [[ -z "$jobs" && -f "$ART/run_meta.json" ]]; then
        jobs="$(grep -oE 'wsjob-[a-z0-9]+' "$ART/run_meta.json" | sort -u || true)"
    fi
    if [[ -n "$jobs" ]]; then
        echo "=== wsjobs (export within the hour; records expire in ~1 week):"
        for j in $jobs; do echo "    bash cerebras/export_logs.sh $j"; done
        if [[ $rc -ne 0 ]]; then
            echo "=== rc!=0 → exporting logs now"
            for j in $jobs; do
                bash "$REPO/cerebras/export_logs.sh" "$j" || echo "export of $j failed" >&2
            done
        fi
    fi
}
trap 'on_exit $?' EXIT

need_data() {   # modes that read the staged data
    for f in embedding_table.pt token_weights.pt parallel_corpus.npz; do
        [[ -e "$DATA_DIR/$f" ]] || { echo "ERROR: $DATA_DIR/$f missing" >&2; exit 2; }
    done
    # Workers memory-map UNPACKED shards (see cerebras/unpack_opus.sh); build once.
    # Once unpacked, the packed $DATA_DIR/opus100 can be deleted to save quota.
    if [[ ! -d "$DATA_DIR/opus100_npy/zh_en" ]]; then
        [[ -d "$DATA_DIR/opus100" ]] || { echo "ERROR: neither $DATA_DIR/opus100_npy nor $DATA_DIR/opus100" >&2; exit 2; }
        echo "=== unpacking opus shards into $DATA_DIR/opus100_npy (one-time)"
        bash "$REPO/cerebras/unpack_opus.sh" "$DATA_DIR/opus100" "$DATA_DIR/opus100_npy"
    fi
}
check_mounts() {   # fail fast; a nonexistent mount dir only fails inside the execute job
    local IFS=,; for d in $1; do
        [[ -d "$d" ]] || { echo "ERROR: mount dir does not exist: $d" >&2; exit 2; }
    done
}

PY_RC=""
run() {   # unbuffered python, proxy-free env, tee to the per-run log
    echo "=== $RUN: ${NOPROXY_ENV[*]} python -u $*" | tee "$LOG"
    set +e
    "${NOPROXY_ENV[@]}" python -u "$@" 2>&1 | tee -a "$LOG"
    PY_RC=${PIPESTATUS[0]}
    set -e
    echo "=== python exit status: $PY_RC" | tee -a "$LOG"
    return "$PY_RC"
}

probe=(cerebras/minimal_probe.py --artifact-dir "$ART")
train=(cerebras/train_cstorch.py --artifact-dir "$ART" --out-dir "$OUT"
       --num-csx "$NUM_CSX" --python-paths "$PYTHON_PATHS")
# The tied token table TRAINS by default (Oct 9: frozen costs ~1.5 nats of recon
# at 2k steps — CS3_BRINGUP §7 5b). FROZEN_EMB=1 restores the frozen graph.
[[ "${FROZEN_EMB:-0}" == 1 ]] || train+=(--train-embeddings)
data=(--embedding-table "$DATA_DIR/embedding_table.pt"
      --token-weights "$DATA_DIR/token_weights.pt"
      --opus-dir "$DATA_DIR/opus100_npy"
      --parallel-corpus "$DATA_DIR/parallel_corpus.npz"
      --sem-dir "$DATA_DIR/opus100_npy")
# M1 graph constants. --steps/--warmup-steps/--lr/--lr-min-ratio feed the
# TRACED LR schedulers, so they are part of the compile hash: smoke, compile
# and m1 must use identical values to share one ~17-minute compile.
M1_STEPS="${M1_STEPS:-2000}"
M2_STEPS="${M2_STEPS:-100000}"
WARMUP="${WARMUP:-500}"

case "$mode" in
  minprobe)        run "${probe[@]}" ;;                       # canonical MLP
  minprobe-raw)    run "${probe[@]}" --raw-input ;;           # raw-generator input
  minprobe-adamw)  run "${probe[@]}" --opt adamw ;;           # AdamW + warmup
  minprobe-attn)   run "${probe[@]}" --model attn ;;          # torch nn.Transformer (fails)
  minprobe-attn2)  run "${probe[@]}" --model attn2 "${@:2}" ;; # hand-rolled attention: KNOWN-GOOD canary
  minprobe-cehead) run "${probe[@]}" --model cehead "${@:2}" ;; # output head + full-vocab CE
  minprobe-vqvae)  run "${probe[@]}" --model vqvae "${@:2}" ;;  # our model, small synthetic
  smoke)    # the real M1 graph, synthetic data, EXECUTE mode (10 steps). No data needed.
    MOUNT_DIRS="${MOUNT_DIRS:-$REPO}"; check_mounts "$MOUNT_DIRS"
    run "${train[@]}" --no-semantic --synthetic-data --steps "$M1_STEPS" \
      --warmup-steps "$WARMUP" --max-steps 10 --ckpt-every 10 --log-every 1 \
      --job-label name=nubun-smoke --job-time-sec 7200 --mount-dirs "$MOUNT_DIRS" "${@:2}" ;;
  smoke-emb) # == smoke now that the table trains by default (kept for old notes)
    MOUNT_DIRS="${MOUNT_DIRS:-$REPO}"; check_mounts "$MOUNT_DIRS"
    run "${train[@]}" --no-semantic --synthetic-data --steps "$M1_STEPS" \
      --warmup-steps "$WARMUP" --max-steps 10 --ckpt-every 10 --log-every 1 \
      --job-label name=nubun-smoke-emb --job-time-sec 7200 --mount-dirs "$MOUNT_DIRS" "${@:2}" ;;
  bench)    # throughput at BATCH (default 256): M2 graph, synthetic, BENCH_STEPS steps.
            # Read samples/s from cerebras_logs/<run>/executors/000001/performance.json.
    MOUNT_DIRS="${MOUNT_DIRS:-$REPO}"; check_mounts "$MOUNT_DIRS"
    run "${train[@]}" --synthetic-data --batch-size "${BATCH:-256}" --steps "$M2_STEPS" \
      --warmup-steps "$WARMUP" --max-steps "${BENCH_STEPS:-300}" --ckpt-every 1000000 --log-every 50 \
      --job-label "name=nubun-bench-b${BATCH:-256}" --job-time-sec 7200 --mount-dirs "$MOUNT_DIRS" "${@:2}" ;;
  smoke-m2) # the real M2 graph (semantic head + loss), synthetic data, EXECUTE mode.
    MOUNT_DIRS="${MOUNT_DIRS:-$REPO}"; check_mounts "$MOUNT_DIRS"
    run "${train[@]}" --synthetic-data --steps "$M2_STEPS" \
      --warmup-steps "$WARMUP" --max-steps 10 --ckpt-every 10 --log-every 1 \
      --job-label name=nubun-smoke-m2 --job-time-sec 7200 --mount-dirs "$MOUNT_DIRS" "${@:2}" ;;
  compile)  # the M1 graph with REAL data, compile-only (same graph as smoke/m1)
    need_data; MOUNT_DIRS="${MOUNT_DIRS:-$REPO,$DATA_DIR}"; check_mounts "$MOUNT_DIRS"
    run "${train[@]}" "${data[@]}" --no-semantic --steps "$M1_STEPS" \
      --warmup-steps "$WARMUP" --compile-only \
      --job-label name=nubun-compile --job-time-sec 7200 --mount-dirs "$MOUNT_DIRS" "${@:2}" ;;
  m1)       # recon + RVQ only, REAL data — prove the model trains on the WSE
    need_data; MOUNT_DIRS="${MOUNT_DIRS:-$REPO,$DATA_DIR}"; check_mounts "$MOUNT_DIRS"
    run "${train[@]}" "${data[@]}" --no-semantic --steps "$M1_STEPS" \
      --warmup-steps "$WARMUP" --ckpt-every "${CKPT_EVERY:-500}" \
      --job-label name=nubun-m1 --job-time-sec "${JOB_TIME_SEC:-82800}" --mount-dirs "$MOUNT_DIRS" "${@:2}" ;;
  m2)       # full recipe (needs <lang>_en.sem.npz in --sem-dir)
    need_data; MOUNT_DIRS="${MOUNT_DIRS:-$REPO,$DATA_DIR}"; check_mounts "$MOUNT_DIRS"
    run "${train[@]}" "${data[@]}" --steps "$M2_STEPS" \
      --warmup-steps "$WARMUP" --ckpt-every "${CKPT_EVERY:-5000}" \
      --job-label name=nubun-m2 --job-time-sec "${JOB_TIME_SEC:-82800}" --mount-dirs "$MOUNT_DIRS" "${@:2}" ;;
  *)
    echo "usage: $0 {minprobe|minprobe-raw|minprobe-adamw|minprobe-attn|minprobe-attn2|minprobe-cehead|minprobe-vqvae|smoke|smoke-emb|smoke-m2|bench|compile|m1|m2} [extra args]" >&2
    exit 2 ;;
esac
