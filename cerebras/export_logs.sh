#!/usr/bin/env bash
# Export a wsjob's cluster-side logs and print the first real error.
#   bash cerebras/export_logs.sh <wsjob-id> [extra csctl log-export flags]
# Lands in logx/<wsjob-id>/ (zip + unpacked). Run it within the hour of a
# failure — job records expire from the cluster in about a week, and the
# client only ever shows a terse 502 / DEADLINE_EXCEEDED.
set -euo pipefail
job="${1:?usage: export_logs.sh <wsjob-id> [flags]}"; shift || true
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$REPO/logx/$job"; mkdir -p "$dest"
csctl log-export "$job" -p "$dest" "$@"
for z in "$dest"/*.zip; do
  [[ -f "$z" ]] || continue
  unzip -oq "$z" -d "$dest" || echo "unzip returned $? for $z (continuing)"
  echo "unpacked $z"
done
echo "=== $job: coordinator stdout (FATAL / errors) ==="
find "$dest" -path "*coordinator-0*" -name "dbg_crd_0.out" -print0 2>/dev/null | xargs -0 -r \
  grep -niE "FATAL|error:|Failed to convert|no corresponding kernel|Unable to insert|no valid layouts|slice_filter|did not fit|Exception|Compilation completed|Initiating fresh compile|cache" \
  | grep -viE "0x0|libMLIR|libLLVM" | head -20 || true
echo "=== $job: coordinator stderr (assertions / signals) ==="
find "$dest" -path "*coordinator-0*" -name "dbg_crd_0.err" -print0 2>/dev/null | xargs -0 -r \
  grep -niE "Assertion|CEREBRAS_THROW|killed by signal|Traceback|Error" | grep -v '^\s*#' | head -20 || true
echo "=== $job: worker / chief / weight / activation errors ==="
find "$dest" \( -path "*worker*" -o -path "*chief*" -o -path "*weight*" -o -path "*activation*" \) -type f \
  \( -name "*.out" -o -name "*.err" -o -name "*.log" \) -print0 2>/dev/null | xargs -0 -r \
  grep -liE "Traceback|Error|FATAL|Killed|OOM|No such file|ModuleNotFoundError" 2>/dev/null | head -20 || true
echo "logs under: $dest"
