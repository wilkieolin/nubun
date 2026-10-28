#!/usr/bin/env bash
# mem_watchdog.sh — kill a job before the GB10's unified memory runs out.
#
# The GB10 shares ~121 GB between CPU and GPU; past ~110 GB it pages to disk
# and locks up. This polls /proc/meminfo and SIGTERMs the given PID (and its
# children) when the chosen metric falls below the floor.
#
# Usage: bash scripts/mem_watchdog.sh <pid> [floor_gb=20] [interval_s=2]
#   floor_gb 0 disables the kill (the loop only logs).
# Env:   WATCHDOG_METRIC=available (default: the kernel's MemAvailable estimate)
#                       | free_cached (MemFree + Cached + Buffers; use this when
#                         MemAvailable is implausibly low, e.g. the GPU driver
#                         pins memory the kernel then excludes from its estimate)
#        WATCHDOG_LOG=1 prints the reading every poll instead of only on kill.
pid=$1; floor=${2:-20}; every=${3:-2}
metric=${WATCHDOG_METRIC:-available}
read_gb() {
    awk -v m="$metric" '
        /^MemTotal:/     {t=$2}  /^MemFree:/   {f=$2}  /^MemAvailable:/ {a=$2}
        /^Buffers:/      {b=$2}  /^Cached:/    {c=$2}
        END { v = (m == "free_cached") ? f + c + b : a;
              printf "%d %d\n", int(v/1048576), int(t/1048576) }' /proc/meminfo
}
while kill -0 "$pid" 2>/dev/null; do
    read -r avail total <<<"$(read_gb)"
    [[ "${WATCHDOG_LOG:-0}" == 1 ]] && echo "mem_watchdog: $metric ${avail} GB of ${total} GB" >&2
    if [[ "$floor" != 0 && -n "$avail" && "$avail" -lt "$floor" ]]; then
        echo "mem_watchdog: $metric ${avail} GB of ${total} GB < floor ${floor} GB, killing $pid" >&2
        grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|Shmem|HugePages_Total)' /proc/meminfo >&2
        pkill -TERM -P "$pid"; kill -TERM "$pid"
        exit 1
    fi
    sleep "$every"
done
