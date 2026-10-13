#!/usr/bin/env bash
# mem_watchdog.sh — kill a job before the GB10's unified memory runs out.
#
# The GB10 shares ~121 GB between CPU and GPU; past ~110 GB it pages to disk
# and locks up. This polls MemAvailable and SIGTERMs the given PID (and its
# children) when it falls below the floor.
#
# Usage: bash scripts/mem_watchdog.sh <pid> [floor_gb=20] [interval_s=2]
pid=$1; floor=${2:-20}; every=${3:-2}
while kill -0 "$pid" 2>/dev/null; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    if [ "$avail" -lt "$floor" ]; then
        echo "mem_watchdog: MemAvailable ${avail} GB < ${floor} GB, killing $pid" >&2
        pkill -TERM -P "$pid"; kill -TERM "$pid"
        exit 1
    fi
    sleep "$every"
done
