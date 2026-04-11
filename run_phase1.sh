#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Phase 1: Synthetic Logographic Radicals"
echo "=========================================="

echo ""
echo "Step 1/3: Building concept inventory..."
python build_concept_inventory.py "$@"

echo ""
echo "Step 2/3: Discovering radicals..."
python discover_radicals.py

echo ""
echo "Step 3/3: Analyzing results..."
python analyze_radicals.py

echo ""
echo "=========================================="
echo "Phase 1 complete! Check results/ for output."
echo "=========================================="
