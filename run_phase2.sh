#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Phase 2: Refined Concepts + Radical Hierarchy"
echo "=========================================="

echo ""
echo "Step 1/4: Building improved concept inventory..."
python build_concept_inventory.py "$@"

echo ""
echo "Step 2/4: Discovering radicals (k-means + OMP, K=500)..."
python discover_radicals.py --k-values 500 --methods kmeans_omp

echo ""
echo "Step 3/4: Building radical hierarchy..."
python build_radical_hierarchy.py

echo ""
echo "Step 4/4: Analyzing results..."
python analyze_radicals.py

echo ""
echo "=========================================="
echo "Phase 2 complete! Check results/ for output."
echo "=========================================="
