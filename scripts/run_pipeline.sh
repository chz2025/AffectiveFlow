#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running the pipeline}"

echo "Step 1/4: Generating trees (build_ex_tree.py)"
python3 "$ROOT_DIR/scripts/build_ex_tree.py"

echo "Step 2/4: Counting/validating trees (analyze/count_trees.py)"
python3 "$ROOT_DIR/analyze/count_trees.py"

echo "Step 3/4: Drawing trees (analyze/draw_tree.py)"
python3 "$ROOT_DIR/analyze/draw_tree.py"

echo "Step 4/4: Extracting paths (scripts/extract_paths.py)"
python3 "$ROOT_DIR/scripts/extract_paths.py"

echo "Done. Outputs are recorded in analyze/tree_paths.json and the corresponding output directories."
