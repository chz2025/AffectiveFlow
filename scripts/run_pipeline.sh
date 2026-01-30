
set -x
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- API key configuration ---
# ⚠️  IMPORTANT: Set your OpenAI API key before running
# Option 1: Set environment variable before running this script
#   export OPENAI_API_KEY="sk-..."
# Option 2: Uncomment and set directly below (NOT RECOMMENDED for security)
# Current value (must not be empty or placeholder):
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-openai_key_here}"
echo "OpenAI API Key configured: ${OPENAI_API_KEY:0:10}..."

echo ""
echo "========================================="
echo "Step 1/4: Generating MCTS trees"
echo "========================================="
echo "Explores action space using LLM-based strategy scoring."
echo "Input: data/raw/{exconv,extes}/"
echo "Output: data/processed/extes/Ex_Tree*.jsonl"
python3 "$ROOT_DIR/scripts/build_ex_tree.py"

echo ""
echo "========================================="
echo "Step 2/4: Validating and analyzing trees"
echo "========================================="
echo "Computes tree statistics: size, depth, path count."
echo "Output: analyze/tree_paths.json"
python3 "$ROOT_DIR/analyze/count_trees.py"

echo ""
echo "========================================="
echo "Step 3/4: Visualizing trees (optional)"
echo "========================================="
echo "Generates tree visualizations for inspection."
python3 "$ROOT_DIR/analyze/draw_tree.py"

echo ""
echo "========================================="
echo "Step 4/4: Extracting training trajectories"
echo "========================================="
echo "Extracts root-to-leaf paths for AFPO training."
echo "Output: data/processed/extes/*_paths.jsonl"
python3 "$ROOT_DIR/scripts/extract_paths.py"

echo ""
echo "========================================="
echo "✓ Pipeline Complete"
echo "========================================="
echo "Generated trajectories are ready for training."
echo "Next step: python scripts/train_AFPO.py"
echo ""

