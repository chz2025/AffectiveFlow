#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CFG="configs/train_emoflow.yaml"
EVAL_DATA=""
OUT_DIR="output/paper_eval"
GPU_ID="${GPU_ID:-0}"
LOG_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cfg)
            CFG="$2"
            shift 2
            ;;
        --eval-data)
            EVAL_DATA="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${EVAL_DATA}" ]]; then
    echo "--eval-data is required" >&2
    exit 2
fi

mkdir -p logs "${OUT_DIR}"
RUN_ID="eval_all_$(date +%Y%m%d_%H%M%S)"
if [[ -z "${LOG_FILE}" ]]; then
    LOG_FILE="logs/${RUN_ID}.log"
fi
mkdir -p "$(dirname "${LOG_FILE}")"
PID_FILE="${LOG_FILE%.log}.pid"

RUNS=(
  "Qwen3.5-9B|${QWEN_CHECKPOINT:-output/afpo_cls/qwen/best}|${QWEN_MODEL:-Qwen/Qwen3.5-9B}"
  "Llama-3.1-8B|${LLAMA_CHECKPOINT:-output/afpo_cls/llama/best}|${LLAMA_MODEL:-meta-llama/Llama-3.1-8B}"
)

run_all_evals() {
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    METRIC_FILES=()

    for entry in "${RUNS[@]}"; do
        IFS='|' read -r label ckpt model_name <<< "${entry}"
        echo "============================================================"
        echo "Evaluating: ${label}"
        echo "  checkpoint : ${ckpt}"
        echo "  model_name : ${model_name}"
        echo "============================================================"
        out_metrics="${OUT_DIR}/${label}_metrics.json"
        python3 scripts/eval_auto.py \
            --eval-data "${EVAL_DATA}" \
            --checkpoint "${ckpt}" \
            --cfg "${CFG}" \
            --model-name "${model_name}" \
            --out-preds "${OUT_DIR}/${label}_preds.jsonl" \
            --out-metrics "${out_metrics}" \
            "${EXTRA_ARGS[@]}"
        METRIC_FILES+=("${label}|${out_metrics}")
    done

    echo
    echo "============================================================"
    echo "Combined metrics"
    echo "============================================================"
    python3 - "${METRIC_FILES[@]}" <<'PYEOF'
import json
import sys

entries = []
for arg in sys.argv[1:]:
    label, path = arg.split("|", 1)
    with open(path) as f:
        entries.append((label, json.load(f)))

def flatten(metrics):
    flat = dict(metrics.get("overall", {}))
    for stage, stage_metrics in (metrics.get("by_stage") or {}).items():
        for k, v in stage_metrics.items():
            flat[f"{stage}/{k}"] = v
    return flat

flat_entries = [(label, flatten(metrics)) for label, metrics in entries]

keys = []
for _, flat in flat_entries:
    for k in flat:
        if k not in keys:
            keys.append(k)

header = ["metric"] + [label for label, _ in flat_entries]
rows = [header]
for k in keys:
    row = [k]
    for _, flat in flat_entries:
        v = flat.get(k)
        if v is None:
            row.append("-")
        elif isinstance(v, float):
            row.append(f"{v:.4f}")
        else:
            row.append(str(v))
    rows.append(row)

widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
for row in rows:
    print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
PYEOF
    echo
    echo "Done."
}

echo "============================================================"
echo "AFPO - Background Evaluation"
echo "============================================================"
echo "Project : ${PROJECT_ROOT}"
echo "GPU     : ${GPU_ID}"
echo "Cfg     : ${CFG}"
echo "Log     : ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "============================================================"

run_all_evals > "${LOG_FILE}" 2>&1 &
PID="$!"
disown
echo "${PID}" > "${PID_FILE}"

echo "Started evaluation in background. PID=${PID}"
echo "View log:"
echo "  tail -f ${LOG_FILE}"
echo "Check process:"
echo "  ps -p ${PID} -f"
echo "Stop process:"
echo "  kill ${PID}"
