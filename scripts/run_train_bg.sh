#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
GPU_IDS=""
RUN_NAME="${RUN_NAME:-afpo_train}"
LOG_FILE=""
CONFIG_PATH="configs/train_emoflow.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/run_train_bg.sh [--gpu 0 | --gpus 0,1] [--run-name name] [--log logs/train.log] [--config configs/xxx.yaml]"
            exit 1
            ;;
    esac
done

cd "${PROJECT_ROOT}"

RUN_ID="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -z "${LOG_FILE}" ]]; then
    LOG_FILE="logs/${RUN_ID}.log"
fi
mkdir -p "$(dirname "${LOG_FILE}")"
PID_FILE="${LOG_FILE%.log}.pid"
INFO_FILE="${LOG_FILE%.log}.info.txt"

echo "============================================================"
echo "AFPO - Background Training"
echo "============================================================"
echo "Project : ${PROJECT_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "Run ID  : ${RUN_ID}"
echo "Config  : ${CONFIG_PATH}"
if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -ra _GPU_ARRAY <<< "${GPU_IDS}"
    NPROC="${#_GPU_ARRAY[@]}"
    echo "GPUs    : ${GPU_IDS}  (accelerate nproc=${NPROC})"
else
    NPROC=1
    echo "GPU     : ${GPU_ID}"
fi
echo "Log     : ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "============================================================"

{
    echo "run_name=${RUN_NAME}"
    echo "run_id=${RUN_ID}"
    echo "project=${PROJECT_ROOT}"
    echo "config=${CONFIG_PATH}"
    echo "log=${LOG_FILE}"
    echo "started_at=$(date -Is)"
    if [[ -n "${GPU_IDS}" ]]; then
        echo "gpus=${GPU_IDS}"
    else
        echo "gpu=${GPU_ID}"
    fi
} > "${INFO_FILE}"

if [[ -n "${GPU_IDS}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" AFPO_CONFIG="${CONFIG_PATH}" nohup accelerate launch \
        --multi_gpu \
        --num_processes "${NPROC}" \
        --mixed_precision fp16 \
        --num_machines 1 \
        --dynamo_backend no \
        scripts/train_afpo.py \
        > "${LOG_FILE}" 2>&1 &
else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" AFPO_CONFIG="${CONFIG_PATH}" nohup python3 scripts/train_afpo.py \
        > "${LOG_FILE}" 2>&1 &
fi

PID="$!"
disown
echo "${PID}" > "${PID_FILE}"
echo "pid=${PID}" >> "${INFO_FILE}"

echo "Started training in background. PID=${PID}"
echo "View log:"
echo "  tail -f ${LOG_FILE}"
echo "Check process:"
echo "  ps -p ${PID} -f"
echo "Stop process:"
echo "  kill ${PID}"
