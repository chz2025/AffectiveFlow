#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${1:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ "${GPU_IDS}" == "all" ]]; then
  unset CUDA_VISIBLE_DEVICES
  NUM_GPUS="$(nvidia-smi -L | wc -l)"
else
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  NUM_GPUS="$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)"
fi

echo "======================================================="
echo "AFPO Training Launcher"
echo "  Root:       ${ROOT_DIR}"
echo "  Devices:    ${GPU_IDS}"
echo "  Num GPUs:   ${NUM_GPUS}"
echo "  Config:     ${ROOT_DIR}/configs/train_emoflow.yaml"
echo "======================================================="

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  accelerate launch \
    --multi_gpu \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision fp16 \
    --num_machines 1 \
    --dynamo_backend no \
    "${ROOT_DIR}/scripts/train_afpo.py"
else
  python3 "${ROOT_DIR}/scripts/train_afpo.py"
fi

echo "======================================================="
echo "Training Finished"
echo "======================================================="
