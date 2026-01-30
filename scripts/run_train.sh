#!/bin/bash

# =========================================================
# AFPO Training Launcher (Single GPU & Multi-GPU Support)
# =========================================================

# --- 1. 默认配置 (可根据你的环境修改) ---
# 数据路径
DATA_PATH="data/processed/extes/Ex_Tree_paths.json"

# 模型名称 (本地路径或 HF Hub 名称)
MODEL_NAME="/home/zch/model-hub/huggingface/Qwen/Qwen2.5-7B-Instruct"

# 输出目录
OUTPUT_DIR="output/afpo_cls"

# 训练超参
EPOCHS=3
BATCH_SIZE=2
LR=2e-5
LIMIT=0  # 0 表示不限制，用于调试时设为小数字(如100)

# --- 2. 模式选择 ---
# 运行方式:
#   bash run_train.sh [gpu_ids]
# 示例:
#   bash run_train.sh 0        (单卡，使用 GPU 0)
#   bash run_train.sh 0,1      (多卡，使用 GPU 0 和 1)
#   bash run_train.sh all      (多卡，使用所有可用 GPU)

# 获取命令行参数，默认为 "0" (单卡)
GPU_IDS=${1:-"0"}

# --- 3. 环境设置 ---
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export OMP_NUM_THREADS=4

# 获取 GPU 数量
if [ "$GPU_IDS" == "all" ]; then
    # 计算 nvidia-smi 列出的行数来估算 GPU 数量
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    export CUDA_VISIBLE_DEVICES="" # all 模式下不限制可见性
else
    # 计算逗号分隔的 ID 数量
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
fi

echo "======================================================="
echo "🚀 Starting AFPO Training"
echo "   Mode:       $(if [ $NUM_GPUS -gt 1 ]; then echo "Multi-GPU ($NUM_GPUS)"; else echo "Single-GPU"; fi)"
echo "   Devices:    ${GPU_IDS:-'All Available'}"
echo "   Model:      $MODEL_NAME"
echo "   Data:       $DATA_PATH"
echo "======================================================="

# --- 4. 构建 Python 命令参数 ---
PY_ARGS=" \
    --data \"$DATA_PATH\" \
    --epochs $EPOCHS \
    --limit $LIMIT \
"

# --- 5. 启动命令 (自动判断使用 python 还是 accelerate launch) ---

if [ $NUM_GPUS -gt 1 ]; then
    # === 多卡模式 (DDP) ===
    # 使用 accelerate launch 启动
    # --multi_gpu: 启用多卡
    # --num_processes: GPU 数量
    # --mixed_precision: fp16 (根据你的显卡支持，也可以改 bf16)
    
    echo "Running with accelerate launch..."
    accelerate launch \
        --multi_gpu \
        --num_processes $NUM_GPUS \
        --mixed_precision fp16 \
        --num_machines 1 \
        --dynamo_backend no \
        train_afpo.py $PY_ARGS

else
    # === 单卡模式 ===
    # 直接运行 python 脚本 (代码内部的 Accelerator 会自动处理单卡)
    
    echo "Running with python..."
    python train_afpo.py $PY_ARGS
fi

echo "======================================================="
echo "✅ Training Script Finished."
echo "======================================================="