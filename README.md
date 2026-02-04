# Affective Flow (AFPO) — Quick Start & Project Layout

Affective Flow (AFPO) implements Action Flow + Direct Preference Optimization (DPO) with value estimation for training strategy-selection models in emotional-support dialogues.

---

## Project Structure ✅

Top-level layout:

```
Emo_Flow_DPO/
├── scripts/                # Training, tree-generation and pipeline scripts
│   ├── train_AFPO.py       # AFPO / AFDPO training entry
│   ├── build_ex_tree.py    # Generate MCTS trees (LLM; requires OPENAI API if used)
│   ├── extract_paths.py    # Extract root-to-leaf trajectories for training
│   ├── run_pipeline.sh     # End-to-end generation + processing pipeline
│   └── run_train.sh        # Training wrapper
├── analyze/                # Tree & trajectory analysis utilities
│   ├── count_trees.py
│   ├── draw_tree.py
│   └── tree_paths.json
├── configs/                # YAML config templates
│   └── train_emoflow.yaml
├── data/                   # Raw and processed data (JSON / JSONL)
├── assets/                 # Images and buttons
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Requirements & Setup 📦

- Python 3.10+
- PyTorch 2.0+ (choose the appropriate CUDA wheel or CPU-only)
- Optional: `accelerate` for multi-GPU training

Install example:

```bash
# Create & activate environment (conda recommended)
conda create -n emoflow python=3.10 -y
conda activate emoflow

# Install PyTorch (see https://pytorch.org for appropriate wheel)
pip install torch torchvision torchaudio

# Install project deps
pip install -r requirements.txt
```

Optional: install `graphviz` for visualization (macOS):

```bash
brew install graphviz
```

---

## Quick Start 🚀

Follow these four steps: 1) Install environment, 2) Download base model, 3) Generate MCTS + extract trajectories, 4) Start training (AFDPO / Flow-Balance).

### 1) Install environment

- Create a Python 3.10 environment and install dependencies as listed above.
- If you plan multi-GPU training, install and configure `accelerate`:

```bash
pip install accelerate
accelerate config
```

---

### 2) Download / prepare base model

- Set `model_name` in `configs/train_emoflow.yaml` (example: `meta-llama/Llama-2-7b-hf` or a local path).
- If using Hugging Face model hub, authenticate or download the model locally:

```bash
huggingface-cli login
# or rely on `transformers` to fetch at runtime
```

Notes: training supports freezing the base model and using a small "doctor" model for alignment, or parameter-efficient LoRA updates.

---

### 3) Generate MCTS trees and extract trajectories (MCTS)

To generate exploration trees (optional) and extract training trajectories:

```bash
# If using OpenAI for generation
export OPENAI_API_KEY="your-openai-api-key"

# Generate MCTS trees
python scripts/build_ex_tree.py

# Validate generated trees
python analyze/count_trees.py

# Extract root-to-leaf trajectories (JSONL)
python scripts/extract_paths.py
```

Outputs are saved under `data/processed/` for model training.

---

### 4) Start training (AFDPO / Flow-Balance) 🔥

Run training with:

```bash
# Single GPU
python scripts/train_AFPO.py --config configs/train_emoflow.yaml

# Multi-GPU (accelerate)
accelerate launch --multi_gpu scripts/train_AFPO.py --config configs/train_emoflow.yaml
```

Training details:
- The default objective is AF + DPO (AFDPO / Flow-Balance), combining Action Flow loss, a DPO-style value margin, and KL regularization.
- Enable LoRA (`use_lora: true`) in config for memory-efficient fine-tuning.

---

## Configuration Example (Configuration) 🛠️

Key fields in `configs/train_emoflow.yaml`:

```yaml
afpo_training:
  # model & data
  model_name: "meta-llama/Llama-2-7b-hf"
  max_length: 512
  batch_size: 8
  tree_path_train: "data/processed/train.jsonl"
  tree_path_val: "data/processed/valid.jsonl"

  # optimization
  lr: 1e-4
  epochs: 3
  gradient_accumulation_steps: 1

  # LoRA (optional)
  use_lora: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05

  # loss / flow control
  beta: 0.1           # KL regularization weight
  gamma: 1.0          # DPO value margin
  flow_balance_coef: 1.0  # Action Flow loss weight (tune experimentally)

  # evaluation & checkpoints
  eval_enabled: true
  eval_every_epochs: 1
  save_best: true
  best_metric: "val_loss"

  # resume training
  resume_from_checkpoint: null
```

Notes:
- `flow_balance_coef` controls the Action Flow loss contribution — tune to balance stability and preference preservation.

---

If you want a bilingual README, more examples, or a short demo section, tell me which parts to add or rephrase. ✨
