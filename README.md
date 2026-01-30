<p align="center">
  <img src="assets/logo.svg" width="220" alt="Affective Flow logo" />
</p>

# Affective Flow Language Model for Emotional Support Conversation

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org) [![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)](https://pytorch.org)

<p align="center">
  <a href="#paper"><img src="assets/buttons/paper.svg" alt="Paper" /></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#quick-start"><img src="assets/buttons/quickstart.svg" alt="Quick Start" /></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#methodology"><img src="assets/buttons/methodology.svg" alt="Methodology" /></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#results"><img src="assets/buttons/results.svg" alt="Results" /></a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#citation"><img src="assets/buttons/citation.svg" alt="Citation" /></a>
</p>

_Align frozen LLMs at test-time with a small doctor model — no fine-tuning required!_

This repository contains the implementation of the **Action Flow and Preference Optimization (AFPO)** approach for dialogue action selection in emotional support conversations. This work was submitted as supplementary material for a conference paper.

## Overview

**Emo_Flow_DPO** combines three key ideas to train dialogue strategies:

1. **Action Flow (AF) Loss**: Enforces cumulative probability constraints across decision steps in a dialogue trajectory, ensuring consistent preference ordering.
2. **Direct Preference Optimization (DPO)**: Uses margin-based learning to distinguish preferred actions from suboptimal alternatives without explicit reward models.
3. **Value Estimation**: Incorporates contextual value estimates to guide strategy selection.

The system is designed for training language model-based dialogue agents to select supportive strategies in emotional support conversations.

## Project Structure

```
Emo_Flow_DPO/
├── scripts/                      # Training and data processing scripts
│   ├── train_AFPO.py            # Main AFPO classifier training pipeline
│   ├── build_ex_tree.py          # Generate MCTS exploration trees (requires OpenAI API)
│   ├── extract_paths.py          # Extract root-to-leaf trajectories from trees
│   ├── run_pipeline.sh           # End-to-end pipeline orchestration
│   └── run_train.sh              # Training script wrapper
├── analyze/                      # Utility scripts for tree analysis
│   ├── count_trees.py            # Tree statistics and validation
│   ├── draw_tree.py              # Visualization utilities
│   └── tree_paths.json           # Metadata for generated trees
├── configs/
│   └── train_emoflow.yaml        # Training hyperparameters and configuration
├── data/
│   ├── raw/                      # Original dataset sources
│   │   ├── exconv/ExConv.json    # ExConv dialogue dataset
│   │   └── extes/ExTES.json      # ExTES emotional support dataset
│   ├── processed/extes/          # Processed trajectories
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   ├── strategies.json           # Support strategy definitions
│   ├── prompt.json               # LLM prompts for tree generation
│   └── evaluation_metrics.json   # Evaluation criterion definitions
├── requirements.txt              # Python package dependencies
└── README.md                     # This file
```

## Installation

### Step 1: Install PyTorch
First, install PyTorch based on your hardware configuration:
```bash
# CUDA 12.1 example
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only
pip install torch torchvision torchaudio

# Visit https://pytorch.org/get-started/locally/ for other configurations
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 (Optional): Install Graphviz
For tree visualization in `analyze/draw_tree.py`, install system graphviz:
```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Others: https://graphviz.org/download/
```

## Quick Start

### 1. Data Preparation

Ensure your raw dialogue data is in:
- `data/raw/exconv/` for ExConv dataset
- `data/raw/extes/` for ExTES dataset

Run preprocessing:
```bash
python scripts/extract_paths.py  # Extract trajectories from existing trees
```

### 2. Training Configuration

Edit `configs/train_emoflow.yaml` to set:
- **Model parameters**: `model_name`, `max_length`, `batch_size`
- **Training hyperparameters**: `lr`, `epochs`, `gradient_accumulation_steps`
- **LoRA settings**: `use_lora`, `lora_rank`, `lora_alpha`, `lora_dropout`
- **Loss coefficients**: `beta` (KL regularization), `gamma` (value margin)
- **Data paths**: `tree_path_train`, `tree_path_val`, `tree_path_test`

### 3. Train the AFPO Classifier

```bash
# Single GPU
python scripts/train_AFPO.py

# Multi-GPU with Accelerate
accelerate launch --multi_gpu scripts/train_AFPO.py
```

### 4. Generate Trees and Extract Trajectories (Optional)

To generate new exploration trees using OpenAI API:
```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Run full pipeline
bash scripts/run_pipeline.sh
```

Individual steps:
```bash
# Generate MCTS trees
python scripts/build_ex_tree.py

# Validate trees
python analyze/count_trees.py

# Extract trajectories for training
python scripts/extract_paths.py
```

## Key Features

### Lazy Tokenization
- Prefixes (scene, description) are cached to avoid redundant tokenization
- Dialogue history is tokenized on-demand during training
- Reduces memory footprint for large-scale datasets

### Distributed Training
- Built on Hugging Face `accelerate` for multi-GPU/multi-node support
- Automatic gradient accumulation and distributed checkpointing
- Fault-tolerant training with resume capability

### LoRA Fine-tuning
- Parameter-efficient adaptation of large language models
- Targets all major transformer projection layers
- Reduces fine-tuning memory and computational requirements

### Comprehensive Evaluation
- Validation during training with configurable frequency
- Metric tracking: loss components, accuracy, flow MSE, KL divergence
- Best checkpoint saving based on custom metrics

## Loss Function Details

### Action Flow Loss
Enforces that cumulative log-probabilities along a trajectory are consistent with estimated cumulative values:

$$\mathcal{L}_{AF} = \mathbb{E}_{(m,n): m < n} \left[ \left( \log F_n - \log F_m - \sum_{i=m}^{n} \log \pi(a_i) \right)^2 \right]$$

where $F_i = Q(s_i) \cdot V(s_i)$ is the flow (cumulative value).

### DPO-Style Value Margin
Enforces margin between preferred and suboptimal actions:

$$\mathcal{L}_{DPO} = \mathbb{E}_{t} \left[ \max(0, \gamma - (V(a_t^+) - V(a_t^-))) \right]$$

### Total Loss
$$\mathcal{L}_{total} = \mathcal{L}_{AF} + \beta \cdot KL(\pi_\theta \| \pi_{ref}) + \mathcal{L}_{DPO}$$

## Configuration Example

See `configs/train_emoflow.yaml` for a complete example. Key sections:

```yaml
afpo_training:
  model_name: "meta-llama/Llama-2-7b-hf"
  max_length: 512
  batch_size: 8
  lr: 1e-4
  epochs: 3
  
  use_lora: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  
  beta: 0.1          # KL regularization
  gamma: 1.0         # Value margin
  
  eval_enabled: true
  eval_every_epochs: 1
  save_best: true
  best_metric: "val_loss"
```

## Dataset Format

Training data should be in JSONL format with trajectories:

```json
{
  "traj_id": "unique-trajectory-id",
  "scene": "Initial emotional situation",
  "description": "Emotional state and context",
  "states": [
    {"history": [{"role": "User", "content": "..."}, {"role": "Bot", "content": "..."}]},
    {"history": [...]}
  ],
  "actions": [
    {"t": 0, "chosen_strategy_id": 3},
    {"t": 1, "chosen_strategy_id": 5}
  ],
  "Q": [value0, value1, ...],           # Q-values (trajectory rewards)
  "V_teacher": [value0, value1, ...],   # V-values (state values)
  "strategy": [
    {"strategy_1": 0.1, "strategy_3": 0.7, ...},  # Action probability distributions
    {...}
  ]
}
```

## Output & Checkpoints

Training outputs are saved to a timestamped directory structure:
```
output_dir/
├── YYYYMMDD_HHMMSS/
│   ├── training_log.jsonl          # Per-epoch metrics
│   ├── checkpoint-N/                # Intermediate checkpoints
│   ├── best/                        # Best checkpoint (if save_best=true)
│   ├── best_metric.json             # Best metric metadata
│   └── logs/                        # Detailed logs
```

Resume training from a checkpoint:
```yaml
afpo_training:
  resume_from_checkpoint: "output_dir/YYYYMMDD_HHMMSS/checkpoint-1000"
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Enable `gradient_checkpointing: true`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use LoRA (`use_lora: true`) for parameter efficiency

### Slow Training
- Increase `num_workers` for data loading (default: 0)
- Set `pin_memory: true` for GPU training
- Use `flatten_steps: true` for optimized batch processing

### API Errors (Tree Generation)
- Ensure `OPENAI_API_KEY` is set and valid
- Check API rate limits and quota
- Verify dataset files exist in `data/raw/`


## Paper

No paper or preprint is available for this repository at this time. If/when a preprint is released, we will add a link here.

## Results

Key results, evaluation metrics, and pointers to notebooks or figures can be added here. Consider adding a short summary and links to visualizations or saved logs.

### Figures (Gallery)

<p align="center">
  <figure style="display:inline-block; margin:8px 18px; text-align:center">
    <a href="assets/figs/fig1.png"><img src="assets/figs/fig1.png" width="480" alt="Figure 1"/></a>
    <figcaption style="font-size:13px; color:#555; margin-top:6px">Figure 1 — model architecture (update caption as needed)</figcaption>
  </figure>
  <figure style="display:inline-block; margin:8px 18px; text-align:center">
    <a href="assets/figs/fig2.png"><img src="assets/figs/fig2.png" width="480" alt="Figure 2"/></a>
    <figcaption style="font-size:13px; color:#555; margin-top:6px">Figure 2 — key evaluation results (update caption as needed)</figcaption>
  </figure>
</p>

> **Note:** Figures are displayed larger and centered for readability. If you prefer different sizes or captions, tell me what to change.

## Citation

Citation information will be added here when a preprint or publication is available.

## Contributing

Contributions are welcome! Please open issues for feature requests or bug reports and submit pull requests for proposed changes. For major changes, please open an issue first to discuss the design.

## License

[![License](https://img.shields.io/github/license/chzou25-lgtm/AffectiveFlow.svg)](https://github.com/chzou25-lgtm/AffectiveFlow/blob/main/LICENSE)

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
