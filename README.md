<p align="center">
  <!-- ====== Link Buttons (replace URLs) ====== -->
  <a href="YOUR_HOMEPAGE_URL"><img src="https://img.shields.io/badge/Homepage-222?style=for-the-badge" /></a>
  <a href="YOUR_PROJECT_PAGE_URL"><img src="https://img.shields.io/badge/AFlow-0ea5e9?style=for-the-badge" /></a>
  <a href="YOUR_DEMO_URL"><img src="https://img.shields.io/badge/Demo-555?style=for-the-badge" /></a>
  <a href="YOUR_ONLINE_URL"><img src="https://img.shields.io/badge/Online-84cc16?style=for-the-badge" /></a>
  <a href="YOUR_PAPER_URL"><img src="https://img.shields.io/badge/Paper-333?style=for-the-badge" /></a>
  <a href="YOUR_PDF_URL"><img src="https://img.shields.io/badge/PDF-b91c1c?style=for-the-badge" /></a>
  <a href="YOUR_HF_MODELS_URL"><img src="https://img.shields.io/badge/Models-HuggingFace-f97316?style=for-the-badge" /></a>
  <a href="YOUR_HF_DATASET_URL"><img src="https://img.shields.io/badge/Dataset-HuggingFace-eab308?style=for-the-badge" /></a>
</p>

# AFlow / Emo_Flow_DPO

Implementation of **AFlow (Affective Flow Language Model)** for **Emotional Support Conversation (ESC)**, with **search-distilled Affective Flow Preference Optimization (AFPO)**.

<!-- ====== Figure 1 Placeholder ====== -->
<p align="center">
  <img src="assets/figs/fig1.png" width="720" />
</p>
<p align="center">
  <em>Figure 1: Comparison of Emotional Support Conversation approaches</em>
</p>

---

## Overview

Large language models (LLMs) have been widely applied to emotional support conversation (ESC), but complex multi-turn support remains challenging because many alignment schemes rely on sparse outcome-level signals and offer limited supervision for intermediate strategy decisions.

**AFlow** introduces **fine-grained supervision on dialogue prefixes** via a **continuous affective flow** along multi-turn trajectories. It:
1) estimates intermediate utility over searched trajectories,  
2) learns preference-consistent strategy transitions, and  
3) improves strategy coherence and empathetic response quality through a **subpath-level flow-balance objective** that propagates preference signals to intermediate states.

---

## Key Features

### 1) Prefix-level affective flow supervision
- Formalizes **Affective Flow** as a **non-negative flow field** over dialogue prefixes, linking terminal affective outcomes to intermediate states under flow-balance constraints.

### 2) Search-distilled training signal (MCTS + role-based environment)
- Uses a **role-separated LLM environment** (Seeker / Supporter / Rewarder) together with **MCTS** to construct fine-grained affective rewards for multi-turn ESC.
- The rewarder evaluates each generated supporter utterance along:
  **Empathy**, **Information Quality**, **Humanoid Naturalness**, and **Strategic Efficacy**.
- The resulting search statistics provide supervision targets over dialogue states and strategy actions.

### 3) AFPO: subpath-level flow-balance preference optimization
- Jointly trains:
  - a **strategy policy** \(\pi_\theta(a\mid s)\)
  - an **evaluation model** \(V_\phi\)
- Enforces **subpath flow-balance** so that preference information is propagated from later outcomes to intermediate decision points.

### 4) Lightweight inference: policy prior + value guidance
At inference time, AFlow selects a supportive strategy by combining the policy prior and the learned action value:
\[
\text{score}(a\mid s_t)=\log \pi_\theta(a\mid s_t)+V_\phi(s_t,a)
\]
and then generates the supporter response conditioned on \((s_t,a_t)\).

<!-- ====== Figure 2 Placeholder ====== -->
<p align="center">
  <img src="assets/figs/fig2.png" width="820" />
</p>
<p align="center">
  <em>Figure 2: Detailed diagram of the AFlow framework for emotional support conversation</em>
</p>

---

## Results

### Automatic evaluation (Table 1)

AFlow shows consistent improvements over competitive baselines on two ESC datasets (**ExTES** and **ESConv**) across:
- **Strategy alignment** (Accuracy / Macro-F1)
- **Generation quality** (BLEU, ROUGE-L, METEOR, PPL)
- **Diversity** (Distinct-1/2)
<p align="center">
  <img src="aassets/figs/table1.png" width="820" />
</p>

### Robustness across backbones (Table 2)

AFlow remains effective across diverse backbone LLMs (e.g., Qwen-2.5 / Gemma-2 / LLaMA-3.1) under different environments (e.g., GPT-4o / Claude-3.5).
<p align="center">
  <img src="assets/figs/table2.pngg" width="500" />
</p>
### Pairwise preference evaluation (Table 3)

AFlow is compared against baselines using **GPT-5.2 judge** and **Human Experts** (Win/Tie/Lose %s).  
<p align="center">
  <img src="assets/figs/table4.pngg" width="500" />
</p>

### Ablation (Table 4)
Removing any core component causes clear degradation:
<p align="center">
  <img src="aassets/figs/table4.png" width="820" />
</p>

---

## Project Structure

```text
Emo_Flow_DPO/
├── scripts/                # Training, tree generation, and data processing
│   ├── build_ex_tree.py    # MCTS tree generation
│   ├── extract_paths.py    # Extract training trajectories from trees
│   ├── train_afpo.py       # AFDPO / Flow-Balance training entry
│   ├── run_pipeline.sh     # End-to-end pipeline (tree → analysis → paths)
│   └── run_train.sh        # Training launcher (verify params inside)
├── analyze/                # Tree stats and visualization
│   ├── count_trees.py
│   └── draw_tree.py
├── configs/                # Config files
│   └── train_emoflow.yaml
├── data/                   # Data and prompts
│   ├── raw/                # Raw datasets (exconv / extes)
│   ├── prompt.json
│   ├── strategies.json
│   └── evaluation_metrics.json
├── requirements.txt        # Python dependencies
└── README.md
````

---

## Requirements

* Python 3.10+
* PyTorch (install the CUDA/CPU build that matches your hardware)
* Other dependencies in `requirements.txt`
* For API-based MCTS: set `OPENAI_API_KEY` and configure `models.*.api_base` in `configs/train_emoflow.yaml`

---

## Quick Start

### 1. Install Environment

```bash
conda create -n emoflow python=3.10 -y
conda activate emoflow

# Install PyTorch (choose the right CUDA/CPU build)
pip install torch torchvision torchaudio

# Install project dependencies
pip install -r requirements.txt
```

Optional: configure accelerate for multi-GPU.

```bash
pip install accelerate
accelerate config
```

### 2. Download Base Model

* Set `afpo_training.model_name` in `configs/train_emoflow.yaml` to a HF model name or a local path.
* If downloading from Hugging Face:

```bash
huggingface-cli login
```

* If using a local model, point `afpo_training.model_name` to the local directory (e.g., `/path/to/model`) and ensure all required model files are present.

### 3. MCTS (Affective Flow signal construction)

MCTS uses `data`, `models`, `mcts`, and `output` in `configs/train_emoflow.yaml`. It reads prompts and strategy definitions from `data/` and writes tree artifacts under `data/processed/...`.

```bash
# If using an API
export OPENAI_API_KEY="your-api-key"

# Generate MCTS trees
python scripts/build_ex_tree.py

# Tree stats (optional)
python analyze/count_trees.py

# Extract root-to-leaf trajectories
python scripts/extract_paths.py
```

Default outputs:

* Trees: `data/processed/extes/Ex_Tree_<split>_<timestamp>.jsonl`
* Paths: `data/processed/extes/Ex_Tree_<split>_<timestamp>_paths.jsonl`
* Metadata: `analyze/tree_paths.json` (auto-read by training)

### 4. Start Training: AFPO (Flow-Balance)

Training reads `afpo_training` from `configs/train_emoflow.yaml` and resolves the latest paths via `analyze/tree_paths.json`.

```bash
python scripts/train_afpo.py
```

Multi-GPU (optional):

```bash
accelerate launch --multi_gpu scripts/train_afpo.py
```

### 5. One-command pipeline (optional)

If you want an end-to-end run (tree → analysis → paths → training), verify params inside scripts first:

```bash
bash scripts/run_pipeline.sh
```

---

## Configuration

Edit `configs/train_emoflow.yaml`:

* `data.processed_dir` / `data.split`: dataset directory and split
* `models.*`: MCTS-stage LLMs (provider / model_name / temperature / api_base / api_key_env)
* `mcts.*`: search settings (`simulations_per_tree`, `max_depth`, `reward_weights`, etc.)
* `output.*`: tree/log outputs and `scene_limit`
* `path_extraction.*`: path length filters (`min_path_length` / `max_path_length`)
* `afpo_training.*`: training hyperparameters (`model_name`, `batch_size`, `lr`, `beta`, `gamma`, `use_lora`, etc.)
* `run.offline`: set `true` to force offline generation (no API calls)

---

## License

This repository is released under the license specified in this project.

```
::contentReference[oaicite:0]{index=0}
```
