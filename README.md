<!-- <p align="center"> -->
  <!-- ====== Link Buttons (replace URLs) ====== -->
  <!-- <a href="YOUR_HOMEPAGE_URL"><img src="https://img.shields.io/badge/Homepage-222?style=for-the-badge" /></a> -->
  <!-- <a href="YOUR_PROJECT_PAGE_URL"><img src="https://img.shields.io/badge/AFlow-0ea5e9?style=for-the-badge" /></a> -->
  <!-- <a href="YOUR_DEMO_URL"><img src="https://img.shields.io/badge/Demo-555?style=for-the-badge" /></a> -->
  <!-- <a href="YOUR_ONLINE_URL"><img src="https://img.shields.io/badge/Online-84cc16?style=for-the-badge" /></a> -->
  <!-- <a href="YOUR_PAPER_URL"><img src="https://img.shields.io/badge/Paper-333?style=for-the-badge" /></a> -->
  <!-- <a href="YOUR_PDF_URL"><img src="https://img.shields.io/badge/PDF-b91c1c?style=for-the-badge" /></a> -->
  <!-- <a href="YOUR_HF_MODELS_URL"><img src="https://img.shields.io/badge/Models-HuggingFace-f97316?style=for-the-badge" /></a> -->
  <!-- <a href="YOUR_HF_DATASET_URL"><img src="https://img.shields.io/badge/Dataset-HuggingFace-eab308?style=for-the-badge" /></a> -->
<!-- </p> -->
<!-- =======================
AFlow Header (Logo + Badges)
Replace ALL YOUR_* placeholders
======================= -->

<p align="center">
  <!-- Logo (replace file) -->
  <img src="assets/figs/aflow_logo.png" width="88" alt="AFlow" />
</p>

<h1 align="center">AFlow</h1>

<p align="center">
  <b>Affective Flow Language Model for Emotional Support Conversation</b>
</p>

<!-- Badges Row (edit to match your paper) -->
<p align="center">
  <!-- Paper / arXiv -->
  <a href="YOUR_PAPER_URL"><img src="https://img.shields.io/badge/Paper-PDF-111?style=for-the-badge" /></a>
  <a href="YOUR_ARXIV_URL"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?style=for-the-badge" /></a>

  <!-- Venue / Year -->
  <a href="YOUR_VENUE_URL"><img src="https://img.shields.io/badge/IJCAI-2025-2563eb?style=for-the-badge" /></a>

  <!-- License -->
  <a href="YOUR_LICENSE_URL"><img src="https://img.shields.io/badge/License-YOUR__LICENSE-84cc16?style=for-the-badge" /></a>

  <!-- Python / PyTorch -->
  <img src="https://img.shields.io/badge/Python-3.10+-334155?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-f97316?style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

<!-- Quick Nav (edit anchors to match your sections) -->
<p align="center">
  <a href="#paper">📄 Overview</a> •
  <a href="#methodology">🔬 Methodology</a> •
  <a href="#quick-start">🚀 Quick Start</a> •
  <a href="#results">📊 Results</a> •
  <a href="#citation">📝 Citation</a>
</p>

<hr/>
# Affective Flow

Implementation of **AFlow (Affective Flow Language Model)** for **Emotional Support Conversation (ESC)**, with **search-distilled Affective Flow Preference Optimization (AFPO)**.

<!-- ====== Figure 1 Placeholder ====== -->
<p align="center">
  <img src="assets/figs/fig1.png" width="720" />
</p>
<p align="center">
  <em>Figure 1: Comparison of Emotional Support Conversation approaches</em>
</p>

---

## 📄Overview

Large language models (LLMs) have been widely used for **Emotional Support Conversation (ESC)**, but **multi-turn** support is still difficult: effective support requires **planning and maintaining strategy coherence** across turns, while many alignment methods provide supervision only at a coarse granularity (e.g., response-level or outcome-level), making **credit assignment for intermediate strategy decisions** challenging.

**AFlow** addresses this by learning from **search-distilled multi-turn trajectories** with **prefix-level supervision**. We model a dialogue prefix as a state $s_t$ and a supportive strategy as an action $a_t$. AFlow trains:

- a **strategy policy** $\pi_\theta(a \mid s)$ for choosing the next strategy, and  
- a **value/evaluation model** $V_\phi(s, a)$ for assessing strategy quality under a given prefix,

and leverages **MCTS** to construct informative multi-turn trajectories that provide stronger supervision signals for intermediate decisions.

At test time, AFlow selects strategies by combining the policy prior and the learned value signal:
$$
a_t = \arg\max_{a \in \mathcal{A}} \Big( \log \pi_\theta(a \mid s_t) + V_\phi(s_t, a) \Big),
$$
then generates the supporter response conditioned on $(s_t, a_t)$.

---

## Key Features

### 1) Problem formulation: prefix state + strategy action
We model a multi-turn ESC dialogue as a sequence of states and actions:
- state (dialogue prefix): $s_t = (x_{\le t})$
- action (supportive strategy): $a_t \in \mathcal{A}$
- transition induced by generation: $s_{t+1} \sim P(\cdot \mid s_t, a_t)$

This formulation makes **strategy learning** explicit, rather than implicitly learning it from raw text generation alone.

### 2) Search-distilled supervision via MCTS
AFlow uses **MCTS** to explore multi-turn continuations from a prefix $s_t$ under candidate strategies $a_t$, producing a trajectory tree with:
- explored rollouts,
- state/action visitation statistics,
- and outcome-dependent signals that reflect long-horizon effects.

These search artifacts are distilled into training data for learning $\pi_\theta$ and $V_\phi$.

> (Place your paper’s Figure 2 here as a placeholder)
<p align="center">
  <img src="assets/figure2_framework.png" width="820" />
</p>
<p align="center">
  <em>Figure 2. (Placeholder) AFlow framework: MCTS-based trajectory construction, AFPO training, and policy/value-guided inference.</em>
</p>

### 3) AFPO / flow-balance style learning objective (prefix-consistent credit assignment)
To propagate supervision from later outcomes back to intermediate prefixes, AFlow introduces a **flow-balance style objective** defined on dialogue prefixes.

Let $F(\cdot)$ denote a non-negative quantity (e.g., “flow” / “mass”) associated with prefixes or state-action pairs. The training objective enforces **consistency constraints** so that intermediate decisions are trained with signals that are consistent with long-horizon evaluations.

A generic flow-balance form can be written as:
$$
\text{(incoming flow at } s) \;\approx\; \text{(outgoing flow from } s) \;+\; \text{(terminal contribution)},
$$
implemented over **subpaths/prefixes** to strengthen intermediate supervision.

> **Important**: Replace the above generic balance equation with the *exact equation from your paper* (same symbols and terms), e.g., your “subpath flow-balance” constraint and your AFPO loss decomposition.

### 4) Lightweight inference with policy/value guidance
AFlow avoids heavy test-time search by using a lightweight scoring rule:
$$
\mathrm{score}(a \mid s_t) = \log \pi_\theta(a \mid s_t) + V_\phi(s_t, a),
$$
then selecting $a_t$ and generating the next response conditioned on $(s_t, a_t)$.
This yields stable strategy selection while keeping inference efficient.


## 🔬Methodology

<!-- ====== Figure 2 Placeholder ====== -->
<p align="center">
  <img src="assets/figs/fig2.png" width="820" />
</p>
<p align="center">
  <em>Figure 2: Detailed diagram of the AFlow framework for emotional support conversation</em>
</p>

---

## 📊Results

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
  <img src="assets/figs/table2.png" width="500" />
</p>
### Pairwise preference evaluation (Table 3)

AFlow is compared against baselines using **GPT-5.2 judge** and **Human Experts** (Win/Tie/Lose %s).  
<p align="center">
  <img src="assets/figs/table4.png" width="500" />
</p>

### Ablation (Table 4)
Removing any core component causes clear degradation:
<p align="center">
  <img src="assets/figs/table4.png" width="820" />
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

## 🚀Quick Start

###  1. Install Environment

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
This project is licensed under the MIT License - see the LICENSE file for details.
```
