#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Flow and Preference Optimization (AFPO) Classifier Training

This module implements the training pipeline for an AFPO classifier that performs
action selection in emotional support conversations. The classifier uses a 
language model backbone with a preference-based learning objective that incorporates:
  - Action Flow (AF): Cumulative log-probability constraints across decision steps
  - Direct Preference Optimization: Margin-based loss between preferred and suboptimal actions
  - Value-based guidance: Contextual value estimates for state-action pairs

Key Features:
  - Lazy tokenization with prefix caching for memory efficiency
  - Distributed training via Hugging Face Accelerate
  - LoRA (Low-Rank Adaptation) for efficient model fine-tuning
  - Comprehensive evaluation metrics and checkpoint management
  - Resume capability from saved checkpoints
"""
from __future__ import annotations
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
import json
import os
import shutil
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import yaml
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from peft import LoraConfig, get_peft_model, TaskType

# Number of support strategies (action space dimension)
NUM_CLASSES = 8


def render_history(history: List[Dict[str, Any]]) -> str:
    """Format dialogue history into a readable conversation string.
    
    Args:
        history: List of dialogue turns, each containing 'role' and 'content'.
        
    Returns:
        Formatted multi-line dialogue string.
    """
    lines: List[str] = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not role or not content:
            continue
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def load_strategy_id_map(path: Path = Path("data/strategies.json")) -> Dict[str, int]:
    """Load mapping from strategy names to integer IDs.
    
    Args:
        path: Path to strategies JSON file.
        
    Returns:
        Dictionary mapping strategy name -> strategy ID.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    name_to_id: Dict[str, int] = {}
    for idx, item in enumerate(data):
        if isinstance(item, dict) and "strategy" in item:
            name_to_id[item["strategy"]] = idx
    return name_to_id


def id_to_name_map(name_to_id: Dict[str, int]) -> List[str]:
    """Convert strategy ID-to-name mapping to a list indexed by ID.
    
    Args:
        name_to_id: Dictionary mapping strategy name -> ID.
        
    Returns:
        List where index i contains strategy name for ID i.
    """
    if not name_to_id:
        return []
    max_idx = max(name_to_id.values())
    names = [""] * (max_idx + 1)
    for n, i in name_to_id.items():
        if 0 <= i < len(names):
            names[i] = n
    return names


def load_samples(path: Path) -> List[Dict[str, Any]]:
    """Load training trajectories from JSONL file.
    
    Supports both JSONL format (one object per line) and flat JSON arrays.
    Each trajectory contains states (dialogue history), actions (strategy selections),
    and auxiliary values (Q-values, V-values, etc.).
    
    Args:
        path: Path to JSONL or JSON trajectory file.
        
    Returns:
        List of trajectory dictionaries.
        
    Raises:
        ValueError: If file content is not a list of samples.
    """
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                data.extend(obj)
            else:
                data.append(obj)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}")
    return data


def load_cls_config(cfg_path: Path = Path("configs/train_emoflow.yaml")) -> Dict[str, Any]:
    """Load training configuration from YAML file.
    
    Validates that all required hyperparameters are present, including:
    - Model and training parameters (learning rate, batch size, epochs)
    - LoRA configuration (if using adapter-based fine-tuning)
    - Evaluation settings (validation frequency, metrics)
    - Loss function coefficients (beta for KL penalty, gamma for value margin)
    
    Args:
        cfg_path: Path to training configuration YAML file.
        
    Returns:
        Dictionary containing all configuration parameters.
        
    Raises:
        FileNotFoundError: If configuration file not found.
        ValueError: If required keys are missing.
    """

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    sec = raw.get("afpo_training") or raw.get("afpo_classifier")
    if not isinstance(sec, dict):
        raise ValueError("Missing afpo_training section in configs/train_emoflow.yaml")

    required = [
        "model_name",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "lr",
        "max_length",
        "log_path",
        "output_dir",
        "save_steps",
        "save_total_limit",
        "beta",
        "use_lora",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "seed",
        "gamma",
        "eval_enabled",
        "eval_every_epochs",
        "eval_use_ref_model",
        "flatten_steps",
    ]
    missing = [k for k in required if k not in sec]
    if missing:
        raise ValueError(f"Missing required AFPO config keys: {missing}")

    return sec

# ========================
# Dataset & Collate Function
# ========================

class LazyTokenizedDataset(Dataset):
    """Lazy tokenization dataset for trajectory sequences.
    
    This dataset loads trajectory data on-the-fly and caches tokenized prefixes
    to avoid redundant tokenization. Each trajectory consists of multiple
    dialogue steps, where each step has:
      - state: dialogue history (context for decision)
      - action: chosen strategy ID (preferred action)
      - auxiliary values: Q-values, V-values for preference learning
      
    The tokenization is done lazily to reduce memory overhead. Prefixes
    (scene + description) are cached since they repeat across steps in
    the same trajectory.
    
    Args:
        samples: List of trajectory dictionaries.
        tokenizer: Hugging Face tokenizer instance.
        max_len: Maximum sequence length (for padding/truncation).
        strat_id_map: Mapping from strategy names to integer IDs.
        desc: Description for progress bar.
    """
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_len: int, strat_id_map: Dict[str, int], desc: str = "Processing"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.strat_id_map = strat_id_map
        self.items = []
        self._prefix_cache: Dict[str, List[int]] = {}

        for sample in tqdm(samples, desc=f"Scanning {desc}", disable=int(os.environ.get("LOCAL_RANK", -1)) > 0):
            actions = sample.get("actions", [])
            states = sample.get("states", [])
            if not actions or not states or len(actions) > len(states):
                continue
            traj_data = {
                "scene": sample.get("scene"),
                "description": sample.get("description"),
                "traj_id": sample.get("traj_id"),
                "actions": actions,
                "states": states,
                "v_teacher": sample.get("V_teacher"),
                "q_source": sample.get("Q"),
                "strategy": sample.get("strategy") or [],
            }
            self.items.append(traj_data)
    def _build_prefix(self, scene: str, description: str) -> str:
        """Build the context prefix for a trajectory.
        
        The prefix provides domain context (scene and emotional description)
        that is constant across all decision steps in a trajectory.
        
        Args:
            scene: Environmental/contextual description.
            description: Emotional situation summary.
            
        Returns:
            Formatted prefix string.
        """
        scene = scene or ""
        description = description or ""
        return f"SCENE: {scene}\nDESC: {description}\nTask: Select the next support strategy id (0-7).\n"

    def _get_prefix_ids(self, prefix: str) -> List[int]:
        """Get cached tokenized prefix IDs to avoid redundant tokenization.
        
        Uses an in-memory cache to store prefix token IDs since prefixes
        repeat across trajectory steps. This significantly reduces tokenization
        overhead during training.
        
        Args:
            prefix: Prefix string to tokenize.
            
        Returns:
            List of token IDs for the prefix (without special tokens).
        """
        if prefix in self._prefix_cache:
            return self._prefix_cache[prefix]

        # Tokenize once without special tokens; they're added during sequence assembly
        ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self._prefix_cache[prefix] = ids
        return ids
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Construct a training example from a single trajectory.
        
        For each trajectory, this method:
        1. Extracts decision steps (state, chosen action, alternative action)
        2. Tokenizes dialogue history with cached prefixes
        3. Applies padding/truncation to fixed sequence length
        4. Returns structured tensor data for the loss computation
        
        Each trajectory yields multiple (state, action) pairs for preference learning.
        
        Args:
            idx: Trajectory index.
            
        Returns:
            Dictionary containing:
            - input_ids: Tokenized sequence [T, max_len]
            - attention_mask: Attention mask [T, max_len]
            - labels: Preferred action IDs [T]
            - q_values: Q-values for each state [T]
            - v_values: Value estimates [T]
            - worst_ids: Alternative (suboptimal) action IDs [T]
            - traj_id: Trajectory identifier
        """

        texts = []
        labels = []
        q_vals = []
        v_vals = []
        worst_ids: List[int] = []
        steps: List[int] = []

        # Extract decision steps from trajectory
        for act in actions:
            t = act.get("t")  # Decision step index
            a_id = act.get("chosen_strategy_id")  # Preferred action
            if t is None or a_id is None or t >= len(states):
                continue

            # Dialogue history at this decision point (conversation context)
            hist = states[t].get("history", [])
            text = render_history(hist)
            texts.append(text)
            labels.append(int(a_id))

            # Q-value and V-value for preference learning
            q_vals.append(float(q_source[t]) if t < len(q_source) else 0.0)
            v_vals.append(float(v_vals_raw[t]) if t < len(v_vals_raw) else 1.0)
            
            # Sample worst (suboptimal) action for DPO-style learning
            worst_id: Optional[int] = None
            strat_list = item.get("strategy") or []
            if t < len(strat_list) and isinstance(strat_list[t], dict):
                probs = strat_list[t]
                if probs:
                    # Select from lowest-probability actions to get true negatives
                    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1])
                    candidates = []
                    for name, _p in sorted_probs:
                        if name in self.strat_id_map:
                            candidates.append(self.strat_id_map[name])
                        if len(candidates) >= 3:
                            break
                    if candidates:
                        worst_id = random.choice(candidates)
            if worst_id is None:
                worst_id = int(a_id)
            worst_ids.append(int(worst_id))
            steps.append(int(t))

        if not texts:
            return None
        
        # Tokenize dialogue history (suffix part, not including prefix)
        suffix_enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_tensors=None
        )
        suffix_ids_list = suffix_enc["input_ids"]

        # Assemble final sequences: BOS + prefix + suffix + EOS, then truncate and pad
        input_ids_batch = []
        attn_mask_batch = []

        for suffix_ids in suffix_ids_list:
            ids = []
            if bos is not None:
                ids.append(bos)
            ids.extend(prefix_ids)
            ids.extend(suffix_ids)
            if eos is not None:
                ids.append(eos)

            # Truncation strategy: preserve prefix (context) and suffix tail (recent history)
            if len(ids) > self.max_len:
                fixed_len = (1 if bos is not None else 0) + len(prefix_ids) + (1 if eos is not None else 0)
                avail = self.max_len - fixed_len
                if avail < 0:
                    # Extreme case: prefix itself exceeds max_len
                    ids = ids[-self.max_len:]
                else:
                    # Keep: BOS + prefix + suffix_tail + EOS
                    suffix_tail = suffix_ids[-avail:] if avail > 0 else []
                    ids = []
                    if bos is not None:
                        ids.append(bos)
                    ids.extend(prefix_ids)
                    ids.extend(suffix_tail)
                    if eos is not None:
                        ids.append(eos)

            attn = [1] * len(ids)

            # Pad to max_len
            if len(ids) < self.max_len:
                pad_len = self.max_len - len(ids)
                ids = ids + [pad] * pad_len
                attn = attn + [0] * pad_len

            input_ids_batch.append(ids)
            attn_mask_batch.append(attn)

        # Convert to tensors: [T, max_len]
        input_ids = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask_batch, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "q_values": torch.tensor(q_vals, dtype=torch.float),
            "v_values": torch.tensor(v_vals, dtype=torch.float),
            "worst_ids": torch.tensor(worst_ids, dtype=torch.long),
            "steps": torch.tensor(steps, dtype=torch.long),
            "traj_id": item.get("traj_id"),
        }


def cls_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collate function for batching trajectories.
    
    Filters out None samples (invalid trajectories) and returns a list
    of valid trajectory samples for the training step.
    
    Args:
        batch: List of samples from dataset.
        
    Returns:
        Filtered list of non-None samples.
    """
    return [b for b in batch if b is not None]


# ========================
# Model Architecture
# ========================
class ClassifierHead(nn.Module):
    """Simple linear classification head for action prediction.
    
    Takes the last hidden state from the language model and projects
    it to the action space (8 support strategies).
    """
    def __init__(self, hidden_size: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to action logits.
        
        Args:
            hidden: [batch_size, hidden_size]
            
        Returns:
            Logits over action space [batch_size, num_classes]
        """
        return self.linear(hidden)


class AFPOClassifier(nn.Module):
    """AFPO Classifier with language model backbone and dual heads.
    
    Architecture:
    - Backbone: Transformer language model (e.g., Llama, Qwen)
    - Classification head: Predicts preferred action logits
    - Value head: Estimates state value for DPO margin computation
    
    Supports:
    - LoRA fine-tuning for parameter efficiency
    - Gradient checkpointing to reduce memory
    - Flexible cache management
    
    Args:
        base_name: HuggingFace model ID for the language model.
        num_classes: Number of action classes (default: 8 strategies).
        use_lora: Whether to apply LoRA adapter.
        lora_config: LoRA hyperparameters (rank, alpha, dropout).
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency.
        use_cache: Whether to cache attention values in model.
    """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_name, trust_remote_code=True)
        
        # Apply LoRA adapter for efficient fine-tuning
        if use_lora:
            peft_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_config.get("lora_rank", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                lora_dropout=lora_config.get("lora_dropout", 0.05),
                bias="none",
                # Target all major projection layers for maximum expressiveness
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","mlp.dense_h_to_4h","mlp.dense_4h_to_h","embed_tokens","embed_positions","lm_head"]
            )
            self.backbone = get_peft_model(self.backbone, peft_cfg)
        
        # Gradient checkpointing: trade compute for memory
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        elif hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

        # Configure attention cache behavior
        if hasattr(self.backbone, "config"):
            if use_cache is None:
                self.backbone.config.use_cache = not gradient_checkpointing
            else:
                self.backbone.config.use_cache = bool(use_cache)
        
        hidden = self.backbone.config.hidden_size
        self.head = ClassifierHead(hidden, num_classes)
        self.head.float()
        
        # Value head: estimates normalized value for each action
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
            nn.Softplus()
        )
        self.value_head.float()

    def forward(self, input_ids, attention_mask):
        """Forward pass: extract features and compute action/value logits.
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Action logits [batch_size, num_classes]
            v_logits: Value estimates [batch_size, num_classes]
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state
        
        # Extract last non-padded token representation
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
        last_hidden = hidden_states.gather(dim=1, index=idx).squeeze(1)
        
        logits = self.head(last_hidden)
        v_logits = self.value_head(last_hidden)
        return logits, v_logits
    
    def print_trainable_parameters(self):
        """Print statistics on trainable vs total parameters."""
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.parameters())
            print(f"trainable params: {trainable} || all params: {all_params} || trainable%: {100 * trainable / all_params:.4f}")


# ========================
# Loss Function
# ========================

def flow_balance_loss(logits, logits_ref, v_logits, actions, worst_ids, q_values, v_teacher, beta=0.1, gamma=1.0):
    """Compute the Action Flow and Preference Optimization (AFPO) loss.
    
    The loss combines three components:
    
    1. Flow Loss (main objective): Enforces action flow constraints
       - Flow = q_value * v_value (cumulative value estimate along trajectory)
       - Constraint: flow difference between state intervals ≈ accumulated log-prob
       
    2. KL Divergence (regularization): Divergence from reference model
       - Coefficient: beta
       
    3. Value Margin Loss (DPO-style): Margin between preferred and worst actions
       - Coefficient: gamma
       - Margin enforces v[preferred] - v[worst] ≥ gamma
    
    Args:
        logits: Action logits from classifier [T, num_classes]
        logits_ref: Action logits from reference model [T, num_classes] or None
        v_logits: Value estimates [T, num_classes]
        actions: Preferred action IDs [T]
        worst_ids: Worst (negative) action IDs [T]
        q_values: Q-values for trajectory states [T]
        v_teacher: Target value estimates [T]
        beta: KL regularization coefficient
        gamma: Value margin coefficient
        
    Returns:
        tuple: (total_loss, flow_mse, kl_div, accuracy, eval_loss)
    """
    logprobs = torch.log_softmax(logits, dim=-1)
    
    # Log-probability of preferred actions
    lp = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    # KL divergence from reference model
    if logits_ref is not None:
        logprobs_ref = torch.log_softmax(logits_ref, dim=-1)
        lp_ref = logprobs_ref.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        kl = (lp - lp_ref).mean()
    else:
        kl = torch.zeros((), device=logits.device)
    
    # Value estimates for preferred and worst actions
    v_pos = v_logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    v_neg = v_logits.gather(-1, worst_ids.unsqueeze(-1)).squeeze(-1)

    T = lp.size(0)
    device = lp.device

    # Classification accuracy on preferred actions
    preds = logits.argmax(dim=-1)
    correct = (preds == actions).float().sum()
    accuracy = correct / T

    # Compute action flow constraints
    # prefix[i] = cumulative log-prob up to step i
    prefix = torch.cat([torch.zeros(1, device=device), torch.cumsum(lp, dim=0)])
    idx = torch.arange(T, device=device)
    m_idx, n_idx = torch.meshgrid(idx, idx, indexing="ij")
    
    # delta_rho[m, n] = log(rho_m, rho_n) = sum_{i=m}^{n} log pi(a_i)
    delta_rho = prefix[n_idx + 1] - prefix[m_idx]
    
    # flow[i] = q_value[i] * v_value[i]
    flow = q_values * v_pos
    logF = torch.log(torch.clamp(flow, min=1e-6))
    logF_diff = logF[n_idx] - logF[m_idx]

    # Only compare pairs (m, n) where m < n (upper triangle)
    tri_mask = torch.triu(torch.ones_like(delta_rho), diagonal=1)
    flow_err = (logF_diff - delta_rho) * tri_mask
    
    pair_count = tri_mask.sum().clamp(min=1.0)
    flow_mse = (flow_err.pow(2).sum() / pair_count)

    # DPO-style value margin loss: ensure v[preferred] - v[worst] ≥ gamma
    eval_loss = torch.relu(gamma - (v_pos - v_neg)).mean()
    
    # Combine all loss terms
    total_loss = flow_mse + beta * torch.relu(kl) + eval_loss
    
    return total_loss, flow_mse, kl, accuracy, eval_loss


# ========================
# Evaluation
# ========================
def evaluate(model, ref_model, loader, beta, gamma, accelerator, use_ref_model: bool, flatten_steps: bool):
    """Run evaluation on validation set.
    
    Computes loss, flow error, KL divergence, and action accuracy across all
    validation trajectories using the same loss computation as training.
    
    Args:
        model: Trained classifier model
        ref_model: Reference model for KL computation
        loader: Validation data loader
        beta: KL coefficient
        gamma: Value margin coefficient
        accelerator: Distributed training context
        use_ref_model: Whether to compute KL divergence
        flatten_steps: Whether to use flattened batch processing
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    # Metric accumulators (distributed safe)
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_flow = torch.tensor(0.0, device=accelerator.device)
    total_kl = torch.tensor(0.0, device=accelerator.device)
    total_eval = torch.tensor(0.0, device=accelerator.device)
    total_acc = torch.tensor(0.0, device=accelerator.device)
    total_samples = torch.tensor(0.0, device=accelerator.device)
    
    for batch in tqdm(loader, desc="Evaluating", disable=not accelerator.is_local_main_process):
        if not batch:
            continue
        if flatten_steps:
            flat_input_ids = []
            flat_attention_mask = []
            sample_meta = []
            offset = 0

            for sample in batch:
                input_ids = sample["input_ids"].to(accelerator.device)
                attention_mask = sample["attention_mask"].to(accelerator.device)
                labels = sample["labels"].to(accelerator.device)
                q_vals = sample["q_values"].to(accelerator.device)
                v_vals = sample["v_values"].to(accelerator.device)
                worst_ids = sample["worst_ids"].to(accelerator.device)

                step_len = labels.size(0)
                if step_len == 0:
                    continue
                if input_ids.size(0) != step_len:
                    min_len = min(step_len, input_ids.size(0))
                    input_ids = input_ids[:min_len]
                    attention_mask = attention_mask[:min_len]
                    labels = labels[:min_len]
                    q_vals = q_vals[:min_len]
                    v_vals = v_vals[:min_len]
                    worst_ids = worst_ids[:min_len]
                    step_len = min_len

                flat_input_ids.append(input_ids)
                flat_attention_mask.append(attention_mask)
                sample_meta.append({
                    "start": offset,
                    "end": offset + step_len,
                    "labels": labels,
                    "q_vals": q_vals,
                    "v_vals": v_vals,
                    "worst_ids": worst_ids,
                })
                offset += step_len

            if not sample_meta:
                continue

            flat_input_ids = torch.cat(flat_input_ids, dim=0)
            flat_attention_mask = torch.cat(flat_attention_mask, dim=0)

            with torch.no_grad():
                logits, v_logits = model(flat_input_ids, flat_attention_mask)
                logits_ref = None
                if use_ref_model:
                    logits_ref, _ = ref_model(flat_input_ids, flat_attention_mask)

            for meta in sample_meta:
                start = meta["start"]
                end = meta["end"]
                logits_s = logits[start:end]
                v_logits_s = v_logits[start:end]
                logits_ref_s = logits_ref[start:end] if logits_ref is not None else None

                loss, flow, kl, acc, eval_loss = flow_balance_loss(
                    logits_s, logits_ref_s, v_logits_s,
                    meta["labels"], meta["worst_ids"], meta["q_vals"], meta["v_vals"],
                    beta, gamma
                )

                total_loss += loss
                total_flow += flow
                total_kl += kl
                total_eval += eval_loss
                total_acc += acc
                total_samples += 1
        else:
            # Batch is list of trajectories
            for sample in batch:
                input_ids = sample["input_ids"].to(accelerator.device)
                attention_mask = sample["attention_mask"].to(accelerator.device)
                labels = sample["labels"].to(accelerator.device)
                q_vals = sample["q_values"].to(accelerator.device)
                v_vals = sample["v_values"].to(accelerator.device)
                worst_ids = sample["worst_ids"].to(accelerator.device)

                with torch.no_grad():
                    logits, v_logits = model(input_ids, attention_mask)
                    logits_ref = None
                    if use_ref_model:
                        logits_ref, _ = ref_model(input_ids, attention_mask)
                    loss, flow, kl, acc, eval_loss = flow_balance_loss(
                        logits, logits_ref, v_logits, labels, worst_ids, q_vals, v_vals, beta, gamma
                    )

                total_loss += loss
                total_flow += flow
                total_kl += kl
                total_eval += eval_loss
                total_acc += acc
                total_samples += 1
            
    # Distributed Gather
    # Sum up all metrics across all processes
    # Note: gather_for_metrics is simpler for tensor batches, but manual reduce is safer for scalars
    all_loss = accelerator.reduce(total_loss, reduction="sum")
    all_flow = accelerator.reduce(total_flow, reduction="sum")
    all_kl = accelerator.reduce(total_kl, reduction="sum")
    all_acc = accelerator.reduce(total_acc, reduction="sum")
    all_eval = accelerator.reduce(total_eval, reduction="sum")
    all_count = accelerator.reduce(total_samples, reduction="sum")
    
    # Safe division
    count = max(1.0, all_count.item())
    metrics = {
        "val_loss": all_loss.item() / count,
        "val_flow_mse": all_flow.item() / count,
        "val_kl": all_kl.item() / count,
        "val_eval": all_eval.item() / count,
        "val_acc": all_acc.item() / count
    }
    
    model.train()
    return metrics

# --- Checkpoint Management ---
def rotate_checkpoints(output_dir: Path, limit: int):
    if not output_dir.exists(): return
    checkpoints = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")], key=lambda x: os.path.getmtime(x))
    if len(checkpoints) > limit:
        for ckpt in checkpoints[:-limit]:
            shutil.rmtree(ckpt)
# 6
# Resolve data path:
# Resolve dataset paths (train/val/test) from analyze/tree_paths.json.
# Falls back to config data_path for train and default paths if missing.
def resolve_split_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    meta = Path("analyze/tree_paths_rel.json")
    train_path = cfg.get("data_path")
    val_path = None
    test_path = None

    if meta.exists():
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            train_path = info.get("tree_path_train") 
            val_path = info.get("tree_path_val")
            test_path = info.get("tree_path_test")
        except Exception:
            pass

    def normalize(p: str | None, default: str) -> Path:
        base = Path(p) if p else Path(default)
        cand = base.with_name(f"{base.stem}_paths.jsonl")
        if cand.exists():
            return cand
        cand_json = base.with_name(f"{base.stem}_paths.json")
        if cand_json.exists():
            return cand_json
        if base.exists():
            return base
        return Path(default)

    train_path_p = normalize(train_path, "data/processed/extes/Ex_Tree_paths.jsonl")
    val_path_p = normalize(val_path, "data/processed/extes/Ex_Tree_val_paths.jsonl")
    test_path_p = normalize(test_path, "data/processed/extes/Ex_Tree_test_paths.jsonl")

    return {"train": train_path_p, "val": val_path_p, "test": test_path_p}

# --- Main ---
def main():
    cfg = load_cls_config()
    paths = resolve_split_paths(cfg)
    train_path = paths["train"]
    val_path = paths["val"]

    # Backup original train paths file
    backup_path = train_path.with_suffix(train_path.suffix + ".bak")
    shutil.copy(train_path, backup_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine Output Dir (always new run; resume via cfg if needed)
    out_dir = Path(cfg["output_dir"]) / ts

    project_config = ProjectConfiguration(project_dir=str(out_dir), automatic_checkpoint_naming=True)
    # 1. 设置超时时间为 3600 秒 (1小时)
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        project_config=project_config,
         kwargs_handlers=[timeout_kwargs]
    )
    set_seed(int(cfg["seed"]))

    # Setup Logging Path (Main Process)
    log_file_path = None
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Create a specific log file for this run inside the output dir to avoid overwrites
        log_file_path = out_dir / "training_log.jsonl"
        print(f"Output Directory: {out_dir}")
        print(f"Logging to: {log_file_path}")
    save_best = bool(cfg.get("save_best", False))
    best_metric = str(cfg.get("best_metric", "val_loss"))
    best_mode = str(cfg.get("best_metric_mode", "min")).lower()
    if best_mode not in ("min", "max"):
        raise ValueError("best_metric_mode must be 'min' or 'max'")
    best_value = float("inf") if best_mode == "min" else -float("inf")
    save_steps = int(cfg.get("save_steps", 0))
    save_epoch_checkpoint = bool(cfg.get("save_epoch_checkpoint", True))
    eval_enabled = bool(cfg.get("eval_enabled", True))
    eval_every_epochs = int(cfg.get("eval_every_epochs", 1))
    eval_use_ref_model = bool(cfg.get("eval_use_ref_model", True))
    flatten_steps = bool(cfg.get("flatten_steps", False))

    # 2. Data
    accelerator.print("Loading Data...")
    train_samples = load_samples(train_path)
    val_samples = load_samples(val_path)
    strat_id_map = load_strategy_id_map()
    id_to_name = id_to_name_map(strat_id_map)
    updated_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    accelerator.print("Initializing Datasets (Lazy)...")
    strat_id_map = load_strategy_id_map()
    id_to_name = id_to_name_map(strat_id_map)
    updated_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    train_ds = LazyTokenizedDataset(train_samples, tokenizer, cfg["max_length"], strat_id_map, "Train")
    val_ds = LazyTokenizedDataset(val_samples, tokenizer, cfg["max_length"], strat_id_map, "Val")
    
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", True))
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "collate_fn": cls_collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # 3. Model
    accelerator.print("Initializing Models...")
    model = AFPOClassifier(
        cfg["model_name"],
        use_lora=cfg["use_lora"],
        lora_config=cfg,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        use_cache=cfg.get("use_cache"),
    )
    ref_model = AFPOClassifier(
        cfg["model_name"],
        use_lora=False,
        gradient_checkpointing=False,
        use_cache=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    
    num_update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    max_train_steps = cfg["epochs"] * num_update_steps_per_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * 0.1),
        num_training_steps=max_train_steps,
    )

    # 4. Prepare
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    ref_model.to(accelerator.device)

    # 5. Resume Logic (config-driven; set cfg['resume_from_checkpoint'] if needed)
    global_step = 0
    start_epoch = 0
    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt:
        accelerator.print(f"Resuming from {resume_ckpt}")
        accelerator.load_state(resume_ckpt)
        try:
            global_step = int(Path(resume_ckpt).name.split("-")[-1])
            start_epoch = global_step // num_update_steps_per_epoch
        except ValueError:
            pass

    accelerator.print(f"Starting training from Epoch {start_epoch}")
    # 6. Training Loop
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss_total = 0.0
        epoch_flow_total = 0.0
        epoch_eval_total = 0.0
        epoch_count_total = 0

        if resume_ckpt and epoch == start_epoch:
            steps_in_epoch = global_step % num_update_steps_per_epoch
            batches_to_skip = steps_in_epoch * accelerator.gradient_accumulation_steps
            active_loader = accelerator.skip_first_batches(train_loader, batches_to_skip)
            accelerator.print(f"Skipping {batches_to_skip} batches...")
        else:
            active_loader = train_loader

        num_batches = len(active_loader)
        pbar = tqdm(
            active_loader,
            total=num_batches,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}",
            leave=True,
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(pbar, start=1):
            with accelerator.accumulate(model):
                batch_loss_accum = 0.0
                valid_count = 0
                flow_list = []
                eval_list = []
                show_inner = cfg.get("show_batch_progress", True) and accelerator.is_local_main_process
                inner_bar = None
                sample_list = batch
                if show_inner:
                    inner_bar = tqdm(
                        total=len(sample_list),
                        leave=False,
                        position=1,
                        dynamic_ncols=True,
                        desc=f"Batch {batch_idx}/{num_batches} (size={len(sample_list)})",
                    )

                if flatten_steps:
                    flat_input_ids = []
                    flat_attention_mask = []
                    sample_meta = []
                    offset = 0

                    for sample in sample_list:
                        input_ids = sample["input_ids"].to(accelerator.device)
                        attention_mask = sample["attention_mask"].to(accelerator.device)
                        labels = sample["labels"].to(accelerator.device)
                        q_vals = sample["q_values"].to(accelerator.device)
                        v_vals = sample["v_values"].to(accelerator.device)
                        worst_ids = sample["worst_ids"].to(accelerator.device)
                        steps = sample["steps"]
                        traj_id = sample.get("traj_id")

                        step_len = labels.size(0)
                        if step_len == 0:
                            continue
                        if input_ids.size(0) != step_len:
                            min_len = min(step_len, input_ids.size(0))
                            input_ids = input_ids[:min_len]
                            attention_mask = attention_mask[:min_len]
                            labels = labels[:min_len]
                            q_vals = q_vals[:min_len]
                            v_vals = v_vals[:min_len]
                            worst_ids = worst_ids[:min_len]
                            steps = steps[:min_len]
                            step_len = min_len

                        flat_input_ids.append(input_ids)
                        flat_attention_mask.append(attention_mask)
                        sample_meta.append({
                            "start": offset,
                            "end": offset + step_len,
                            "labels": labels,
                            "q_vals": q_vals,
                            "v_vals": v_vals,
                            "worst_ids": worst_ids,
                            "steps": steps,
                            "traj_id": traj_id,
                        })
                        offset += step_len

                    if sample_meta:
                        flat_input_ids = torch.cat(flat_input_ids, dim=0)
                        flat_attention_mask = torch.cat(flat_attention_mask, dim=0)

                        logits, v_logits = model(flat_input_ids, flat_attention_mask)
                        with torch.no_grad():
                            logits_ref, _ = ref_model(flat_input_ids, flat_attention_mask)

                        denom = max(1, len(sample_meta))
                        loss_sum = None
                        for meta in sample_meta:
                            start = meta["start"]
                            end = meta["end"]
                            logits_s = logits[start:end]
                            v_logits_s = v_logits[start:end]
                            logits_ref_s = logits_ref[start:end]

                            loss, flow, kl, acc, eval_loss = flow_balance_loss(
                                logits_s, logits_ref_s, v_logits_s,
                                meta["labels"], meta["worst_ids"], meta["q_vals"], meta["v_vals"],
                                beta=cfg["beta"], gamma=cfg["gamma"]
                            )
                            with torch.no_grad():
                                probs = torch.softmax(logits_s, dim=-1).detach().cpu()
                                v_pos = v_logits_s.gather(-1, meta["labels"].unsqueeze(-1)).squeeze(-1).detach().cpu()
                                traj_id = meta["traj_id"]
                                if traj_id is not None:
                                    if traj_id not in updated_map:
                                        updated_map[traj_id] = {}
                                    steps = meta["steps"]
                                    for i in range(len(steps)):
                                        step_idx = int(steps[i])
                                        updated_map[traj_id][step_idx] = {
                                            "V_teacher": float(v_pos[i].item()),
                                            "strategy_probs": {name: float(probs[i, j].item()) for j, name in enumerate(id_to_name) if name},
                                        }

                            if loss_sum is None:
                                loss_sum = loss
                            else:
                                loss_sum = loss_sum + loss

                            batch_loss_accum += loss.item()
                            flow_list.append(flow.item())
                            eval_list.append(eval_loss.item())
                            valid_count += 1
                            if inner_bar is not None:
                                inner_bar.update(1)
                        if loss_sum is not None:
                            loss_norm = loss_sum / denom
                            accelerator.backward(loss_norm)
                else:
                    denom = max(1, len(sample_list))
                    for sample in sample_list:
                        input_ids = sample["input_ids"].to(accelerator.device)
                        attention_mask = sample["attention_mask"].to(accelerator.device)
                        labels = sample["labels"].to(accelerator.device)
                        q_vals = sample["q_values"].to(accelerator.device)
                        v_vals = sample["v_values"].to(accelerator.device)
                        worst_ids = sample["worst_ids"].to(accelerator.device)
                        steps = sample["steps"]
                        traj_id = sample.get("traj_id")

                        logits, v_logits = model(input_ids, attention_mask)
                        with torch.no_grad():
                            logits_ref, _ = ref_model(input_ids, attention_mask)

                        loss, flow, kl, acc, eval_loss = flow_balance_loss(
                            logits, logits_ref, v_logits, labels, worst_ids, q_vals, v_vals, beta=cfg["beta"], gamma=cfg["gamma"]
                        )
                        with torch.no_grad():
                            probs = torch.softmax(logits, dim=-1).detach().cpu()
                            v_pos = v_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).detach().cpu()
                            if traj_id is not None:
                                if traj_id not in updated_map:
                                    updated_map[traj_id] = {}
                                for i in range(len(steps)):
                                    step_idx = int(steps[i])
                                    updated_map[traj_id][step_idx] = {
                                        "V_teacher": float(v_pos[i].item()),
                                        "strategy_probs": {name: float(probs[i, j].item()) for j, name in enumerate(id_to_name) if name},
                                    }

                        loss_norm = loss / denom
                        accelerator.backward(loss_norm)

                        batch_loss_accum += loss.item()
                        flow_list.append(flow.item())
                        eval_list.append(eval_loss.item())
                        valid_count += 1
                        if inner_bar is not None:
                            inner_bar.update(1)

                if inner_bar is not None:
                    inner_bar.close()
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Checkpoint
                    if save_steps > 0 and global_step % save_steps == 0:
                        save_path = out_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(save_path)
                        if accelerator.is_main_process:
                            rotate_checkpoints(out_dir, cfg.get("save_total_limit", 2))

                # Logging
                if valid_count > 0:
                    avg_loss = batch_loss_accum / valid_count
                    avg_flow = sum(flow_list) / valid_count
                    avg_eval = sum(eval_list) / valid_count if eval_list else 0.0
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "flow": f"{avg_flow:.4f}", "eval": f"{avg_eval:.4f}"})
                    epoch_loss_total += batch_loss_accum
                    epoch_flow_total += sum(flow_list)
                    epoch_eval_total += sum(eval_list)
                    epoch_count_total += valid_count

        train_metrics = None
        if epoch_count_total > 0:
            train_metrics = {
                "train_loss": epoch_loss_total / epoch_count_total,
                "train_flow_mse": epoch_flow_total / epoch_count_total,
                "train_eval": epoch_eval_total / epoch_count_total,
            }

        do_eval = eval_enabled and eval_every_epochs > 0 and ((epoch + 1) % eval_every_epochs == 0)
        metrics = None
        if do_eval:
            accelerator.print(f"Validating Epoch {epoch+1}...")
            metrics = evaluate(
                model,
                ref_model,
                val_loader,
                cfg["beta"],
                cfg["gamma"],
                accelerator,
                use_ref_model=eval_use_ref_model,
                flatten_steps=flatten_steps,
            )
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1} Metrics: {metrics}")

        # Save Epoch Checkpoint (all ranks) if enabled
        if save_epoch_checkpoint:
            save_path = out_dir / f"checkpoint-{global_step}"
            accelerator.save_state(save_path)
            if accelerator.is_main_process:
                rotate_checkpoints(out_dir, cfg.get("save_total_limit", 2))

        # Log to file (independent of checkpoints)
        if log_file_path and (train_metrics is not None or metrics is not None):
            log_entry = {"epoch": epoch + 1, "step": global_step, "timestamp": str(datetime.now())}
            if train_metrics is not None:
                log_entry.update(train_metrics)
            if metrics is not None:
                log_entry.update(metrics)
            with open(log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # Save best checkpoint by metric
        if metrics is not None and save_best and best_metric in metrics:
            metric_val = float(metrics[best_metric])
            is_better = metric_val < best_value if best_mode == "min" else metric_val > best_value
            if is_better:
                best_value = metric_val
                best_path = out_dir / "best"
                if accelerator.is_main_process and best_path.exists():
                    shutil.rmtree(best_path)
                accelerator.wait_for_everyone()
                accelerator.save_state(best_path)
                if accelerator.is_main_process:
                    best_meta = {
                        "metric": best_metric,
                        "mode": best_mode,
                        "value": best_value,
                        "epoch": epoch + 1,
                        "step": global_step,
                        "timestamp": str(datetime.now()),
                    }
                    with open(out_dir / "best_metric.json", "w", encoding="utf-8") as f:
                        json.dump(best_meta, f)
                accelerator.wait_for_everyone()
        accelerator.wait_for_everyone()
    # Persist updated V_teacher and strategy_probs back to train_paths file (backup already made)
    if updated_map:
        # Load original JSONL
        lines: List[str] = train_path.read_text(encoding="utf-8").splitlines()
        updated_lines: List[str] = []
        for line in lines:
            if not line.strip():
                continue
            obj = json.loads(line)
            traj_id = obj.get("traj_id")
            if traj_id and traj_id in updated_map:
                per_traj = updated_map[traj_id]
                # update states-aligned arrays
                v_teacher = obj.get("V_teacher") or []
                strategy = obj.get("strategy") or []
                for step_idx, upd in per_traj.items():
                    # ensure length
                    while len(v_teacher) <= step_idx:
                        v_teacher.append(0.0)
                    v_teacher[step_idx] = upd.get("V_teacher", v_teacher[step_idx])
                    while len(strategy) <= step_idx:
                        strategy.append({})
                    strategy[step_idx] = upd.get("strategy_probs", strategy[step_idx])
                obj["V_teacher"] = v_teacher
                obj["strategy"] = strategy
            updated_lines.append(json.dumps(obj, ensure_ascii=False))
        train_path.write_text("\n".join(updated_lines), encoding="utf-8")

    accelerator.print("Training Finished!")

if __name__ == "__main__":
    main()
