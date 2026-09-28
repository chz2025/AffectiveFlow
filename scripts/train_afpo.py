#!/usr/bin/env python3
"""
AFPO Classifier Training
- Loads trajectory paths JSON, lazily tokenizes each trajectory step.
- Supports LoRA fine-tuning, distributed training via Accelerate, checkpointing, and validation metrics.
"""
from __future__ import annotations
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
import json
import logging
import os
import shutil
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import yaml
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from peft import LoraConfig, get_peft_model, TaskType

NUM_CLASSES = 8
def render_history(history: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not role or not content:
            continue
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)
def load_strategy_id_map(path: Path = Path("data/strategies.json")) -> Dict[str, int]:
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
    if not name_to_id:
        return []
    max_idx = max(name_to_id.values())
    names = [""] * (max_idx + 1)
    for n, i in name_to_id.items():
        if 0 <= i < len(names):
            names[i] = n
    return names

def id_to_name_map(name_to_id: Dict[str, int]) -> List[str]:
    if not name_to_id:
        return []
    max_idx = max(name_to_id.values())
    names = [""] * (max_idx + 1)
    for n, i in name_to_id.items():
        if i >= 0 and i < len(names):
            names[i] = n
    return names

def load_samples(path: Path) -> List[Dict[str, Any]]:
    """
    Simple JSONL reader: one sample per line; if a line is a list, flatten it.
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
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

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
class LazyTokenizedDataset(Dataset):
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

        action_counts: Dict[int, int] = {}
        for traj in self.items:
            for act in traj["actions"]:
                a_id = act.get("chosen_strategy_id")
                if a_id is not None:
                    action_counts[int(a_id)] = action_counts.get(int(a_id), 0) + 1

        self.sample_weights: List[float] = []
        for traj in self.items:
            ids = [int(act["chosen_strategy_id"]) for act in traj["actions"] if act.get("chosen_strategy_id") is not None]
            if ids:
                weight = sum(1.0 / action_counts[a] for a in ids) / len(ids)
            else:
                weight = 1.0
            self.sample_weights.append(weight)

    def _build_prefix(self, scene: str, description: str) -> str:
        scene = scene or ""
        description = description or ""
        return f"SCENE: {scene}\nDESC: {description}\nTask: Select the next support strategy id (0-7).\n"

    def _get_prefix_ids(self, prefix: str) -> List[int]:
        if prefix in self._prefix_cache:
            return self._prefix_cache[prefix]

        ids = self.tokenizer.encode(prefix, add_special_tokens=False)

        self._prefix_cache[prefix] = ids
        return ids
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        actions = item["actions"]
        states = item["states"]

        v_vals_raw = item["v_teacher"] or [1.0] * len(states)   
        q_source = item["q_source"] or [0.0] * len(states)

        prefix_str = self._build_prefix(item.get("scene"), item.get("description"))
        prefix_ids = self._get_prefix_ids(prefix_str)

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = eos if eos is not None else 0

        texts = []
        labels = []
        q_vals = []
        v_vals = []
        worst_ids: List[int] = []
        steps: List[int] = []

        for act in actions:
            t = act.get("t")
            a_id = act.get("chosen_strategy_id")
            if t is None or a_id is None or t >= len(states):
                continue

            hist = states[t].get("history", [])
            text = render_history(hist)  
            texts.append(text)
            labels.append(int(a_id))

            q_vals.append(float(q_source[t]) if t < len(q_source) else 0.0)
            v_vals.append(float(v_vals_raw[t]) if t < len(v_vals_raw) else 1.0)
            worst_id: Optional[int] = None
            strat_list = item.get("strategy") or []
            if t < len(strat_list) and isinstance(strat_list[t], dict):
                probs = strat_list[t]
                if probs:
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
        suffix_enc = self.tokenizer(
            texts,
            add_special_tokens=False,   
            truncation=False,           
            padding=False,
            return_attention_mask=False,
            return_tensors=None
        )
        suffix_ids_list = suffix_enc["input_ids"]  

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

            if len(ids) > self.max_len:
                fixed_len = (1 if bos is not None else 0) + len(prefix_ids) + (1 if eos is not None else 0)
                avail = self.max_len - fixed_len
                if avail < 0:
                    ids = ids[-self.max_len:]
                else:
                    suffix_tail = suffix_ids[-avail:] if avail > 0 else []
                    ids = []
                    if bos is not None:
                        ids.append(bos)
                    ids.extend(prefix_ids)
                    ids.extend(suffix_tail)
                    if eos is not None:
                        ids.append(eos)

            attn = [1] * len(ids)

            if len(ids) < self.max_len:
                pad_len = self.max_len - len(ids)
                ids = ids + [pad] * pad_len
                attn = attn + [0] * pad_len

            input_ids_batch.append(ids)
            attn_mask_batch.append(attn)

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
    return [b for b in batch if b is not None]

class ClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)
class AFPOClassifier(nn.Module):
    def __init__(
        self,
        base_name: str,
        num_classes: int = NUM_CLASSES,
        use_lora: bool = False,
        lora_config: Dict | None = None,
        gradient_checkpointing: bool = True,
        use_cache: bool | None = None,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_name, trust_remote_code=True)
        
        if use_lora:
            peft_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_config.get("lora_rank", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                lora_dropout=lora_config.get("lora_dropout", 0.05),
                bias="none",
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","embed_tokens","in_proj_qkv","in_proj_a","in_proj_b","in_proj_z","out_proj"]
            )
            self.backbone = get_peft_model(self.backbone, peft_cfg)
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        elif hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

        if hasattr(self.backbone, "config"):
            text_config = self.backbone.config.get_text_config()
            if use_cache is None:
                text_config.use_cache = not gradient_checkpointing
            else:
                text_config.use_cache = bool(use_cache)
        hidden = self.backbone.config.get_text_config().hidden_size
        self.head = ClassifierHead(hidden, num_classes)
        self.head.float()
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
            nn.Softplus()
        )
        self.value_head.float()

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
        last_hidden = hidden_states.gather(dim=1, index=idx).squeeze(1)
        last_hidden = last_hidden.float()
        logits = self.head(last_hidden)
        v_logits = self.value_head(last_hidden)
        return logits, v_logits
    
    def print_trainable_parameters(self):
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.parameters())
            logging.info(f"trainable params: {trainable} || all params: {all_params} || trainable%: {100 * trainable / all_params:.4f}")

def flow_balance_loss(logits, logits_ref, v_logits, actions, worst_ids, q_values, v_teacher, beta=0.1, gamma=1.0):
    logprobs = torch.log_softmax(logits, dim=-1)

    lp = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    if logits_ref is not None:
        logprobs_ref = torch.log_softmax(logits_ref, dim=-1)
        lp_ref = logprobs_ref.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        kl = (lp - lp_ref).mean()
    else:
        kl = torch.zeros((), device=logits.device)
    v_pos = v_logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    v_neg = v_logits.gather(-1, worst_ids.unsqueeze(-1)).squeeze(-1)

    T = lp.size(0)
    device = lp.device

    preds = logits.argmax(dim=-1)
    correct = (preds == actions).float().sum()
    accuracy = correct / T

    prefix = torch.cat([torch.zeros(1, device=device), torch.cumsum(lp, dim=0)])
    idx = torch.arange(T, device=device)
    m_idx, n_idx = torch.meshgrid(idx, idx, indexing="ij")
    
    delta_rho = prefix[n_idx + 1] - prefix[m_idx] 
    flow = q_values * v_pos
    logF = torch.log(torch.clamp(flow, min=1e-6))
    logF_diff = logF[n_idx] - logF[m_idx]

    tri_mask = torch.triu(torch.ones_like(delta_rho), diagonal=1) 
    flow_err = (logF_diff - delta_rho) * tri_mask
    
    pair_count = tri_mask.sum().clamp(min=1.0)
    flow_mse = (flow_err.pow(2).sum() / pair_count)

    eval_loss = torch.relu(gamma - (v_pos - v_neg)).mean()
    total_loss = flow_mse + beta * torch.relu(kl) + eval_loss

    return total_loss, flow_mse, kl, accuracy, eval_loss

def evaluate(model, ref_model, loader, beta, gamma, accelerator, use_ref_model: bool, flatten_steps: bool):
    model.eval()

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

    all_loss = accelerator.reduce(total_loss, reduction="sum")
    all_flow = accelerator.reduce(total_flow, reduction="sum")
    all_kl = accelerator.reduce(total_kl, reduction="sum")
    all_acc = accelerator.reduce(total_acc, reduction="sum")
    all_eval = accelerator.reduce(total_eval, reduction="sum")
    all_count = accelerator.reduce(total_samples, reduction="sum")

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

def rotate_checkpoints(output_dir: Path, limit: int):
    if not output_dir.exists(): return
    checkpoints = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")], key=lambda x: os.path.getmtime(x))
    if len(checkpoints) > limit:
        for ckpt in checkpoints[:-limit]:
            shutil.rmtree(ckpt)
def resolve_split_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    meta = Path("analyze/tree_paths_rel.json")
    train_path = cfg.get("data_path")

    if meta.exists():
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            train_path = info.get("tree_path_train") 
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

    train_path_p = normalize(train_path, "data/processed/extes/paths/Ex_Tree_train_paths.jsonl")
    val_path_p = Path("__auto_split_validation__.jsonl")

    return {"train": train_path_p, "val": val_path_p}

def main():
    cfg = load_cls_config(Path(os.environ.get("AFPO_CONFIG", "configs/train_emoflow.yaml")))
    paths = resolve_split_paths(cfg)
    train_path = paths["train"]
    val_path = paths["val"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    out_dir = Path(cfg["output_dir"]) / ts

    project_config = ProjectConfiguration(project_dir=str(out_dir), automatic_checkpoint_naming=False)
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        project_config=project_config,
         kwargs_handlers=[timeout_kwargs]
    )
    set_seed(int(cfg["seed"]))

    log_file_path = None
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = out_dir / "training_log.jsonl"

        text_log_path = Path(cfg.get("log_path", "logs/afpo_cls.log"))
        text_log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[logging.FileHandler(text_log_path), logging.StreamHandler()],
            force=True,
        )
        logging.info(f"Output Directory: {out_dir}")
        logging.info(f"Logging to: {log_file_path}")
        logging.info(f"Text log: {text_log_path}")

    def log_main(msg: str) -> None:
        if accelerator.is_main_process:
            logging.info(msg)
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

    log_main("Loading Data...")
    train_samples = load_samples(train_path)
    if val_path.exists():
        val_samples = load_samples(val_path)
    else:
        if len(train_samples) < 2:
            raise ValueError("At least two training trajectories are required")
        rng = random.Random(int(cfg.get("seed", 1)))
        rng.shuffle(train_samples)
        val_count = max(1, int(round(len(train_samples) * float(cfg.get("val_ratio", 0.1)))))
        val_count = min(val_count, len(train_samples) - 1)
        val_samples = train_samples[:val_count]
        train_samples = train_samples[val_count:]
    strat_id_map = load_strategy_id_map()
    id_to_name = id_to_name_map(strat_id_map)
    updated_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_main("Initializing Datasets (Lazy)...")
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

    train_sampler = WeightedRandomSampler(
        train_ds.sample_weights, num_samples=len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    log_main("Initializing Models...")
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

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    ref_model.to(accelerator.device)

    global_step = 0
    start_epoch = 0
    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt:
        log_main(f"Resuming from {resume_ckpt}")
        accelerator.load_state(resume_ckpt)
        try:
            global_step = int(Path(resume_ckpt).name.split("-")[-1])
            start_epoch = global_step // num_update_steps_per_epoch
        except ValueError:
            pass

    log_main(f"Starting training from Epoch {start_epoch}")
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
            log_main(f"Skipping {batches_to_skip} batches...")
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
                        logits_ref = None
                        if cfg["beta"] > 0:
                            with torch.no_grad():
                                logits_ref, _ = ref_model(flat_input_ids, flat_attention_mask)

                        denom = max(1, len(sample_meta))
                        loss_sum = None
                        for meta in sample_meta:
                            start = meta["start"]
                            end = meta["end"]
                            logits_s = logits[start:end]
                            v_logits_s = v_logits[start:end]
                            logits_ref_s = logits_ref[start:end] if logits_ref is not None else None

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
                        logits_ref = None
                        if cfg["beta"] > 0:
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
                    
                    if save_steps > 0 and global_step % save_steps == 0:
                        save_path = out_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(save_path)
                        if accelerator.is_main_process:
                            rotate_checkpoints(out_dir, cfg.get("save_total_limit", 2))

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
            log_main(f"Validating Epoch {epoch+1}...")
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
                logging.info(f"Epoch {epoch+1} Metrics: {metrics}")

        if save_epoch_checkpoint:
            save_path = out_dir / f"checkpoint-{global_step}"
            accelerator.save_state(save_path)
            if accelerator.is_main_process:
                rotate_checkpoints(out_dir, cfg.get("save_total_limit", 2))

        if log_file_path and (train_metrics is not None or metrics is not None):
            log_entry = {"epoch": epoch + 1, "step": global_step, "timestamp": str(datetime.now())}
            if train_metrics is not None:
                log_entry.update(train_metrics)
            if metrics is not None:
                log_entry.update(metrics)
            with open(log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

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
    if updated_map:
        lines: List[str] = train_path.read_text(encoding="utf-8").splitlines()
        updated_lines: List[str] = []
        for line in lines:
            if not line.strip():
                continue
            obj = json.loads(line)
            traj_id = obj.get("traj_id")
            if traj_id and traj_id in updated_map:
                per_traj = updated_map[traj_id]
                v_teacher = obj.get("V_teacher") or []
                strategy = obj.get("strategy") or []
                for step_idx, upd in per_traj.items():
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

    log_main("Training Finished!")

if __name__ == "__main__":
    main()
