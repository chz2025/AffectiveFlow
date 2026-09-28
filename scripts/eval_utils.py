#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STAGE_NAMES = ("Exploration", "Comforting", "Action")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for this script. Install project dependencies first."
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load_strategy_catalog(path: Path = Path("data/strategies.json")) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in data if isinstance(item, dict)]


def strategy_name_to_id(path: Path = Path("data/strategies.json")) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, item in enumerate(load_strategy_catalog(path)):
        name = item.get("strategy")
        if isinstance(name, str):
            mapping[name] = idx
    return mapping


def strategy_id_to_name(path: Path = Path("data/strategies.json")) -> List[str]:
    mapping = strategy_name_to_id(path)
    out = [""] * (max(mapping.values()) + 1 if mapping else 0)
    for name, idx in mapping.items():
        out[idx] = name
    return out


def render_history(history: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in history:
        role = str(turn.get("role", "")).capitalize()
        content = str(turn.get("content", "")).strip()
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text.strip())


def word_tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def stage_from_step(step_idx: int, total_steps: int) -> str:
    if total_steps <= 1:
        return STAGE_NAMES[0]
    ratio = step_idx / max(1, total_steps - 1)
    if ratio < (1.0 / 3.0):
        return STAGE_NAMES[0]
    if ratio < (2.0 / 3.0):
        return STAGE_NAMES[1]
    return STAGE_NAMES[2]


def extract_reference_transition(
    current_state: Dict[str, Any],
    next_state: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    current_hist = current_state.get("history") or []
    next_hist = next_state.get("history") or []
    if not isinstance(current_hist, list) or not isinstance(next_hist, list):
        return None, None
    extra = next_hist[len(current_hist) :]
    ref_supporter = None
    ref_seeker = None
    for turn in extra:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role == "supporter" and ref_supporter is None and isinstance(content, str):
            ref_supporter = content
        elif role == "seeker" and ref_seeker is None and isinstance(content, str):
            ref_seeker = content
    if ref_supporter is None and len(next_hist) >= 2:
        tail = next_hist[-2]
        if isinstance(tail, dict) and tail.get("role") == "supporter":
            content = tail.get("content")
            if isinstance(content, str):
                ref_supporter = content
    return ref_supporter, ref_seeker


def iter_step_records(
    path_samples: Sequence[Dict[str, Any]],
    source_name: str = "",
) -> Iterable[Dict[str, Any]]:
    id_counter = 0
    for sample in path_samples:
        actions = sample.get("actions") or []
        states = sample.get("states") or []
        if not isinstance(actions, list) or not isinstance(states, list):
            continue
        total_steps = len(actions)
        for local_step, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            t = action.get("t")
            chosen_strategy_id = action.get("chosen_strategy_id")
            if not isinstance(t, int) or not isinstance(chosen_strategy_id, int):
                continue
            if t < 0 or t + 1 >= len(states):
                continue
            cur_state = states[t]
            next_state = states[t + 1]
            if not isinstance(cur_state, dict) or not isinstance(next_state, dict):
                continue
            reference_response, reference_seeker = extract_reference_transition(cur_state, next_state)
            eval_id = sample.get("traj_id") or f"eval_{id_counter:06d}"
            stage = stage_from_step(local_step, total_steps)
            record = {
                "eval_id": f"{eval_id}::step{local_step}",
                "source_name": source_name,
                "traj_id": sample.get("traj_id"),
                "tree_id": sample.get("tree_id"),
                "scene": sample.get("scene", ""),
                "description": sample.get("description", ""),
                "state_index": t,
                "step_index": local_step,
                "total_steps": total_steps,
                "stage": stage,
                "history": cur_state.get("history") or [],
                "history_text": render_history(cur_state.get("history") or []),
                "reference_strategy_id": chosen_strategy_id,
                "reference_response": reference_response,
                "reference_seeker": reference_seeker,
            }
            id_counter += 1
            yield record


def load_eval_records(path: Path) -> List[Dict[str, Any]]:
    rows = load_jsonl(path)
    if rows and "eval_id" in rows[0]:
        return rows
    return list(iter_step_records(rows, source_name=path.stem))


def load_path_records_from_meta(
    meta_path: Path = Path("analyze/tree_paths_rel.json"),
    split: str = "val",
) -> List[Dict[str, Any]]:
    data = load_json(meta_path)
    key = f"tree_path_{split}"
    path_str = data.get(key)
    if not isinstance(path_str, str):
        raise KeyError(f"Missing {key} in {meta_path}")
    return load_jsonl(Path(path_str))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def corpus_bleu(predictions: Sequence[str], references: Sequence[str], max_order: int) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be aligned")
    clipped = [0] * max_order
    totals = [0] * max_order
    pred_len = 0
    ref_len = 0
    for pred, ref in zip(predictions, references):
        pred_toks = word_tokenize(pred)
        ref_toks = word_tokenize(ref)
        pred_len += len(pred_toks)
        ref_len += len(ref_toks)
        for n in range(1, max_order + 1):
            pred_counts = Counter(ngrams(pred_toks, n))
            ref_counts = Counter(ngrams(ref_toks, n))
            totals[n - 1] += sum(pred_counts.values())
            for gram, count in pred_counts.items():
                clipped[n - 1] += min(count, ref_counts.get(gram, 0))
    if pred_len == 0:
        return 0.0
    precisions: List[float] = []
    for num, den in zip(clipped, totals):
        if den == 0:
            precisions.append(0.0)
        else:
            precisions.append(num / den)
    if any(p == 0.0 for p in precisions):
        precisions = [max(p, 1e-12) for p in precisions]
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / max(pred_len, 1)))
    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_order)
    return 100.0 * score


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    scores: List[float] = []
    for pred, ref in zip(predictions, references):
        pred_toks = word_tokenize(pred)
        ref_toks = word_tokenize(ref)
        if not pred_toks or not ref_toks:
            scores.append(0.0)
            continue
        lcs = _lcs_len(pred_toks, ref_toks)
        prec = lcs / len(pred_toks)
        rec = lcs / len(ref_toks)
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append((2 * prec * rec) / (prec + rec))
    return 100.0 * mean(scores)


def meteor_exact(predictions: Sequence[str], references: Sequence[str]) -> float:
    scores: List[float] = []
    for pred, ref in zip(predictions, references):
        pred_toks = word_tokenize(pred)
        ref_toks = word_tokenize(ref)
        if not pred_toks or not ref_toks:
            scores.append(0.0)
            continue

        ref_positions: Dict[str, List[int]] = defaultdict(list)
        for idx, tok in enumerate(ref_toks):
            ref_positions[tok].append(idx)

        matches = 0
        chunks = 0
        last_match = -2
        used_positions: set[int] = set()

        for tok in pred_toks:
            candidate_positions = ref_positions.get(tok, [])
            pos = None
            for c in candidate_positions:
                if c not in used_positions:
                    pos = c
                    break
            if pos is None:
                continue
            used_positions.add(pos)
            matches += 1
            if pos != last_match + 1:
                chunks += 1
            last_match = pos

        if matches == 0:
            scores.append(0.0)
            continue
        precision = matches / len(pred_toks)
        recall = matches / len(ref_toks)
        f_mean = (10 * precision * recall) / max(recall + 9 * precision, 1e-12)
        penalty = 0.5 * ((chunks / matches) ** 3)
        scores.append((1 - penalty) * f_mean)
    return 100.0 * mean(scores)


def distinct_n(texts: Sequence[str], n: int) -> float:
    all_ngrams: List[Tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(ngrams(word_tokenize(text), n))
    if not all_ngrams:
        return 0.0
    return 100.0 * (len(set(all_ngrams)) / len(all_ngrams))


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return 100.0 * (correct / len(y_true))


def macro_f1_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
) -> float:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    f1s: List[float] = []
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append((2 * prec * rec) / (prec + rec))
    return 100.0 * mean(f1s)


def entropy_from_counts(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return ent


def group_by(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key, ""))].append(record)
    return groups


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_checkpoint_dir(run_path: Path) -> Path:
    if run_path.is_file():
        return run_path.parent
    model_file = run_path / "model.safetensors"
    if model_file.exists():
        return run_path
    best_dir = run_path / "best"
    if best_dir.exists() and (best_dir / "model.safetensors").exists():
        return best_dir
    ckpt_root = run_path / "checkpoints"
    if ckpt_root.exists():
        candidates = [d for d in ckpt_root.iterdir() if d.is_dir() and (d / "model.safetensors").exists()]
        if candidates:
            return sorted(candidates)[-1]
    flat_candidates = [d for d in run_path.iterdir() if d.is_dir() and (d / "model.safetensors").exists()]
    if flat_candidates:
        return sorted(flat_candidates)[-1]
    raise FileNotFoundError(f"Could not find a checkpoint under {run_path}")


def load_policy_model(
    checkpoint_path: Path,
    cfg_path: Path = Path("configs/train_emoflow.yaml"),
    model_name_override: Optional[str] = None,
):
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model evaluation requires torch, transformers, and safetensors."
        ) from exc

    import train_afpo as afpo_train

    cfg = afpo_train.load_cls_config(cfg_path)
    if model_name_override:
        cfg["model_name"] = model_name_override
    model = afpo_train.AFPOClassifier(
        cfg["model_name"],
        use_lora=cfg["use_lora"],
        lora_config=cfg,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        use_cache=cfg.get("use_cache"),
    )
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    state = load_file(str(checkpoint_dir / "model.safetensors"))
    load_result = model.load_state_dict(state, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    id_to_name = afpo_train.id_to_name_map(afpo_train.load_strategy_id_map())
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return {
        "config": cfg,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "strategy_names": id_to_name,
        "load_result": load_result,
    }


def encode_policy_input(
    tokenizer: Any,
    max_length: int,
    scene: str,
    description: str,
    history: Sequence[Dict[str, Any]],
) -> Tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for policy inference") from exc

    scene = scene or ""
    description = description or ""
    prefix = f"SCENE: {scene}\nDESC: {description}\nTask: Select the next support strategy id (0-7).\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer(
        render_history(history),
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos if eos is not None else 0)

    ids: List[int] = []
    if bos is not None:
        ids.append(bos)
    ids.extend(prefix_ids)
    ids.extend(suffix_ids)
    if eos is not None:
        ids.append(eos)

    if len(ids) > max_length:
        fixed_len = (1 if bos is not None else 0) + len(prefix_ids) + (1 if eos is not None else 0)
        avail = max_length - fixed_len
        if avail < 0:
            ids = ids[-max_length:]
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
    if len(ids) < max_length:
        pad_len = max_length - len(ids)
        ids.extend([pad] * pad_len)
        attn.extend([0] * pad_len)

    return torch.tensor([ids], dtype=torch.long), torch.tensor([attn], dtype=torch.long)


def predict_strategy(
    model_bundle: Dict[str, Any],
    scene: str,
    description: str,
    history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for policy inference") from exc

    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    cfg = model_bundle["config"]
    strategy_names = model_bundle["strategy_names"]
    device = model_bundle["device"]

    input_ids, attention_mask = encode_policy_input(
        tokenizer=tokenizer,
        max_length=int(cfg["max_length"]),
        scene=scene,
        description=description,
        history=history,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        logits, v_logits = model(input_ids, attention_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = log_probs + v_logits
        pred_id = int(torch.argmax(scores, dim=-1).item())
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        values = v_logits[0].detach().cpu().tolist()
        score_vals = scores[0].detach().cpu().tolist()

    pred_name = strategy_names[pred_id] if pred_id < len(strategy_names) else str(pred_id)
    return {
        "pred_strategy_id": pred_id,
        "pred_strategy_name": pred_name,
        "policy_probs": {
            strategy_names[i]: float(probs[i]) for i in range(min(len(strategy_names), len(probs))) if strategy_names[i]
        },
        "value_scores": {
            strategy_names[i]: float(values[i]) for i in range(min(len(strategy_names), len(values))) if strategy_names[i]
        },
        "decision_scores": {
            strategy_names[i]: float(score_vals[i]) for i in range(min(len(strategy_names), len(score_vals))) if strategy_names[i]
        },
    }


def randomize_pair_order(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    if random.random() < 0.5:
        return a, b, {"1": "a", "2": "b"}
    return b, a, {"1": "b", "2": "a"}
