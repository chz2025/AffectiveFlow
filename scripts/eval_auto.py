#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval_utils import (
    STAGE_NAMES,
    accuracy_score,
    corpus_bleu,
    distinct_n,
    group_by,
    load_eval_records,
    load_jsonl,
    load_policy_model,
    load_strategy_catalog,
    macro_f1_score,
    mean,
    meteor_exact,
    predict_strategy,
    rouge_l,
    save_json,
    try_load_yaml,
    write_jsonl,
)


def maybe_build_supporter_llm(cfg_path: Path, offline: bool):
    try:
        import build_ex_tree as ex_tree
    except ImportError as exc:
        raise RuntimeError("Could not import scripts/build_ex_tree.py") from exc

    cfg = try_load_yaml(cfg_path)
    prompts = ex_tree.load_prompts(Path("data/prompt.json"))
    strategies = load_strategy_catalog()
    strategy_detail_map = {item["strategy"]: item.get("strategy_detail", "") for item in strategies if "strategy" in item}
    supporter_cfg = (cfg.get("models") or {}).get("supporter", {})
    llm = ex_tree.LLMInterface(supporter_cfg, "supporter", offline=offline)
    return llm, prompts, strategy_detail_map, ex_tree


def maybe_build_ppl_scorer(model_name_or_path: Optional[str]):
    if not model_name_or_path:
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("PPL scoring requires torch and transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "device": device}


def score_ppl(scorer: Optional[Dict[str, Any]], texts: List[str]) -> Optional[float]:
    if scorer is None:
        return None
    import torch

    model = scorer["model"]
    tokenizer = scorer["tokenizer"]
    device = scorer["device"]

    losses: List[float] = []
    total_tokens = 0
    for text in texts:
        text = text.strip()
        if not text:
            continue
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        token_count = int(attention_mask.sum().item())
        if token_count <= 1:
            continue
        losses.append(float(out.loss.item()) * token_count)
        total_tokens += token_count
    if total_tokens == 0:
        return None
    return float(torch.exp(torch.tensor(sum(losses) / total_tokens)).item())


def summarize_auto_metrics(rows: List[Dict[str, Any]], ppl_value: Optional[float]) -> Dict[str, Any]:
    strat_rows = [row for row in rows if row.get("reference_strategy_id") is not None and row.get("pred_strategy_id") is not None]
    y_true = [int(row["reference_strategy_id"]) for row in strat_rows]
    y_pred = [int(row["pred_strategy_id"]) for row in strat_rows]
    labels = sorted(set(y_true) | set(y_pred))

    text_rows = [row for row in rows if row.get("generated_response") and row.get("reference_response")]
    preds = [str(row["generated_response"]) for row in text_rows]
    refs = [str(row["reference_response"]) for row in text_rows]

    metrics = {
        "num_strategy_examples": len(strat_rows),
        "num_text_examples": len(text_rows),
        "strategy_acc": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "strategy_f1": macro_f1_score(y_true, y_pred, labels=labels) if y_true else 0.0,
        "bleu2": corpus_bleu(preds, refs, max_order=2) if preds else 0.0,
        "bleu4": corpus_bleu(preds, refs, max_order=4) if preds else 0.0,
        "rouge_l": rouge_l(preds, refs) if preds else 0.0,
        "meteor": meteor_exact(preds, refs) if preds else 0.0,
        "dist1": distinct_n(preds, 1) if preds else 0.0,
        "dist2": distinct_n(preds, 2) if preds else 0.0,
        "ppl": ppl_value,
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-style automatic evaluation for AFlo w.")
    ap.add_argument("--eval-data", required=True, help="Step-level eval JSONL or path-level *_paths.jsonl file.")
    ap.add_argument("--checkpoint", required=True, help="Training run directory or checkpoint directory.")
    ap.add_argument("--cfg", default="configs/train_emoflow.yaml", help="Training config path.")
    ap.add_argument("--model-name", default=None, help="Override cfg's model_name (e.g. to evaluate a checkpoint trained with a different backbone than the one currently in --cfg).")
    ap.add_argument("--out-preds", default="output/paper_eval/auto_predictions.jsonl", help="Where to save per-example predictions.")
    ap.add_argument("--out-metrics", default="output/paper_eval/auto_metrics.json", help="Where to save aggregated metrics.")
    ap.add_argument("--generate-responses", action="store_true", help="Generate supporter responses and compute text metrics.")
    ap.add_argument("--offline-supporter", action="store_true", help="Use build_ex_tree offline responder instead of API.")
    ap.add_argument("--ppl-model", default=None, help="Optional fixed LM scorer path/name for perplexity.")
    args = ap.parse_args()

    eval_path = Path(args.eval_data)
    rows = load_eval_records(eval_path)

    model_bundle = load_policy_model(Path(args.checkpoint), Path(args.cfg), model_name_override=args.model_name)
    supporter = None
    prompts = None
    strategy_detail_map = None
    ex_tree = None
    if args.generate_responses:
        supporter, prompts, strategy_detail_map, ex_tree = maybe_build_supporter_llm(Path(args.cfg), args.offline_supporter)
    ppl_scorer = maybe_build_ppl_scorer(args.ppl_model)

    outputs: List[Dict[str, Any]] = []
    for row in rows:
        pred = predict_strategy(
            model_bundle=model_bundle,
            scene=str(row.get("scene", "")),
            description=str(row.get("description", "")),
            history=row.get("history") or [],
        )
        out_row = dict(row)
        out_row.update(pred)
        if supporter is not None and prompts is not None and strategy_detail_map is not None and ex_tree is not None:
            chosen = pred["pred_strategy_name"]
            prompt = prompts["supporter prompt"].format(
                strategy=chosen,
                strategy_detail=strategy_detail_map.get(chosen, ""),
                chat_history=ex_tree.format_chat_history(row.get("history") or []),
            )
            generated_response = supporter.generate(prompt).strip()
            out_row["generated_response"] = generated_response
        outputs.append(out_row)

    ppl_value = None
    if ppl_scorer is not None:
        pred_texts = [str(row.get("generated_response", "")).strip() for row in outputs if row.get("generated_response")]
        ppl_value = score_ppl(ppl_scorer, pred_texts)

    overall = summarize_auto_metrics(outputs, ppl_value=ppl_value)
    by_stage = {
        stage: summarize_auto_metrics(group, ppl_value=None)
        for stage, group in group_by(outputs, "stage").items()
        if stage in STAGE_NAMES
    }

    summary = {
        "eval_data": str(eval_path),
        "checkpoint": str(args.checkpoint),
        "generate_responses": bool(args.generate_responses),
        "overall": overall,
        "by_stage": by_stage,
        "sample_prediction": outputs[0] if outputs else None,
    }

    write_jsonl(Path(args.out_preds), outputs)
    save_json(Path(args.out_metrics), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
