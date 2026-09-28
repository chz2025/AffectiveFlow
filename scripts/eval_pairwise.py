#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_utils import load_jsonl, randomize_pair_order, save_json, write_jsonl


JUDGE_PROMPT = """You are an impartial evaluator for emotional support conversation.

Background: {background}
Dialogue Context: {dialogue_context}
Candidate 1: {candidate_1}
Candidate 2: {candidate_2}

Rubric:
Fluency: prefer the response that is more natural, clear, and well-formed in language.
Identification: prefer the response that better identifies the seeker's emotions, concerns, and core needs.
Comfort: prefer the response that offers stronger emotional support and appropriate reassurance with a fitting tone.
Suggest: prefer the response that provides more useful, feasible next-step guidance when appropriate, without being premature or prescriptive.
Overall: prefer the response that is better on balance for this turn, considering the above dimensions.

Output JSON only:
{{
  "fluency": "1|2|tie",
  "identification": "1|2|tie",
  "comfort": "1|2|tie",
  "suggest": "1|2|tie",
  "overall": "1|2|tie"
}}
"""


def build_client(model: str, api_base: str, api_key_env: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Pairwise judging requires the openai package.") from exc
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env: {api_key_env}")
    return OpenAI(base_url=api_base, api_key=api_key), model


def parse_judge_json(text: str) -> Dict[str, str]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"Judge output is not valid JSON: {text}")
    data = json.loads(match.group(0))
    out: Dict[str, str] = {}
    for key in ["fluency", "identification", "comfort", "suggest", "overall"]:
        value = str(data.get(key, "tie")).strip().lower()
        if value not in {"1", "2", "tie"}:
            value = "tie"
        out[key] = value
    return out


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = ["fluency", "identification", "comfort", "suggest", "overall"]
    agg: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        wins = sum(1 for row in rows if row["result"].get(metric) == "a")
        losses = sum(1 for row in rows if row["result"].get(metric) == "b")
        ties = sum(1 for row in rows if row["result"].get(metric) == "tie")
        total = max(1, len(rows))
        agg[metric] = {
            "win": 100.0 * wins / total,
            "tie": 100.0 * ties / total,
            "lose": 100.0 * losses / total,
        }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Pairwise LLM judging or human template export.")
    ap.add_argument("--file-a", required=True, help="JSONL with model A outputs.")
    ap.add_argument("--file-b", required=True, help="JSONL with model B outputs.")
    ap.add_argument("--field-a", default="generated_response", help="Response field in file A.")
    ap.add_argument("--field-b", default="generated_response", help="Response field in file B.")
    ap.add_argument("--out-raw", default="output/paper_eval/pairwise_raw.jsonl")
    ap.add_argument("--out-summary", default="output/paper_eval/pairwise_summary.json")
    ap.add_argument("--human-template", default=None, help="Optional JSONL path for manual annotation export.")
    ap.add_argument("--judge-model", default=None, help="OpenAI-compatible judge model.")
    ap.add_argument("--judge-api-base", default="https://openrouter.ai/api/v1")
    ap.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows_a = {row["eval_id"]: row for row in load_jsonl(Path(args.file_a)) if "eval_id" in row}
    rows_b = {row["eval_id"]: row for row in load_jsonl(Path(args.file_b)) if "eval_id" in row}
    shared_ids = sorted(set(rows_a) & set(rows_b))
    if args.limit is not None:
        shared_ids = shared_ids[: args.limit]

    judge_client = None
    if args.judge_model:
        judge_client = build_client(args.judge_model, args.judge_api_base, args.judge_api_key_env)

    raw_rows: List[Dict[str, Any]] = []
    human_rows: List[Dict[str, Any]] = []
    for eval_id in shared_ids:
        a = rows_a[eval_id]
        b = rows_b[eval_id]
        cand_1, cand_2, slot_map = randomize_pair_order(a, b)
        cand_1_text = a.get(args.field_a, "") if slot_map["1"] == "a" else b.get(args.field_b, "")
        cand_2_text = a.get(args.field_a, "") if slot_map["2"] == "a" else b.get(args.field_b, "")
        prompt = JUDGE_PROMPT.format(
            background=a.get("description", ""),
            dialogue_context=a.get("history_text", ""),
            candidate_1=cand_1_text,
            candidate_2=cand_2_text,
        )
        result: Dict[str, str]
        raw_output = None
        if judge_client is None:
            result = {k: "tie" for k in ["fluency", "identification", "comfort", "suggest", "overall"]}
        else:
            client, model_name = judge_client
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            raw_output = completion.choices[0].message.content.strip()
            parsed = parse_judge_json(raw_output)
            result = {}
            for metric, choice in parsed.items():
                if choice == "tie":
                    result[metric] = "tie"
                else:
                    result[metric] = slot_map[choice]
        raw_rows.append(
            {
                "eval_id": eval_id,
                "candidate_a": a.get(args.field_a, ""),
                "candidate_b": b.get(args.field_b, ""),
                "prompt": prompt,
                "raw_output": raw_output,
                "result": result,
            }
        )
        if args.human_template:
            human_rows.append(
                {
                    "eval_id": eval_id,
                    "description": a.get("description", ""),
                    "history_text": a.get("history_text", ""),
                    "candidate_1": cand_1_text,
                    "candidate_2": cand_2_text,
                    "candidate_1_source": slot_map["1"],
                    "candidate_2_source": slot_map["2"],
                }
            )

    summary = {
        "file_a": args.file_a,
        "file_b": args.file_b,
        "judge_model": args.judge_model,
        "num_pairs": len(raw_rows),
        "metrics": aggregate(raw_rows),
    }

    write_jsonl(Path(args.out_raw), raw_rows)
    save_json(Path(args.out_summary), summary)
    if args.human_template:
        write_jsonl(Path(args.human_template), human_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    random.seed(7)
    main()
