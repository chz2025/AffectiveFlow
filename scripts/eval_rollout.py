#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from eval_utils import (
    STAGE_NAMES,
    entropy_from_counts,
    load_json,
    load_policy_model,
    mean,
    predict_strategy,
    render_history,
    save_json,
    try_load_yaml,
    write_jsonl,
)


def load_seed_samples(path: Path) -> List[Dict[str, Any]]:
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


def build_role_env(cfg_path: Path, offline: bool):
    import build_ex_tree as ex_tree

    cfg = try_load_yaml(cfg_path)
    prompts = ex_tree.load_prompts(Path("data/prompt.json"))
    strategies = load_json(Path("data/strategies.json"))
    eval_metrics = load_json(Path("data/evaluation_metrics.json"))
    builder = ex_tree.MCTSBuilder(
        cfg=cfg,
        prompts=prompts,
        strategies=strategies,
        evaluation_metrics=eval_metrics,
        offline=offline,
    )
    return builder, cfg


def initial_seeker_message(sample: Dict[str, Any]) -> str:
    memory = sample.get("memory") or []
    if memory and isinstance(memory, list) and isinstance(memory[0], dict):
        first = next(iter(memory[0].values()), "")
        if isinstance(first, str) and first.strip():
            return first.strip()
    return str(sample.get("scene", "")).strip() or "Hello."


def stage_from_turn(turn_idx: int, total_turns: int) -> str:
    if total_turns <= 1:
        return STAGE_NAMES[0]
    ratio = turn_idx / max(1, total_turns - 1)
    if ratio < (1.0 / 3.0):
        return STAGE_NAMES[0]
    if ratio < (2.0 / 3.0):
        return STAGE_NAMES[1]
    return STAGE_NAMES[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-style rollout analysis for AFlo w.")
    ap.add_argument("--seed-data", required=True, help="External evaluation JSONL")
    ap.add_argument("--checkpoint", required=True, help="Training run dir or checkpoint dir.")
    ap.add_argument("--cfg", default="configs/train_emoflow.yaml", help="Training config path.")
    ap.add_argument("--offline", action="store_true", help="Use offline role agents instead of API calls.")
    ap.add_argument("--max-turns", type=int, default=8, help="Maximum supporter turns per generated dialogue.")
    ap.add_argument("--limit", type=int, default=None, help="Optional sample cap.")
    ap.add_argument("--out-dialogues", default="output/paper_eval/rollout_dialogues.jsonl")
    ap.add_argument("--out-summary", default="output/paper_eval/rollout_summary.json")
    args = ap.parse_args()

    samples = load_seed_samples(Path(args.seed_data))
    if args.limit is not None:
        samples = samples[: args.limit]

    model_bundle = load_policy_model(Path(args.checkpoint), Path(args.cfg))
    builder, cfg = build_role_env(Path(args.cfg), args.offline)

    all_dialogues: List[Dict[str, Any]] = []
    reward_by_turn: Dict[int, List[float]] = {}
    stage_scores: Dict[str, List[float]] = {name: [] for name in STAGE_NAMES}
    abs_adjacent_changes: List[float] = []
    strategy_counts_by_turn: Dict[int, Dict[str, int]] = {}

    max_turns = int(args.max_turns)
    for idx, sample in enumerate(samples):
        history = [{"role": "seeker", "content": initial_seeker_message(sample)}]
        per_turn_rewards: List[float] = []
        per_turn_metrics: List[Dict[str, Any]] = []
        chosen_strategies: List[str] = []

        for turn_idx in range(max_turns):
            pred = predict_strategy(
                model_bundle=model_bundle,
                scene=str(sample.get("scene", "")),
                description=str(sample.get("description", "")),
                history=history,
            )
            chosen_strategy = pred["pred_strategy_name"]
            chosen_strategies.append(chosen_strategy)

            supporter_resp = builder.generate_supporter(chosen_strategy, history).strip()
            history.append({"role": "supporter", "content": supporter_resp, "strategy": chosen_strategy})
            reward_scores = builder.score_reward(history)
            direct_reward = (
                sum(
                    float(reward_scores.get(name, 0.0)) * float(builder.weights.get(name, 0.0))
                    for name in builder.weights
                )
                / max(1.0, float(builder.reward_norm))
            )
            per_turn_rewards.append(direct_reward)
            per_turn_metrics.append(
                {
                    "turn": turn_idx,
                    "strategy": chosen_strategy,
                    "reward_scores": reward_scores,
                    "direct_reward": direct_reward,
                    "supporter_response": supporter_resp,
                }
            )

            seeker_resp = builder.generate_seeker(sample, history).strip()
            history.append({"role": "seeker", "content": seeker_resp})
            seeker_trim = seeker_resp.strip()
            if seeker_trim == "</end/>" or seeker_trim.endswith("<ok>"):
                break

        for turn_idx, reward in enumerate(per_turn_rewards):
            reward_by_turn.setdefault(turn_idx, []).append(reward)
            stage_scores[stage_from_turn(turn_idx, len(per_turn_rewards))].append(reward)
            strategy_counts_by_turn.setdefault(turn_idx, {})
            strategy_counts_by_turn[turn_idx][chosen_strategies[turn_idx]] = (
                strategy_counts_by_turn[turn_idx].get(chosen_strategies[turn_idx], 0) + 1
            )
            if turn_idx > 0:
                abs_adjacent_changes.append(abs(per_turn_rewards[turn_idx] - per_turn_rewards[turn_idx - 1]))

        all_dialogues.append(
            {
                "sample_index": idx,
                "scene": sample.get("scene", ""),
                "description": sample.get("description", ""),
                "history_text": render_history(history),
                "history": history,
                "turn_metrics": per_turn_metrics,
                "strategies": chosen_strategies,
                "reward_trajectory": per_turn_rewards,
            }
        )

    summary = {
        "seed_data": args.seed_data,
        "checkpoint": args.checkpoint,
        "offline": bool(args.offline),
        "num_dialogues": len(all_dialogues),
        "avg_dialogue_turns": mean([len(d["reward_trajectory"]) for d in all_dialogues]),
        "avg_reward_by_turn": {str(k + 1): mean(v) for k, v in sorted(reward_by_turn.items())},
        "stage_effectiveness": {stage: mean(vals) for stage, vals in stage_scores.items()},
        "avg_adjacent_turn_abs_change": mean(abs_adjacent_changes),
        "strategy_entropy_by_turn": {
            str(k + 1): entropy_from_counts(v) for k, v in sorted(strategy_counts_by_turn.items())
        },
        "strategy_distribution_by_turn": {
            str(k + 1): v for k, v in sorted(strategy_counts_by_turn.items())
        },
        "sample_dialogue": all_dialogues[0] if all_dialogues else None,
    }

    write_jsonl(Path(args.out_dialogues), all_dialogues)
    save_json(Path(args.out_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
