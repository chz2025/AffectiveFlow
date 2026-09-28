#!/usr/bin/env python3
"""Benchmark end-to-end AFPO strategy-selection latency on saved checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_utils import load_eval_records, load_policy_model, predict_strategy  


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--eval-data",
        required=True,
    )
    parser.add_argument("--cfg", default="configs/train_emoflow.yaml")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency benchmark")

    rows = load_eval_records(Path(args.eval_data))
    if len(rows) < args.warmup + args.samples:
        raise ValueError("Not enough evaluation records for requested benchmark")

    bundle = load_policy_model(
        Path(args.checkpoint),
        Path(args.cfg),
        model_name_override=args.model_name,
    )

    def run_one(row: dict) -> None:
        predict_strategy(
            model_bundle=bundle,
            scene=str(row.get("scene", "")),
            description=str(row.get("description", "")),
            history=row.get("history") or [],
        )

    for row in rows[: args.warmup]:
        run_one(row)
    torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    for row in rows[args.warmup : args.warmup + args.samples]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_one(row)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000)

    result = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "warmup": args.warmup,
        "samples": args.samples,
        "scope": "tokenization + host-to-device + AFPO forward + result transfer",
        "mean_ms": statistics.fmean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "std_ms": statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
        "p95_ms": percentile(elapsed_ms, 0.95),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
        "raw_ms": elapsed_ms,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_ms"}, indent=2))


if __name__ == "__main__":
    main()
