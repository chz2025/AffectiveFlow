#!/usr/bin/env python3
"""Estimate token usage and cost from an Ex_Tree prompt log."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import re
from pathlib import Path


ENTRY = re.compile(
    r"role=(?P<role>\w+) mode=(?P<mode>online|offline)\n"
    r"PROMPT:\n(?P<prompt>.*?)\nOUTPUT:\n(?P<output>.*?)\n---",
    re.DOTALL,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--input-per-million", type=float, default=0.15)
    parser.add_argument("--output-per-million", type=float, default=0.60)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8")
    entries = list(ENTRY.finditer(text))
    online = [item for item in entries if item.group("mode") == "online"]
    offline = [item for item in entries if item.group("mode") == "offline"]
    input_chars = sum(len(item.group("prompt")) for item in online)
    output_chars = sum(len(item.group("output")) for item in online)
    input_tokens = input_chars / args.chars_per_token
    output_tokens = output_chars / args.chars_per_token
    cost = (
        input_tokens * args.input_per_million
        + output_tokens * args.output_per_million
    ) / 1_000_000

    print(f"attempts={len(entries)} online={len(online)} offline={len(offline)}")
    print(f"online_input_chars={input_chars}")
    print(f"online_output_chars={output_chars}")
    print(f"estimated_input_tokens={input_tokens:.0f}")
    print(f"estimated_output_tokens={output_tokens:.0f}")
    print(f"estimated_successful_request_cost_usd={cost:.6f}")

    role_attempts = Counter(item.group("role") for item in entries)
    role_online = Counter(item.group("role") for item in online)
    role_input_chars = defaultdict(int)
    role_output_chars = defaultdict(int)
    for item in online:
        role = item.group("role")
        role_input_chars[role] += len(item.group("prompt"))
        role_output_chars[role] += len(item.group("output"))

    for role in sorted(role_attempts):
        successes = role_online[role]
        attempted = role_attempts[role]
        in_tokens = role_input_chars[role] / args.chars_per_token
        out_tokens = role_output_chars[role] / args.chars_per_token
        if successes:
            projected_in = in_tokens / successes * attempted
            projected_out = out_tokens / successes * attempted
        else:
            projected_in = projected_out = 0.0
        print(
            f"role={role} attempts={attempted} online={successes} "
            f"measured_input_tokens={in_tokens:.0f} "
            f"measured_output_tokens={out_tokens:.0f} "
            f"projected_full_input_tokens={projected_in:.0f} "
            f"projected_full_output_tokens={projected_out:.0f}"
        )

    if online:
        scale = len(entries) / len(online)
        full_cost = cost * scale
        print(f"estimated_all_online_cost_per_tree_usd={full_cost:.6f}")
        print(f"estimated_cost_per_1000_trees_usd={full_cost * 1000:.2f}")


if __name__ == "__main__":
    main()
