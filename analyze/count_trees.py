#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def default_tree_path() -> Path:
    meta = Path("analyze/tree_paths.json")
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            tp = data.get("tree_path_train") or data.get("tree_path")
            if tp:
                return Path(tp)
        except Exception:
            pass
    return Path("data/processed/extes/Ex_Tree.jsonl")


def load_records(path: Path) -> List[Dict[str, Any]]:
    """
    Best-effort loader for json / jsonl / concatenated json objects.
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    def as_list(obj: Any) -> List[Dict[str, Any]]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        return []

    try:
        parsed = json.loads(text)
        recs = as_list(parsed)
        if recs:
            return recs
    except Exception:
        pass

    # stream decode multiple concatenated json objects
    out: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        out.extend(as_list(obj))
        idx = end
    if out:
        return out

    # fallback line-by-line jsonl
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            out.extend(as_list(obj))
        except Exception:
            continue
    return out


def count_paths_and_depth(tree: Dict[str, Any], depth: int = 0) -> Tuple[int, List[int]]:
    """Returns (path_count, leaf_depths)."""
    children_raw = tree.get("children") or []
    children = list(children_raw.values()) if isinstance(children_raw, dict) else list(children_raw)
    if not children:
        return 1, [depth]
    paths = 0
    depths: List[int] = []
    for ch in children:
        p, ds = count_paths_and_depth(ch, depth + 1)
        paths += p
        depths.extend(ds)
    return paths, depths


def main() -> None:
    ap = argparse.ArgumentParser(description="Count trees, path counts, and average leaf depth.")
    ap.add_argument("path", nargs="?", default=None, help="Tree file (.json or .jsonl).")
    args = ap.parse_args()

    tree_path = Path(args.path) if args.path else default_tree_path()
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")

    records = load_records(tree_path)
    trees: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        t = rec.get("tree") or rec.get("root")
        if t:
            trees.append(t)

    lines: List[str] = []
    lines.append(f"File: {tree_path}")
    lines.append(f"Total trees: {len(trees)}")

    for idx, t in enumerate(trees):
        path_count, leaf_depths = count_paths_and_depth(t, depth=0)
        avg_depth = sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0.0
        lines.append(f"  Tree {idx}: paths={path_count}, avg_depth={avg_depth:.2f}")

    summary = "\n".join(lines)
    print(summary)

    out_dir = tree_path.parent
    out_path = out_dir / f"{tree_path.stem}_count.text"
    out_path.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
