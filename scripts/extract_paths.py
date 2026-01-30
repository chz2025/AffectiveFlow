#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Training Trajectories from MCTS Trees

This module extracts root-to-leaf paths from MCTS exploration trees generated
during the dialogue planning phase. Each extracted path represents a complete
dialogue trajectory suitable for preference-based learning.

Features:
  - Flexible path length filtering (minimum/maximum nodes per trajectory)
  - Automatic tree path resolution from metadata or defaults
  - Support for JSON, JSONL, and mixed format trees
  - Outputs standardized JSONL for training

Configuration:
  - Path length constraints: configs/train_emoflow.yaml:[path_extraction]
  - Tree source: analyze/tree_paths.json or data/processed/extes/Ex_Tree.jsonl

Output Format:
  Each line is a JSON object representing one root-to-leaf trajectory:
  {
    "traj_id": "unique-id",
    "scene": "...",
    "description": "...",
    "states": [...],      # Dialogue history at each step
    "actions": [...],     # Strategy selections
    "Q": [...],           # Trajectory values
    "V_teacher": [...],   # State value estimates
    "strategy": [...]     # Strategy probability distributions
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


# ---------------- helpers ---------------- #
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


def load_minmax() -> Tuple[int, int | None]:
    min_len, max_len = 1, None
    cfg_path = Path("configs/train_emoflow.yaml")
    if not cfg_path.exists():
        return min_len, max_len
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return min_len, max_len
    if not isinstance(cfg, dict):
        return min_len, max_len
    pe = cfg.get("path_extraction") or {}
    if isinstance(pe, dict):
        if pe.get("min_path_length") is not None:
            try:
                min_len = int(pe["min_path_length"])
            except Exception:
                pass
        if pe.get("max_path_length") is not None:
            try:
                max_len = int(pe["max_path_length"])
            except Exception:
                pass
    return min_len, max_len


def load_records(path: Path) -> List[Dict[str, Any]]:
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


def load_strategy_map(path: Path = Path("data/strategies.json")) -> Tuple[Dict[str, int], Dict[str, str]]:
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    name_to_id: Dict[str, int] = {}
    name_to_abbr: Dict[str, str] = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        name = item.get("strategy")
        abbr = item.get("abbreviation")
        if name:
            name_to_id[name] = idx
            if abbr:
                name_to_abbr[name] = abbr
    return name_to_id, name_to_abbr


def build_history(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role and content:
            out.append({"role": role, "content": content})
    return out


def mean_reward(node: Dict[str, Any]) -> float:
    mr = node.get("mean_reward")
    if mr is not None:
        return mr
    tot = node.get("total_reward")
    vis = node.get("visits")
    try:
        if tot is not None and vis:
            return float(tot) / float(vis)
    except Exception:
        pass
    return 0.0


def max_action_q(node: Dict[str, Any]) -> Any:
    stats = node.get("action_stats") or {}
    if not isinstance(stats, dict) or not stats:
        return None
    q_vals = []
    for entry in stats.values():
        if isinstance(entry, dict) and entry.get("q") is not None:
            try:
                q_vals.append(float(entry["q"]))
            except Exception:
                continue
    if q_vals:
        return max(q_vals)
    return None


def enumerate_paths(tree: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    paths: List[List[Dict[str, Any]]] = []

    def dfs(node: Dict[str, Any], acc: List[Dict[str, Any]]):
        acc.append(node)
        children_raw = node.get("children") or []
        children = list(children_raw.values()) if isinstance(children_raw, dict) else list(children_raw)
        if not children:
            paths.append(list(acc))
        else:
            for ch in children:
                dfs(ch, acc)
        acc.pop()

    dfs(tree, [])
    return paths


def path_to_sample(path: List[Dict[str, Any]], rec: Dict[str, Any], strat_id_map: Dict[str, int], abbr_map: Dict[str, str]) -> Dict[str, Any]:
    scene = rec.get("scene", "")
    desc = rec.get("description", "")
    epoch = rec.get("epoch")
    scene_idx = rec.get("scene_index")
    tree_id = f"epoch{epoch}_scene{scene_idx}" if epoch is not None and scene_idx is not None else "tree"

    states: List[Dict[str, Any]] = []
    Q: List[float] = []
    V_teacher: List[float] = []
    strategy_probs: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    traj_id_parts = ["root"]

    for t, node in enumerate(path):
        mr = mean_reward(node)
        if t == 0:
            node_q = max_action_q(node)
            if node_q is None:
                node_q = mr
        else:
            node_q = None
            parent = path[t - 1]
            s = node.get("strategy")
            if s:
                p_stats = parent.get("action_stats") or {}
                if isinstance(p_stats, dict):
                    entry = p_stats.get(s)
                    if isinstance(entry, dict) and entry.get("q") is not None:
                        try:
                            node_q = float(entry["q"])
                        except Exception:
                            node_q = None
            if node_q is None:
                node_q = mr

        states.append(
            {
                "t": t,
                "node_id": node.get("node_id", f"n{t}"),
                "history": build_history(node.get("history") or []),
            }
        )
        Q.append(node_q)
        V_teacher.append(mr)
        strategy_probs.append(node.get("strategy_probs") or {})
        if t > 0:
            s = node.get("strategy")
            if s:
                if s in strat_id_map:
                    actions.append({"t": t - 1, "chosen_strategy_id": strat_id_map[s]})
                traj_id_parts.append(abbr_map.get(s, s))

    traj_id = "->".join(traj_id_parts)
    return {
        "epoch": epoch,
        "scene_index": scene_idx,
        "scene": scene,
        "description": desc,
        "tree_id": tree_id,
        "traj_id": traj_id,
        "states": states,
        "Q": Q,
        "V_teacher": V_teacher,
        "strategy": strategy_probs,
        "actions": actions,
    }


# ---------------- main ---------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract trajectories from MCTS tree JSON/JSONL.")
    ap.add_argument("path", nargs="?", default=None, help="Tree file path.")
    ap.add_argument("--out", type=str, default=None, help="Output JSONL path.")
    args = ap.parse_args()

    tree_path = Path(args.path) if args.path else default_tree_path()
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    min_len, max_len = load_minmax()
    strat_id_map, abbr_map = load_strategy_map()

    records = load_records(tree_path)
    samples: List[Dict[str, Any]] = []
    for rec in records:
        tree = rec.get("tree") or rec.get("root")
        if not tree:
            continue
        paths = enumerate_paths(tree)
        for p in paths:
            if len(p) < min_len:
                continue
            if max_len is not None and len(p) > max_len:
                continue
            samples.append(path_to_sample(p, rec, strat_id_map, abbr_map))

    out_path = Path(args.out) if args.out else tree_path.with_name(f"{tree_path.stem}_paths.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} trajectories to {out_path} (JSONL)")


if __name__ == "__main__":
    main()
