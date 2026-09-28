#!/usr/bin/env python3
"""
Render a tree from Ex_Tree*.json/jsonl into a simple node-link plot.

Usage:
  python analyze/draw_tree.py [TREE_PATH] [--index N] [--out PATH]
If TREE_PATH is omitted, uses analyze/tree_paths.json -> tree_path, else falls back to data/processed/extes/trees/Ex_Tree.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")


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
    return Path("data/processed/extes/trees/Ex_Tree.jsonl")


def default_figure_path(tree_path: Path, index: int) -> Path:
    if tree_path.parent.name == "runs" and tree_path.parent.parent.name == "trees":
        return tree_path.parent.parent.parent / "figures" / f"{tree_path.stem}_tree{index}.png"
    if tree_path.parent.name == "trees":
        return tree_path.parent.parent / "figures" / f"{tree_path.stem}_tree{index}.png"
    return tree_path.parent / f"{tree_path.stem}_tree{index}.png"


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


def collect(tree: Dict[str, Any], nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], depth: int = 0) -> None:
    nid = str(tree.get("node_id", f"n{len(nodes)}"))
    label = tree.get("strategy") or "ROOT"
    action_stats = tree.get("action_stats") or {}
    qs_actions = [v.get("q") for v in action_stats.values() if isinstance(v, dict) and v.get("q") is not None]
    if qs_actions:
        q_state = max(qs_actions)
    else:
        q_state = tree.get("mean_reward")
        if q_state is None:
            tot = tree.get("total_reward")
            vis = tree.get("visits")
            if tot is not None and vis:
                try:
                    q_state = float(tot) / float(vis)
                except Exception:
                    q_state = None
    n_state = tree.get("visits")

    actions: List[Dict[str, Any]] = []
    if isinstance(action_stats, dict):
        for k, v in action_stats.items():
            actions.append(
                {
                    "strategy": k,
                    "q": v.get("q"),
                    "visits": v.get("visits"),
                }
            )

    nodes.append(
        {
            "id": nid,
            "label": label,
            "depth": depth,
            "q_state": q_state,
            "n_state": n_state,
            "actions": actions,
            "strategy_probs": tree.get("strategy_probs"),
        }
    )
    children_raw = tree.get("children") or []
    children = list(children_raw.values()) if isinstance(children_raw, dict) else list(children_raw)
    for ch in children:
        cid = str(ch.get("node_id", f"{nid}_c{len(edges)}"))
        a_name = ch.get("strategy")
        a_stat = action_stats.get(a_name, {}) if isinstance(action_stats, dict) else {}
        edge_q = a_stat.get("q")
        edge_n = a_stat.get("visits")
        edge_prob = None
        probs = tree.get("strategy_probs")
        if isinstance(probs, dict):
            edge_prob = probs.get(a_name)
        elif isinstance(probs, list):
            idx = children.index(ch)
            if idx < len(probs):
                edge_prob = probs[idx]
        edges.append({"parent": nid, "child": cid, "strategy": a_name, "q": edge_q, "n": edge_n, "prob": edge_prob})
        ch = dict(ch)
        ch["node_id"] = cid
        collect(ch, nodes, edges, depth + 1)


def layout(nodes: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    children_map = {n["id"]: [] for n in nodes}
    for n in nodes:
        children_map.setdefault(n["id"], [])
    depth_map = {n["id"]: n["depth"] for n in nodes}
    positions: Dict[str, Tuple[float, float]] = {}
    x_cursor = 0.0

    id_set = {n["id"] for n in nodes}
    child_ids = set()
    for _, cid in children_map.items():
        child_ids.update(cid)
    return positions


def compute_positions(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    children_map = {nid: [] for nid in [n["id"] for n in nodes]}
    for e in edges:
        p, c = e["parent"], e["child"]
        children_map.setdefault(p, []).append(c)
    depth_map = {n["id"]: n["depth"] for n in nodes}
    positions: Dict[str, Tuple[float, float]] = {}
    x_cursor = 0.0

    def dfs(nid: str) -> float:
        nonlocal x_cursor
        kids = children_map.get(nid, [])
        if not kids:
            x = x_cursor
            x_cursor += 2.5
        else:
            xs = [dfs(k) for k in kids]
            x = sum(xs) / len(xs)
        y = -depth_map.get(nid, 0)
        positions[nid] = (x, y)
        return x

    root_id = min(nodes, key=lambda n: n["depth"])["id"] if nodes else "root"
    dfs(root_id)
    return positions


def draw(tree: Dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    abbr_map = load_abbr_map()

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    collect(tree, nodes, edges, depth=0)
    pos = compute_positions(nodes, edges)

    def abbr(label: str) -> str:
        if not label or label == "ROOT":
            return "ROOT"
        if label in abbr_map:
            return abbr_map[label]
        parts = label.replace("_", " ").split()
        initials = "".join(p[0].upper() for p in parts if p)
        return initials if initials else label[:6]

    palette = ["#4a90e2", "#50e3c2", "#f5a623", "#e94e77", "#7b7bff", "#8bd350", "#f45c43", "#9c27b0"]
    color_map: Dict[str, str] = {"ROOT": "#555555"}

    def color_for(label: str) -> str:
        a = abbr(label)
        if a in color_map:
            return color_map[a]
        color = palette[len(color_map) % len(palette)]
        color_map[a] = color
        return color

    fig, ax = plt.subplots(figsize=(12, 7))
    for e in edges:
        p, c = e["parent"], e["child"]
        if p in pos and c in pos:
            x0, y0 = pos[p]
            x1, y1 = pos[c]
            ax.plot([x0, x1], [y0, y1], color="#999999", linewidth=1.0, zorder=0)
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            parts = []
            if e.get("strategy"):
                parts.append(f"{abbr(e['strategy'])}")
            if e.get("q") is not None:
                parts.append(f"Q(s,a)={e['q']:.3f}")
            if e.get("n") is not None:
                parts.append(f"N(s,a)={e['n']:.1f}")
            if e.get("prob") is not None:
                parts.append(f"p={e['prob']:.3f}")
            if parts:
                ax.text(mx, my, " ".join(parts), ha="center", va="center", color="#555555", fontsize=6, zorder=1)
    for n in nodes:
        x, y = pos.get(n["id"], (0, 0))
        col = color_for(n["label"])
        ax.scatter(x, y, s=700, color=col, edgecolor="#333333", linewidth=1.0, zorder=2)
        ax.text(x, y, abbr(n["label"]), ha="center", va="center", color="white", weight="bold", fontsize=9, zorder=3)
        q_val = n.get("q_state")
        n_val = n.get("n_state")
        offset = 0.35
        if q_val is not None:
            ax.text(x, y + offset, f"Q(s)={q_val:.3f}", ha="center", va="bottom", color=col, fontsize=7, zorder=3)
        if n_val is not None:
            ax.text(x, y - offset, f"N(s)={n_val:.1f}", ha="center", va="top", color=col, fontsize=7, zorder=3)
            
    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

def load_abbr_map() -> Dict[str, str]:
    """Load strategy -> abbreviation from data/strategies.json if present."""
    path = Path("data/strategies.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, str] = {}
        for item in data:
            if isinstance(item, dict):
                name = item.get("strategy")
                abbr = item.get("abbreviation")
                if name and abbr:
                    out[str(name)] = str(abbr)
        return out
    except Exception:
        return {}
    
def main() -> None:
    ap = argparse.ArgumentParser(description="Draw a tree as a node-link plot.")
    ap.add_argument("path", nargs="?", default=None, help="Tree file (.json/.jsonl).")
    ap.add_argument("--index", type=int, default=0, help="Tree index to draw.")
    ap.add_argument("--out", type=str, default=None, help="Output image path.")
    args = ap.parse_args()

    tree_path = Path(args.path) if args.path else default_tree_path()
    records = load_records(tree_path)
    trees: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        t = rec.get("tree") or rec.get("root")
        if t:
            trees.append(t)
    if not trees:
        raise ValueError(f"No trees found in {tree_path}")
    if args.index < 0 or args.index >= len(trees):
        raise IndexError(f"index {args.index} out of range (total {len(trees)})")
    tree = trees[args.index]
    out_path = Path(args.out) if args.out else default_figure_path(tree_path, args.index)
    draw(tree, out_path)


if __name__ == "__main__":
    main()
