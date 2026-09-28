#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for this script.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for this script.") from exc
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def patch_tree_paths_rel(new_train_path: Path, rel_meta_path: Path) -> None:
    info = {}
    if rel_meta_path.exists():
        info = json.loads(rel_meta_path.read_text(encoding="utf-8"))
    info["tree_path_train"] = str(new_train_path)
    rel_meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def derive_paths_output(tree_path: Path) -> Path:
    if tree_path.parent.name == "runs" and tree_path.parent.parent.name == "trees":
        return tree_path.parent.parent.parent / "paths" / "runs" / f"{tree_path.stem}_paths.jsonl"
    if tree_path.parent.name == "trees":
        return tree_path.parent.parent / "paths" / f"{tree_path.stem}_paths.jsonl"
    return tree_path.with_name(f"{tree_path.stem}_paths.jsonl")


def run(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run paper Appendix-I style MCTS sensitivity sweeps.")
    ap.add_argument("--axis", required=True, choices=["depth", "rollout", "sims"])
    ap.add_argument("--values", required=True, help="Comma-separated values, e.g. 6,8,10,12")
    ap.add_argument("--cfg", default="configs/train_emoflow.yaml")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--checkpoint", default=None, help="Optional checkpoint/run dir to evaluate instead of freshly trained model.")
    args = ap.parse_args()
    if not args.skip_eval and not args.eval_data:
        ap.error("--eval-data is required unless --skip-eval is set")

    root = Path(args.project_root).resolve()
    cfg_path = (root / args.cfg).resolve()
    rel_meta = root / "analyze/tree_paths_rel.json"
    original_cfg_text = cfg_path.read_text(encoding="utf-8")

    values = [int(v.strip()) for v in args.values.split(",") if v.strip()]
    try:
        for value in values:
            cfg = load_yaml(cfg_path)
            mcts = cfg.setdefault("mcts", {})
            output = cfg.setdefault("output", {})

            if args.axis == "depth":
                mcts["max_depth"] = value
            elif args.axis == "rollout":
                mcts["rollout_steps"] = value
            elif args.axis == "sims":
                mcts["simulations_per_tree"] = value

            base_tree = Path(output.get("tree_path", "data/processed/extes/trees/Ex_Tree.jsonl"))
            base_log = Path(output.get("log_path", "logs/ex_tree.log"))
            output["tree_path"] = str(base_tree.with_name(f"{base_tree.stem}_{args.axis}{value}{base_tree.suffix}"))
            output["log_path"] = str(base_log.with_name(f"{base_log.stem}_{args.axis}{value}{base_log.suffix}"))
            save_yaml(cfg_path, cfg)

            if not args.skip_build:
                run(["python3", "scripts/build_ex_tree.py"], cwd=root, dry_run=args.dry_run)
            if not args.skip_extract:
                run(["python3", "scripts/extract_paths.py"], cwd=root, dry_run=args.dry_run)

            if not args.dry_run:
                meta = json.loads((root / "analyze/tree_paths.json").read_text(encoding="utf-8"))
                tree_path = Path(meta["tree_path_train"])
                train_paths = derive_paths_output(tree_path)
                patch_tree_paths_rel(train_paths, rel_meta)

            if not args.skip_train:
                run(["python3", "scripts/train_afpo.py"], cwd=root, dry_run=args.dry_run)

            if not args.skip_eval:
                checkpoint = args.checkpoint or "output/afpo_cls"
                out_prefix = root / "output" / "paper_eval" / f"{args.axis}_{value}"
                run(
                    [
                        "python3",
                        "scripts/eval_auto.py",
                        "--eval-data",
                        args.eval_data,
                        "--checkpoint",
                        checkpoint,
                        "--out-preds",
                        str(out_prefix.with_name(out_prefix.name + "_preds.jsonl")),
                        "--out-metrics",
                        str(out_prefix.with_name(out_prefix.name + "_metrics.json")),
                    ],
                    cwd=root,
                    dry_run=args.dry_run,
                )
    finally:
        cfg_path.write_text(original_cfg_text, encoding="utf-8")


if __name__ == "__main__":
    main()
