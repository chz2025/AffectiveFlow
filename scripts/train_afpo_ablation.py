#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def build_variant_loss(base_module, variant: str):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Ablation training requires torch.") from exc

    def variant_loss(logits, logits_ref, v_logits, actions, worst_ids, q_values, v_teacher, beta=0.1, gamma=1.0):
        logprobs = torch.log_softmax(logits, dim=-1)
        lp_good = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        lp_bad = logprobs.gather(-1, worst_ids.unsqueeze(-1)).squeeze(-1)

        if logits_ref is not None:
            ref_logprobs = torch.log_softmax(logits_ref, dim=-1)
            lp_ref_good = ref_logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            lp_ref_bad = ref_logprobs.gather(-1, worst_ids.unsqueeze(-1)).squeeze(-1)
            kl = (lp_good - lp_ref_good).mean()
            dpo_margin = (lp_good - lp_ref_good) - (lp_bad - lp_ref_bad)
            dpo_loss = (-F.logsigmoid(dpo_margin)).mean()
        else:
            kl = torch.zeros((), device=logits.device)
            dpo_margin = lp_good - lp_bad
            dpo_loss = (-F.logsigmoid(dpo_margin)).mean()

        v_pos = v_logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        v_neg = v_logits.gather(-1, worst_ids.unsqueeze(-1)).squeeze(-1)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == actions).float().mean()

        prefix = torch.cat([torch.zeros(1, device=logits.device), torch.cumsum(lp_good, dim=0)])
        idx = torch.arange(lp_good.size(0), device=logits.device)
        m_idx, n_idx = torch.meshgrid(idx, idx, indexing="ij")
        delta_rho = prefix[n_idx + 1] - prefix[m_idx]
        flow = q_values * v_pos
        logF = torch.log(torch.clamp(flow, min=1e-6))
        logF_diff = logF[n_idx] - logF[m_idx]
        tri_mask = torch.triu(torch.ones_like(delta_rho), diagonal=1)
        flow_err = (logF_diff - delta_rho) * tri_mask
        pair_count = tri_mask.sum().clamp(min=1.0)
        flow_mse = (flow_err.pow(2).sum() / pair_count)
        eval_loss = torch.relu(gamma - (v_pos - v_neg)).mean()

        if variant == "full":
            total_loss = flow_mse + beta * torch.relu(kl) + eval_loss
        elif variant == "wo_flow_balance":
            total_loss = dpo_loss + eval_loss
        elif variant == "wo_process_sup":
            total_loss = flow_mse + beta * torch.relu(kl)
        else:
            total_loss = flow_mse + beta * torch.relu(kl) + eval_loss

        return total_loss, flow_mse, kl, accuracy, eval_loss

    return variant_loss


def build_wo_mcts_resolver(train_path: Path, val_path: Path):
    def resolver(_cfg: Dict[str, Any]) -> Dict[str, Path]:
        return {"train": train_path, "val": val_path}

    return resolver


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation wrapper around scripts/train_afpo.py")
    ap.add_argument(
        "--variant",
        required=True,
        choices=["full", "wo_flow_balance", "wo_process_sup", "wo_mcts"],
        help="Paper ablation variant.",
    )
    ap.add_argument(
        "--wo-mcts-train-path",
        default=None,
        help="Reference-path training data for the w/o MCTS variant.",
    )
    ap.add_argument(
        "--wo-mcts-val-path",
        default=None,
        help="Reference-path validation data for the w/o MCTS variant.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for a short resource benchmark.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Optional output-directory override.",
    )
    args = ap.parse_args()

    import train_afpo as base

    original_load_config = base.load_cls_config

    def load_config_with_overrides(*load_args, **load_kwargs):
        cfg = original_load_config(*load_args, **load_kwargs)
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
            cfg["save_steps"] = 0
            cfg["save_epoch_checkpoint"] = False
            cfg["save_best"] = False
            cfg["show_batch_progress"] = False
        if args.output_dir:
            cfg["output_dir"] = args.output_dir
        return cfg

    base.load_cls_config = load_config_with_overrides
    base.flow_balance_loss = build_variant_loss(base, args.variant)
    if args.variant == "wo_mcts":
        if not args.wo_mcts_train_path:
            ap.error("--wo-mcts-train-path is required for the wo_mcts variant")
        val_path = Path(args.wo_mcts_val_path) if args.wo_mcts_val_path else Path("__auto_split_validation__.jsonl")
        base.resolve_split_paths = build_wo_mcts_resolver(
            Path(args.wo_mcts_train_path),
            val_path,
        )
    base.main()


if __name__ == "__main__":
    main()
