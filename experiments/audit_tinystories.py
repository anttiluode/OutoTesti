from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outotesti.audit import audit_matrix, jsonable_audit


def load_model_and_tokenizer(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer_name = "EleutherAI/gpt-neo-125M" if "TinyStories" in model_name else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def square_projection_parameters(model):
    wanted = ("q_proj", "k_proj", "v_proj", "out_proj")
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2:
            continue
        if parameter.shape[0] != parameter.shape[1]:
            continue
        if not any(token in name for token in wanted):
            continue
        rows.append((name, parameter))
    return rows


def reference_logits(model, tokenizer, device: str, text: str):
    import torch

    batch = tokenizer(text, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits.float().cpu()
    return batch, logits


def behavior_damage(model, batch, reference, parameter, replacement):
    import torch

    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.copy_(
                torch.as_tensor(
                    replacement,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )
            logits = model(**batch).logits.float().cpu()

        p_log = torch.log_softmax(reference, dim=-1)
        q_log = torch.log_softmax(logits, dim=-1)
        kl = torch.sum(p_log.exp() * (p_log - q_log), dim=-1).mean()
        agreement = (
            reference.argmax(dim=-1) == logits.argmax(dim=-1)
        ).float().mean()
        rel = torch.linalg.vector_norm(logits - reference) / torch.clamp_min(
            torch.linalg.vector_norm(reference),
            1e-12,
        )
        return {
            "mean_logit_KL": float(kl.item()),
            "argmax_token_agreement": float(agreement.item()),
            "relative_logit_error": float(rel.item()),
        }
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--random-trees", type=int, default=8)
    ap.add_argument("--behavior", action="store_true")
    ap.add_argument(
        "--text",
        default=(
            "Once upon a time there was a small red bird. "
            "The bird wanted to find a warm place to sleep. "
            "It flew over the garden and saw a little house."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "tinystories_1m",
    )
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    params = square_projection_parameters(model)
    if not params:
        raise RuntimeError("no square q/k/v/out projection matrices found")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    batch = ref = None
    if args.behavior:
        batch, ref = reference_logits(model, tokenizer, args.device, args.text)

    rows = []
    full = {}

    for idx, (name, parameter) in enumerate(params):
        W = parameter.detach().float().cpu().numpy()
        result = audit_matrix(
            W,
            random_tree_controls=args.random_trees,
            seed=1000 + idx,
        )
        payload = jsonable_audit(result)
        row = {
            "name": name,
            "n": int(W.shape[0]),
            "four_point_p95": payload["channel_four_point"]["p95_gap"],
            "tree_error": payload["tree"]["error"],
            "random_tree_median_error": payload["random_tree"]["median_error"],
            "svd_error": payload["svd"]["error"],
            "svd_rank": payload["svd"]["rank"],
            "sparse_error": payload["sparse"]["error"],
            "tree_nominal_budget": payload["tree"]["parameter_budget"],
        }

        if args.behavior:
            row["tree_behavior"] = behavior_damage(
                model, batch, ref, parameter, result["tree"]["W_hat"]
            )
            row["svd_behavior"] = behavior_damage(
                model, batch, ref, parameter, result["svd"]["W_hat"]
            )
            row["sparse_behavior"] = behavior_damage(
                model, batch, ref, parameter, result["sparse"]["W_hat"]
            )

        rows.append(row)
        full[name] = row
        print(
            f"[{idx+1:02d}/{len(params):02d}] {name:55s} "
            f"tree={row['tree_error']:.3f} "
            f"rand={row['random_tree_median_error']:.3f} "
            f"svd={row['svd_error']:.3f} "
            f"sparse={row['sparse_error']:.3f}"
        )

    fields = [
        "name",
        "n",
        "four_point_p95",
        "tree_error",
        "random_tree_median_error",
        "svd_error",
        "svd_rank",
        "sparse_error",
        "tree_nominal_budget",
    ]
    with (args.out_dir / "matrix_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "behavior_enabled": bool(args.behavior),
        "matrix_count": len(rows),
        "matrices": full,
        "caveat": (
            "The tree nominal parameter budget omits the discrete topology "
            "encoding cost. SVD and sparse comparisons are attackers, not a "
            "formal compressed-bit-rate benchmark."
        ),
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
