from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from outotesti.behavior import (
    centered_cosine_distance,
    combine_qk_distances,
    label_permutation_correlation_test,
    upper_triangle_correlation,
)
from outotesti.subspace import head_subspace_distance_matrix
from real7_head_subspace_hierarchy import load_trained_and_init, qk_by_layer


MODELS = (
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-Instruct-1M",
)

TEXTS = [
    "Once upon a time a little fox found a blue key beside the river.",
    "Mia planted a seed in a red pot and checked it every morning before school.",
    "The small robot wanted to bake a cake, but it had never seen flour before.",
    "Ben heard a quiet sound under the table and discovered a sleepy kitten.",
    "A green bird flew into the garden and dropped a shiny button near the gate.",
    "Lena carried her umbrella all day even though the sky was bright and clear.",
    "The old clock stopped at noon, so Sam opened the back and found a loose gear.",
    "Two rabbits built a tiny bridge from sticks so they could cross the puddle.",
    "Nora lost her yellow scarf on the hill and followed the wind to find it.",
    "A friendly dragon was afraid of smoke and asked the baker for help.",
    "The toy boat drifted away from the dock until a duck pushed it back.",
    "Omar found three stones that made different sounds when he tapped them.",
    "The moon was bright, and the children made long shadows on the snow.",
    "A tiny mouse wanted the cheese on the shelf, so it built stairs from books.",
    "Emma painted a door on cardboard and her brother made a paper handle for it.",
    "The puppy carried one sock from every room and hid them behind the chair.",
    "A farmer left a basket outside, and by morning it was full of fallen apples.",
    "The little train could not climb the hill until the cars shared their load.",
    "Sofia made a paper bird, opened the window, and watched the wind lift it.",
    "The frog sat on the warm stone until the first drops of rain began to fall.",
    "A boy found a cracked cup and used it as a tiny home for a garden snail.",
    "The snowman lost his hat in the wind, and a crow returned it from the fence.",
    "A lamp in the attic still worked, but only when the old wooden door was open.",
    "Three friends followed a trail of silver wrappers and found the missing picnic.",
]


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def attention_behavior_distances(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int = 64,
) -> dict[int, np.ndarray]:
    import torch

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(
            **batch,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = out.attentions
    if attentions is None:
        raise RuntimeError("model did not return attentions")

    mask = batch["attention_mask"].bool().cpu().numpy()
    B, T = mask.shape
    causal = np.tril(np.ones((T, T), dtype=bool))
    valid = (
        mask[:, :, None]
        & mask[:, None, :]
        & causal[None, :, :]
    )

    result = {}
    for layer, tensor in enumerate(attentions):
        A = tensor.detach().float().cpu().numpy()
        if A.ndim != 4:
            raise RuntimeError(f"unexpected attention shape {A.shape}")
        H = A.shape[1]

        X = np.empty((H, int(np.sum(valid))), dtype=float)
        for h in range(H):
            X[h] = A[:, h, :, :][valid]

        result[layer] = centered_cosine_distance(X)

    return result


def weight_distances(model) -> dict[int, dict[str, np.ndarray]]:
    layers = qk_by_layer(model)
    num_heads = int(
        getattr(model.config, "num_heads", getattr(model.config, "num_attention_heads"))
    )
    result = {}

    for layer in sorted(layers):
        Q = layers[layer]["Q"].detach().float().cpu().numpy().astype(float)
        K = layers[layer]["K"].detach().float().cpu().numpy().astype(float)
        Dq = head_subspace_distance_matrix(Q, num_heads=num_heads)
        Dk = head_subspace_distance_matrix(K, num_heads=num_heads)
        result[layer] = {
            "Q": Dq,
            "K": Dk,
            "QK": combine_qk_distances(Dq, Dk),
        }
    return result


def one_model(
    model,
    tokenizer,
    *,
    permutation_controls: int,
    seed: int,
) -> dict:
    weights = weight_distances(model)
    full_behavior = attention_behavior_distances(model, tokenizer, TEXTS)
    first_behavior = attention_behavior_distances(
        model, tokenizer, TEXTS[: len(TEXTS) // 2]
    )
    second_behavior = attention_behavior_distances(
        model, tokenizer, TEXTS[len(TEXTS) // 2 :]
    )

    detail = {}
    rows = []

    for layer in sorted(weights):
        Db = full_behavior[layer]
        reliability = upper_triangle_correlation(
            first_behavior[layer],
            second_behavior[layer],
        )

        primary = label_permutation_correlation_test(
            weights[layer]["QK"],
            Db,
            controls=permutation_controls,
            seed=seed + layer * 100 + 1,
        )
        q_only = label_permutation_correlation_test(
            weights[layer]["Q"],
            Db,
            controls=permutation_controls,
            seed=seed + layer * 100 + 2,
        )
        k_only = label_permutation_correlation_test(
            weights[layer]["K"],
            Db,
            controls=permutation_controls,
            seed=seed + layer * 100 + 3,
        )

        row = {
            "layer": int(layer),
            "behavior_split_half_reliability": float(reliability),
            "QK_behavior_correlation": float(primary["observed"]),
            "QK_behavior_p": float(primary["empirical_p_upper"]),
            "QK_behavior_z": float(primary["z"]),
            "Q_behavior_correlation": float(q_only["observed"]),
            "K_behavior_correlation": float(k_only["observed"]),
        }
        rows.append(row)
        detail[layer] = {
            **row,
            "QK_test": primary,
            "Q_test": q_only,
            "K_test": k_only,
        }

        print(
            f"layer {layer} "
            f"reliability={reliability:+.3f} "
            f"QK->behavior r={primary['observed']:+.3f} "
            f"p={primary['empirical_p_upper']:.4f} "
            f"Q={q_only['observed']:+.3f} "
            f"K={k_only['observed']:+.3f}"
        )

    rel = np.asarray(
        [r["behavior_split_half_reliability"] for r in rows], dtype=float
    )
    corr = np.asarray(
        [r["QK_behavior_correlation"] for r in rows], dtype=float
    )
    p = np.asarray([r["QK_behavior_p"] for r in rows], dtype=float)

    aggregate = {
        "layer_count": len(rows),
        "median_behavior_reliability": float(np.median(rel)),
        "fraction_behavior_reliability_gt_0p3": float(np.mean(rel > 0.3)),
        "median_QK_behavior_correlation": float(np.median(corr)),
        "fraction_QK_behavior_correlation_positive": float(np.mean(corr > 0)),
        "QK_behavior_p_lt_0p05_layers": int(np.sum(p < 0.05)),
        "median_Q_behavior_correlation": float(
            np.median([r["Q_behavior_correlation"] for r in rows])
        ),
        "median_K_behavior_correlation": float(
            np.median([r["K_behavior_correlation"] for r in rows])
        ),
    }
    return {"aggregate": aggregate, "layers": detail, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutation-controls", type=int, default=512)
    ap.add_argument("--init-seed", type=int, default=20260903)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "real10" / "audit.json",
    )
    args = ap.parse_args()

    model_results = {}
    trained_rows = []
    init_rows = []

    for mi, model_name in enumerate(MODELS):
        print()
        print(f"MODEL {model_name}")
        trained, fresh = load_trained_and_init(
            model_name, args.init_seed + mi
        )
        tokenizer = load_tokenizer(model_name)

        print("TRAINED")
        tr = one_model(
            trained,
            tokenizer,
            permutation_controls=args.permutation_controls,
            seed=1010000 + mi * 10000,
        )

        print("RANDOM INIT")
        ri = one_model(
            fresh,
            tokenizer,
            permutation_controls=args.permutation_controls,
            seed=1210000 + mi * 10000,
        )

        model_results[model_name] = {
            "trained": tr,
            "random_initialization": ri,
        }
        trained_rows.extend(tr["rows"])
        init_rows.extend(ri["rows"])

    trained_corr = np.asarray(
        [r["QK_behavior_correlation"] for r in trained_rows], dtype=float
    )
    init_corr = np.asarray(
        [r["QK_behavior_correlation"] for r in init_rows], dtype=float
    )
    trained_rel = np.asarray(
        [r["behavior_split_half_reliability"] for r in trained_rows], dtype=float
    )
    trained_p = np.asarray(
        [r["QK_behavior_p"] for r in trained_rows], dtype=float
    )

    aggregate = {
        "trained": {
            "model_layer_count": len(trained_rows),
            "median_behavior_reliability": float(np.median(trained_rel)),
            "fraction_behavior_reliability_gt_0p3": float(
                np.mean(trained_rel > 0.3)
            ),
            "median_QK_behavior_correlation": float(np.median(trained_corr)),
            "fraction_QK_behavior_correlation_positive": float(
                np.mean(trained_corr > 0)
            ),
            "QK_behavior_p_lt_0p05_layers": int(np.sum(trained_p < 0.05)),
        },
        "random_initialization": {
            "median_QK_behavior_correlation": float(np.median(init_corr)),
            "fraction_QK_behavior_correlation_positive": float(
                np.mean(init_corr > 0)
            ),
        },
        "trained_minus_init_median_correlation": float(
            np.median(trained_corr) - np.median(init_corr)
        ),
        "fraction_paired_trained_correlation_better": float(
            np.mean(trained_corr > init_corr)
        ),
    }

    reliability_pass = (
        aggregate["trained"]["median_behavior_reliability"] >= 0.5
        and aggregate["trained"][
            "fraction_behavior_reliability_gt_0p3"
        ] >= 0.75
    )
    predictive_pass = (
        aggregate["trained"]["median_QK_behavior_correlation"] >= 0.2
        and aggregate["trained"][
            "fraction_QK_behavior_correlation_positive"
        ] >= 0.75
        and aggregate["trained"]["QK_behavior_p_lt_0p05_layers"] >= 8
    )
    training_specific = (
        aggregate["trained_minus_init_median_correlation"] >= 0.15
        and aggregate[
            "fraction_paired_trained_correlation_better"
        ] >= 0.75
    )

    if reliability_pass and predictive_pass and training_specific:
        classification = "HEAD_SUBSPACE_GEOMETRY_PREDICTS_ATTENTION_BEHAVIOR"
    elif reliability_pass:
        classification = "HEAD_SUBSPACE_GEOMETRY_NOT_BEHAVIOR_PREDICTIVE"
    else:
        classification = "ATTENTION_BEHAVIOR_METRIC_NOT_RELIABLE_ENOUGH"

    summary = {
        "experiment": "REAL10",
        "models": list(MODELS),
        "texts": TEXTS,
        "weight_geometry": (
            "mean of Q and K gauge-invariant head row-subspace chordal distances"
        ),
        "behavior_geometry": (
            "centered-cosine distance between flattened per-head attention maps "
            "on 24 fixed story-like texts"
        ),
        "protocol": {
            "permutation_controls_per_layer": int(args.permutation_controls),
            "primary_measure": (
                "upper-triangle correlation between weight and behavioral "
                "head-distance matrices"
            ),
            "behavior_reliability": (
                "correlation between attention-distance matrices from two "
                "12-text halves"
            ),
            "predeclared_reliability_pass": (
                "median split-half >=.5 and >.3 on >=75% of 16 trained model-layers"
            ),
            "predeclared_predictive_pass": (
                "median correlation>=.2, positive>=75%, permutation p<.05 "
                "in >=8/16 trained model-layers"
            ),
            "predeclared_training_specific": (
                "trained-init median correlation>=.15 and trained larger on "
                ">=75% paired model-layers"
            ),
        },
        "aggregate": aggregate,
        "classification": classification,
        "model_results": model_results,
        "stopping_line": (
            "A positive result earns extraction as a reusable head-geometry "
            "interpretability audit. A negative result ends the chase at "
            "replicated parameter-space organization."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("REAL10 aggregate")
    print(f"trained behavior reliability:         {aggregate['trained']['median_behavior_reliability']:+.3f}")
    print(f"trained median weight-behavior corr:  {aggregate['trained']['median_QK_behavior_correlation']:+.3f}")
    print(f"trained corr positive:                {aggregate['trained']['fraction_QK_behavior_correlation_positive']:.3f}")
    print(f"trained p<.05 layers:                 {aggregate['trained']['QK_behavior_p_lt_0p05_layers']} / 16")
    print(f"random-init median corr:              {aggregate['random_initialization']['median_QK_behavior_correlation']:+.3f}")
    print(f"trained-init median corr:             {aggregate['trained_minus_init_median_correlation']:+.3f}")
    print(f"paired trained corr better:           {aggregate['fraction_paired_trained_correlation_better']:.3f}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
