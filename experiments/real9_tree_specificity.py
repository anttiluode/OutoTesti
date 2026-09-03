from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from outotesti.metric_compression import target_errors
from outotesti.subspace import head_subspace_distance_matrix
from real7_head_subspace_hierarchy import load_trained_and_init, qk_by_layer


MODELS = (
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-Instruct-1M",
)


def one_model(model) -> dict:
    layers = qk_by_layer(model)
    num_heads = int(
        getattr(model.config, "num_heads", getattr(model.config, "num_attention_heads"))
    )

    rows = []
    detail = {}

    for layer in sorted(layers):
        Q = layers[layer]["Q"].detach().float().cpu().numpy().astype(float)
        K = layers[layer]["K"].detach().float().cpu().numpy().astype(float)

        Dq = head_subspace_distance_matrix(Q, num_heads=num_heads)
        Dk = head_subspace_distance_matrix(K, num_heads=num_heads)

        layer_detail = {}
        for direction, source, target in (
            ("Q_to_K", Dq, Dk),
            ("K_to_Q", Dk, Dq),
        ):
            err = target_errors(source, target)
            row = {
                "layer": int(layer),
                "direction": direction,
                "raw_error": float(err["raw"]),
                "star_error": float(err["star"]),
                "tree_error": float(err["tree"]),
                "mds2_error": float(err["mds2"]),
                "tree_vs_mds2_gain": float(err["mds2"] - err["tree"]),
                "tree_vs_star_gain": float(err["star"] - err["tree"]),
                "tree_compression_penalty": float(err["tree"] - err["raw"]),
                "mds2_compression_penalty": float(err["mds2"] - err["raw"]),
            }
            rows.append(row)
            layer_detail[direction] = row

            print(
                f"layer {layer} {direction:6s} "
                f"raw={row['raw_error']:.4f} "
                f"star={row['star_error']:.4f} "
                f"tree={row['tree_error']:.4f} "
                f"mds2={row['mds2_error']:.4f} "
                f"tree-MDSgain={row['tree_vs_mds2_gain']:+.4f}"
            )
        detail[layer] = layer_detail

    def arr(field):
        return np.asarray([r[field] for r in rows], dtype=float)

    aggregate = {
        "transfer_count": len(rows),
        "median_raw_error": float(np.median(arr("raw_error"))),
        "median_star_error": float(np.median(arr("star_error"))),
        "median_tree_error": float(np.median(arr("tree_error"))),
        "median_mds2_error": float(np.median(arr("mds2_error"))),
        "median_tree_vs_mds2_gain": float(
            np.median(arr("tree_vs_mds2_gain"))
        ),
        "fraction_tree_beats_mds2": float(
            np.mean(arr("tree_vs_mds2_gain") > 0)
        ),
        "median_tree_vs_star_gain": float(
            np.median(arr("tree_vs_star_gain"))
        ),
        "fraction_tree_beats_star": float(
            np.mean(arr("tree_vs_star_gain") > 0)
        ),
        "median_tree_compression_penalty": float(
            np.median(arr("tree_compression_penalty"))
        ),
        "median_mds2_compression_penalty": float(
            np.median(arr("mds2_compression_penalty"))
        ),
    }
    return {"aggregate": aggregate, "rows": rows, "layers": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-seed", type=int, default=20260903)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "real9" / "audit.json",
    )
    args = ap.parse_args()

    model_results = {}
    pooled_trained = []
    pooled_init = []

    for mi, model_name in enumerate(MODELS):
        print()
        print(f"MODEL {model_name}")
        trained, fresh = load_trained_and_init(
            model_name, args.init_seed + mi
        )

        print("TRAINED")
        tr = one_model(trained)
        print("RANDOM INIT")
        ri = one_model(fresh)

        model_results[model_name] = {
            "trained": tr,
            "random_initialization": ri,
        }
        pooled_trained.extend(tr["rows"])
        pooled_init.extend(ri["rows"])

    def pooled_summary(rows):
        def arr(field):
            return np.asarray([r[field] for r in rows], dtype=float)
        return {
            "count": len(rows),
            "median_raw_error": float(np.median(arr("raw_error"))),
            "median_star_error": float(np.median(arr("star_error"))),
            "median_tree_error": float(np.median(arr("tree_error"))),
            "median_mds2_error": float(np.median(arr("mds2_error"))),
            "median_tree_vs_mds2_gain": float(
                np.median(arr("tree_vs_mds2_gain"))
            ),
            "fraction_tree_beats_mds2": float(
                np.mean(arr("tree_vs_mds2_gain") > 0)
            ),
            "median_tree_vs_star_gain": float(
                np.median(arr("tree_vs_star_gain"))
            ),
            "fraction_tree_beats_star": float(
                np.mean(arr("tree_vs_star_gain") > 0)
            ),
            "median_tree_compression_penalty": float(
                np.median(arr("tree_compression_penalty"))
            ),
            "median_mds2_compression_penalty": float(
                np.median(arr("mds2_compression_penalty"))
            ),
        }

    trained_summary = pooled_summary(pooled_trained)
    init_summary = pooled_summary(pooled_init)

    per_model_tree_gains = {
        name: result["trained"]["aggregate"]["median_tree_vs_mds2_gain"]
        for name, result in model_results.items()
    }

    paired_better = np.asarray(
        [
            t["tree_vs_mds2_gain"] > i["tree_vs_mds2_gain"]
            for t, i in zip(pooled_trained, pooled_init)
        ],
        dtype=bool,
    )

    aggregate = {
        "trained": trained_summary,
        "random_initialization": init_summary,
        "per_model_trained_tree_vs_mds2_gain": per_model_tree_gains,
        "trained_minus_init_tree_vs_mds2_gain": float(
            trained_summary["median_tree_vs_mds2_gain"]
            - init_summary["median_tree_vs_mds2_gain"]
        ),
        "fraction_paired_tree_advantage_trained_better": float(
            np.mean(paired_better)
        ),
    }

    tree_specific = (
        trained_summary["median_tree_vs_mds2_gain"] >= 0.01
        and trained_summary["fraction_tree_beats_mds2"] >= 0.75
        and all(v >= 0.005 for v in per_model_tree_gains.values())
        and trained_summary["median_tree_vs_star_gain"] >= 0.01
        and trained_summary["fraction_tree_beats_star"] >= 0.75
    )

    training_specific = (
        aggregate["trained_minus_init_tree_vs_mds2_gain"] >= 0.005
        and aggregate[
            "fraction_paired_tree_advantage_trained_better"
        ] >= 0.75
    )

    if tree_specific and training_specific:
        classification = "TREE_IS_DISTINCT_COMPRESSION_OF_SHARED_HEAD_GEOMETRY"
    else:
        classification = "SHARED_HEAD_GEOMETRY_NOT_TREE_SPECIFIC"

    summary = {
        "experiment": "REAL9",
        "models": list(MODELS),
        "object": "gauge-invariant Q/K attention-head row-subspace distance",
        "protocol": {
            "representations": {
                "raw": "120 source pairwise distances; uncompressed ceiling",
                "star": "16 nonnegative leaf radii",
                "tree": "neighbor-joining; 29 branch lengths plus discrete topology",
                "mds2": "classical 2-D MDS; 32 nominal coordinates",
            },
            "target_fit": "one nonnegative global scale for every representation",
            "no_target_structure_fit": True,
            "predeclared_tree_specific": (
                "pooled trained median MDS2-error minus tree-error >=.01; "
                "tree beats MDS2 >=75%; each model median gain >=.005; "
                "tree beats star by median >=.01 and >=75%"
            ),
            "predeclared_training_specific": (
                "trained-init tree-vs-MDS2 median gain >=.005 and trained "
                "advantage larger on >=75% paired transfers"
            ),
        },
        "aggregate": aggregate,
        "classification": classification,
        "model_results": model_results,
        "stopping_line": (
            "If tree-specificity fails, retain the replicated shared Q/K "
            "head-subspace geometry result but drop the hierarchy claim. "
            "Do not rescue with a more flexible tree."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("REAL9 aggregate — TRAINED")
    print(f"raw median error:                    {trained_summary['median_raw_error']:.4f}")
    print(f"star median error:                   {trained_summary['median_star_error']:.4f}")
    print(f"tree median error:                   {trained_summary['median_tree_error']:.4f}")
    print(f"MDS2 median error:                   {trained_summary['median_mds2_error']:.4f}")
    print(f"tree-vs-MDS2 median gain:            {trained_summary['median_tree_vs_mds2_gain']:+.4f}")
    print(f"tree beats MDS2:                     {trained_summary['fraction_tree_beats_mds2']:.3f}")
    print(f"tree-vs-star median gain:            {trained_summary['median_tree_vs_star_gain']:+.4f}")
    print("RANDOM INIT")
    print(f"tree-vs-MDS2 median gain:            {init_summary['median_tree_vs_mds2_gain']:+.4f}")
    print()
    print(f"trained-init tree advantage:         {aggregate['trained_minus_init_tree_vs_mds2_gain']:+.4f}")
    print(f"paired trained advantage better:     {aggregate['fraction_paired_tree_advantage_trained_better']:.3f}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
