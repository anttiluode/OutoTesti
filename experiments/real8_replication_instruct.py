from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from real7_head_subspace_hierarchy import load_trained_and_init, one_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="roneneldan/TinyStories-Instruct-1M",
    )
    ap.add_argument("--controls", type=int, default=128)
    ap.add_argument("--init-seed", type=int, default=20260903)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "real8" / "audit.json",
    )
    args = ap.parse_args()

    trained, fresh = load_trained_and_init(args.model, args.init_seed)

    print("REAL8 TRAINED REPLICATION")
    trained_result = one_model(
        trained, controls=args.controls, seed=1710000
    )

    print()
    print("REAL8 RANDOM-INIT CONTROL")
    init_result = one_model(
        fresh, controls=args.controls, seed=1910000
    )

    ta = trained_result["aggregate"]
    ia = init_result["aggregate"]

    trained_transfers = []
    init_transfers = []
    for layer in range(8):
        for direction in ("Q_to_K", "K_to_Q"):
            trained_transfers.append(
                trained_result["layers"][layer][direction]
            )
            init_transfers.append(
                init_result["layers"][layer][direction]
            )

    paired_label_better = np.asarray(
        [
            t["label_shuffle_gain"] > i["label_shuffle_gain"]
            for t, i in zip(trained_transfers, init_transfers)
        ],
        dtype=bool,
    )
    paired_random_better = np.asarray(
        [
            t["random_topology_gain"] > i["random_topology_gain"]
            for t, i in zip(trained_transfers, init_transfers)
        ],
        dtype=bool,
    )

    aggregate = {
        "trained": ta,
        "random_initialization": ia,
        "trained_minus_init_median_label_gain": float(
            ta["median_label_shuffle_gain"]
            - ia["median_label_shuffle_gain"]
        ),
        "trained_minus_init_median_random_gain": float(
            ta["median_random_topology_gain"]
            - ia["median_random_topology_gain"]
        ),
        "fraction_paired_label_gain_trained_better": float(
            np.mean(paired_label_better)
        ),
        "fraction_paired_random_gain_trained_better": float(
            np.mean(paired_random_better)
        ),
    }

    transfer_pass = (
        ta["median_label_shuffle_gain"] >= 0.01
        and ta["fraction_label_gain_positive"] >= 0.75
        and ta["median_random_topology_gain"] >= 0.01
        and ta["fraction_random_gain_positive"] >= 0.75
    )
    training_specific = (
        aggregate["trained_minus_init_median_label_gain"] >= 0.01
        and aggregate["fraction_paired_label_gain_trained_better"] >= 0.75
    )
    correlation_support = (
        ta["median_QK_distance_correlation"] >= 0.2
        and ta["correlation_p_lt_0p05_layers"] >= 4
    )

    if transfer_pass and training_specific and correlation_support:
        classification = "REAL7_HEAD_SUBSPACE_HIERARCHY_REPLICATES"
    else:
        classification = "REAL7_HEAD_SUBSPACE_HIERARCHY_DOES_NOT_REPLICATE"

    summary = {
        "experiment": "REAL8",
        "replicates": "REAL7",
        "model": args.model,
        "locked_protocol": True,
        "protocol": {
            "controls_per_transfer": int(args.controls),
            "metric": "normalized Grassmann chordal distance",
            "tree": "neighbor-joining on source Q/K head-subspace metric",
            "target_fit": "one global distance scale only",
            "attackers": [
                "same tree with head labels shuffled",
                "random topology with same branch-length multiset",
                "fresh random-initialized exact architecture",
                "direct Q/K distance correlation with label permutation",
            ],
            "thresholds": {
                "median_label_gain": 0.01,
                "fraction_label_positive": 0.75,
                "median_random_gain": 0.01,
                "fraction_random_positive": 0.75,
                "trained_minus_init_label_gain": 0.01,
                "paired_trained_label_better": 0.75,
                "median_QK_distance_correlation": 0.2,
                "correlation_p_lt_0p05_layers": 4,
            },
        },
        "aggregate": aggregate,
        "classification": classification,
        "trained": trained_result,
        "random_initialization": init_result,
        "stopping_line": (
            "If replication passes unchanged, next attack tree-specificity rather "
            "than model size. If it fails, keep REAL7 model-specific."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("REAL8 aggregate")
    print("TRAINED")
    print(f"  median label gain:                 {ta['median_label_shuffle_gain']:+.4f}")
    print(f"  label gain positive:               {ta['fraction_label_gain_positive']:.3f}")
    print(f"  median random-topology gain:       {ta['median_random_topology_gain']:+.4f}")
    print(f"  median QK distance correlation:    {ta['median_QK_distance_correlation']:+.3f}")
    print(f"  corr p<.05 layers:                 {ta['correlation_p_lt_0p05_layers']} / 8")
    print("RANDOM INIT")
    print(f"  median label gain:                 {ia['median_label_shuffle_gain']:+.4f}")
    print(f"  median QK distance correlation:    {ia['median_QK_distance_correlation']:+.3f}")
    print()
    print(f"trained-init median label gain:      {aggregate['trained_minus_init_median_label_gain']:+.4f}")
    print(f"paired trained label better:         {aggregate['fraction_paired_label_gain_trained_better']:.3f}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
