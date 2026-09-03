from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outotesti.green import random_topology_with_lengths, relabel_leaves
from outotesti.stability import scaled_tree_metric_error
from outotesti.subspace import (
    head_subspace_distance_matrix,
    upper_triangle_correlation,
)
from outotesti.tree import neighbor_joining


def load_trained_and_init(model_name: str, seed: int):
    import torch
    from transformers import AutoModelForCausalLM

    trained = AutoModelForCausalLM.from_pretrained(model_name)
    trained.eval()

    torch.manual_seed(int(seed))
    fresh = AutoModelForCausalLM.from_config(trained.config)
    fresh.eval()
    return trained, fresh


def qk_by_layer(model):
    out = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2:
            continue
        if ".q_proj.weight" in name:
            layer = int(name.split(".h.")[1].split(".")[0])
            out.setdefault(layer, {})["Q"] = parameter
        elif ".k_proj.weight" in name:
            layer = int(name.split(".h.")[1].split(".")[0])
            out.setdefault(layer, {})["K"] = parameter
    return out


def transfer_metric(
    source_D: np.ndarray,
    target_D: np.ndarray,
    *,
    controls: int,
    seed: int,
) -> dict:
    tree = neighbor_joining(source_D)
    err = scaled_tree_metric_error(target_D, tree)
    lengths = np.asarray([w for _, _, w in tree.edges], dtype=float)

    rng = np.random.default_rng(seed)
    label_errors = []
    random_errors = []
    for _ in range(controls):
        lt = relabel_leaves(tree, rng.permutation(tree.n_leaves))
        label_errors.append(scaled_tree_metric_error(target_D, lt))

        rt = random_topology_with_lengths(tree.n_leaves, lengths, rng)
        random_errors.append(scaled_tree_metric_error(target_D, rt))

    label_errors = np.asarray(label_errors, dtype=float)
    random_errors = np.asarray(random_errors, dtype=float)

    return {
        "target_error": float(err),
        "label_shuffle_median_error": float(np.median(label_errors)),
        "label_shuffle_gain": float(np.median(label_errors) - err),
        "label_shuffle_positive": bool(err < np.median(label_errors)),
        "beats_all_label_shuffles": bool(err < np.min(label_errors)),
        "random_topology_median_error": float(np.median(random_errors)),
        "random_topology_gain": float(np.median(random_errors) - err),
        "random_topology_positive": bool(err < np.median(random_errors)),
        "beats_all_random_topologies": bool(err < np.min(random_errors)),
        "label_errors": label_errors.tolist(),
        "random_errors": random_errors.tolist(),
    }


def correlation_null(
    Dq: np.ndarray,
    Dk: np.ndarray,
    *,
    controls: int,
    seed: int,
) -> dict:
    observed = upper_triangle_correlation(Dq, Dk)
    rng = np.random.default_rng(seed)
    vals = np.empty(controls, dtype=float)

    n = Dq.shape[0]
    for i in range(controls):
        p = rng.permutation(n)
        Dkp = Dk[np.ix_(p, p)]
        vals[i] = upper_triangle_correlation(Dq, Dkp)

    return {
        "observed": float(observed),
        "null_median": float(np.median(vals)),
        "null_mean": float(np.mean(vals)),
        "null_std": float(np.std(vals, ddof=1)) if controls > 1 else 0.0,
        "empirical_p_upper": float(
            (1 + np.sum(vals >= observed)) / (controls + 1)
        ),
        "z": float(
            (observed - np.mean(vals))
            / max(float(np.std(vals, ddof=1)), 1e-12)
        ) if controls > 1 else 0.0,
    }


def one_model(model, *, controls: int, seed: int):
    layers = qk_by_layer(model)
    num_heads = int(
        getattr(model.config, "num_heads", getattr(model.config, "num_attention_heads"))
    )

    detail = {}
    transfers = []
    correlations = []

    for layer in sorted(layers):
        Q = layers[layer]["Q"].detach().float().cpu().numpy().astype(float)
        K = layers[layer]["K"].detach().float().cpu().numpy().astype(float)

        Dq = head_subspace_distance_matrix(Q, num_heads=num_heads)
        Dk = head_subspace_distance_matrix(K, num_heads=num_heads)

        q_to_k = transfer_metric(
            Dq, Dk, controls=controls, seed=seed + layer * 1000 + 1
        )
        k_to_q = transfer_metric(
            Dk, Dq, controls=controls, seed=seed + layer * 1000 + 2
        )
        corr = correlation_null(
            Dq, Dk, controls=controls, seed=seed + layer * 1000 + 3
        )

        detail[layer] = {
            "Q_to_K": q_to_k,
            "K_to_Q": k_to_q,
            "QK_distance_correlation": corr,
            "Q_distance_mean": float(np.mean(Dq[np.triu_indices(num_heads, 1)])),
            "Q_distance_std": float(np.std(Dq[np.triu_indices(num_heads, 1)])),
            "K_distance_mean": float(np.mean(Dk[np.triu_indices(num_heads, 1)])),
            "K_distance_std": float(np.std(Dk[np.triu_indices(num_heads, 1)])),
        }
        transfers.extend([q_to_k, k_to_q])
        correlations.append(corr)

        print(
            f"layer {layer} "
            f"corr={corr['observed']:+.3f} p={corr['empirical_p_upper']:.4f} "
            f"Q->K label={q_to_k['label_shuffle_gain']:+.4f} "
            f"rand={q_to_k['random_topology_gain']:+.4f} "
            f"K->Q label={k_to_q['label_shuffle_gain']:+.4f} "
            f"rand={k_to_q['random_topology_gain']:+.4f}"
        )

    aggregate = {
        "num_heads": int(num_heads),
        "transfer_count": len(transfers),
        "median_label_shuffle_gain": float(
            np.median([x["label_shuffle_gain"] for x in transfers])
        ),
        "fraction_label_gain_positive": float(
            np.mean([x["label_shuffle_positive"] for x in transfers])
        ),
        "beats_all_label_shuffles": int(
            np.sum([x["beats_all_label_shuffles"] for x in transfers])
        ),
        "median_random_topology_gain": float(
            np.median([x["random_topology_gain"] for x in transfers])
        ),
        "fraction_random_gain_positive": float(
            np.mean([x["random_topology_positive"] for x in transfers])
        ),
        "beats_all_random_topologies": int(
            np.sum([x["beats_all_random_topologies"] for x in transfers])
        ),
        "median_QK_distance_correlation": float(
            np.median([x["observed"] for x in correlations])
        ),
        "correlation_p_lt_0p05_layers": int(
            np.sum([x["empirical_p_upper"] < 0.05 for x in correlations])
        ),
        "median_correlation_z": float(
            np.median([x["z"] for x in correlations])
        ),
    }
    return {"aggregate": aggregate, "layers": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--controls", type=int, default=128)
    ap.add_argument("--init-seed", type=int, default=20260903)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "real7" / "audit.json",
    )
    args = ap.parse_args()

    trained, fresh = load_trained_and_init(args.model, args.init_seed)

    print("TRAINED HEAD-SUBSPACE GEOMETRY")
    trained_result = one_model(
        trained, controls=args.controls, seed=710000
    )

    print()
    print("RANDOM-INIT HEAD-SUBSPACE GEOMETRY")
    init_result = one_model(
        fresh, controls=args.controls, seed=910000
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
        classification = "GAUGE_INVARIANT_HEAD_SUBSPACE_HIERARCHY_PRESENT"
    elif transfer_pass and training_specific:
        classification = "HEAD_SUBSPACE_TREE_TRANSFER_WITHOUT_DISTANCE_CORRELATION"
    else:
        classification = "NO_REUSABLE_GAUGE_INVARIANT_HEAD_HIERARCHY"

    summary = {
        "experiment": "REAL7",
        "model": args.model,
        "object": (
            "16 gauge-invariant 4-D row subspaces per Q/K projection, "
            "measured by normalized Grassmann chordal distance"
        ),
        "protocol": {
            "controls_per_transfer": int(args.controls),
            "Q_to_K_and_K_to_Q": True,
            "tree_fit": (
                "neighbor-joining on source head-subspace distance; "
                "target evaluation fits one global scale only"
            ),
            "attacker_1": "same source tree and lengths with head labels shuffled",
            "attacker_2": "random binary topology with same branch-length multiset",
            "attacker_3": "fresh random-initialized exact architecture",
            "supporting_measure": (
                "direct Q/K head-subspace distance correlation vs label permutation"
            ),
            "predeclared_transfer_pass": (
                "median gain>=.01 vs label and random topology, positive>=75%"
            ),
            "predeclared_training_specific": (
                "trained-init label gain>=.01 and trained better on >=75% paired transfers"
            ),
            "predeclared_correlation_support": (
                "median QK distance correlation>=.2 and permutation p<.05 in >=4/8 layers"
            ),
        },
        "aggregate": aggregate,
        "classification": classification,
        "trained": trained_result,
        "random_initialization": init_result,
        "stopping_line": (
            "If this fails, stop the TinyStories tree chase completely. "
            "If it passes, the surviving object is a hierarchy among attention-head "
            "input subspaces, not a tree over raw weight coordinates."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("REAL7 aggregate")
    print("TRAINED")
    print(f"  median label gain:                 {ta['median_label_shuffle_gain']:+.4f}")
    print(f"  label gain positive:               {ta['fraction_label_gain_positive']:.3f}")
    print(f"  median random-topology gain:       {ta['median_random_topology_gain']:+.4f}")
    print(f"  random gain positive:              {ta['fraction_random_gain_positive']:.3f}")
    print(f"  median QK distance correlation:    {ta['median_QK_distance_correlation']:+.3f}")
    print(f"  corr p<.05 layers:                 {ta['correlation_p_lt_0p05_layers']} / 8")
    print("RANDOM INIT")
    print(f"  median label gain:                 {ia['median_label_shuffle_gain']:+.4f}")
    print(f"  median random-topology gain:       {ia['median_random_topology_gain']:+.4f}")
    print(f"  median QK distance correlation:    {ia['median_QK_distance_correlation']:+.3f}")
    print()
    print(f"trained-init median label gain:      {aggregate['trained_minus_init_median_label_gain']:+.4f}")
    print(f"paired trained label better:         {aggregate['fraction_paired_label_gain_trained_better']:.3f}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
