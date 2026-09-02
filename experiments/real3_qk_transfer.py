from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outotesti.geometry import geometry_null_audit
from outotesti.green import (
    fit_green_operator,
    random_topology_with_lengths,
    relabel_leaves,
)
from outotesti.metrics import channel_distance
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


def projections_by_layer(model):
    out = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or parameter.shape[0] != parameter.shape[1]:
            continue
        if ".q_proj.weight" in name:
            layer = int(name.split(".h.")[1].split(".")[0])
            out.setdefault(layer, {})["Q"] = parameter
        elif ".k_proj.weight" in name:
            layer = int(name.split(".h.")[1].split(".")[0])
            out.setdefault(layer, {})["K"] = parameter
    if sorted(out) != list(range(8)):
        raise RuntimeError(f"expected layers 0..7, got {sorted(out)}")
    for layer in out:
        if set(out[layer]) != {"Q", "K"}:
            raise RuntimeError(f"layer {layer} missing Q/K")
    return out


def fit_transfer(
    target: np.ndarray,
    source_tree,
    *,
    controls: int,
    seed: int,
    leak_grid: np.ndarray,
) -> dict:
    inferred = fit_green_operator(
        target,
        source_tree,
        leak_grid=leak_grid,
        exponents=(1.0,),
        wrapper_iterations=40,
    )

    lengths = np.asarray([w for _, _, w in source_tree.edges], dtype=float)
    rng = np.random.default_rng(seed)

    random_errors = []
    label_errors = []
    for _ in range(controls):
        rt = random_topology_with_lengths(target.shape[0], lengths, rng)
        rr = fit_green_operator(
            target,
            rt,
            leak_grid=leak_grid,
            exponents=(1.0,),
            wrapper_iterations=40,
        )
        random_errors.append(float(rr["error"]))

        perm = rng.permutation(target.shape[0])
        lt = relabel_leaves(source_tree, perm)
        lr = fit_green_operator(
            target,
            lt,
            leak_grid=leak_grid,
            exponents=(1.0,),
            wrapper_iterations=40,
        )
        label_errors.append(float(lr["error"]))

    random_errors = np.asarray(random_errors, dtype=float)
    label_errors = np.asarray(label_errors, dtype=float)
    err = float(inferred["error"])

    return {
        "error": err,
        "random_topology_median_error": float(np.median(random_errors)),
        "random_topology_gain": float(np.median(random_errors) - err),
        "random_topology_positive": bool(err < np.median(random_errors)),
        "beats_all_random_topologies": bool(err < np.min(random_errors)),
        "label_shuffle_median_error": float(np.median(label_errors)),
        "label_shuffle_gain": float(np.median(label_errors) - err),
        "label_shuffle_positive": bool(err < np.median(label_errors)),
        "beats_all_label_shuffles": bool(err < np.min(label_errors)),
        "random_topology_errors": random_errors.tolist(),
        "label_shuffle_errors": label_errors.tolist(),
        "leak_ratio": float(inferred["leak_ratio"]),
    }


def one_model(model, *, controls: int, geometry_controls: int, seed: int):
    layers = projections_by_layer(model)
    leak_grid = np.geomspace(1e-3, 1e3, 17)

    result = {}
    for layer in range(8):
        Q = layers[layer]["Q"].detach().float().cpu().numpy().astype(float)
        K = layers[layer]["K"].detach().float().cpu().numpy().astype(float)

        tq = neighbor_joining(channel_distance(Q))
        tk = neighbor_joining(channel_distance(K))

        q_geom = geometry_null_audit(
            Q, controls=geometry_controls, quartets=4096, seed=seed + 100 * layer + 1
        )
        k_geom = geometry_null_audit(
            K, controls=geometry_controls, quartets=4096, seed=seed + 100 * layer + 2
        )

        q_to_k = fit_transfer(
            K,
            tq,
            controls=controls,
            seed=seed + 1000 + 10 * layer + 1,
            leak_grid=leak_grid,
        )
        k_to_q = fit_transfer(
            Q,
            tk,
            controls=controls,
            seed=seed + 1000 + 10 * layer + 2,
            leak_grid=leak_grid,
        )

        result[layer] = {
            "Q_geometry_z": float(q_geom["p95_gap"]["tree_likeness_z"]),
            "K_geometry_z": float(k_geom["p95_gap"]["tree_likeness_z"]),
            "Q_geometry_p": float(q_geom["p95_gap"]["empirical_p_lower"]),
            "K_geometry_p": float(k_geom["p95_gap"]["empirical_p_lower"]),
            "Q_to_K": q_to_k,
            "K_to_Q": k_to_q,
        }

        print(
            f"layer {layer} "
            f"Qz={result[layer]['Q_geometry_z']:+.2f} "
            f"Kz={result[layer]['K_geometry_z']:+.2f} "
            f"Q->K rand={q_to_k['random_topology_gain']:+.4f} "
            f"label={q_to_k['label_shuffle_gain']:+.4f} "
            f"K->Q rand={k_to_q['random_topology_gain']:+.4f} "
            f"label={k_to_q['label_shuffle_gain']:+.4f}"
        )

    transfers = []
    for layer in range(8):
        transfers.extend([result[layer]["Q_to_K"], result[layer]["K_to_Q"]])

    return {
        "layers": result,
        "aggregate": {
            "median_geometry_z": float(
                np.median(
                    [
                        result[l]["Q_geometry_z"] for l in range(8)
                    ]
                    + [
                        result[l]["K_geometry_z"] for l in range(8)
                    ]
                )
            ),
            "geometry_p_lt_0p05": int(
                np.sum(
                    np.asarray(
                        [
                            result[l]["Q_geometry_p"] for l in range(8)
                        ]
                        + [
                            result[l]["K_geometry_p"] for l in range(8)
                        ]
                    )
                    < 0.05
                )
            ),
            "median_random_topology_gain": float(
                np.median([x["random_topology_gain"] for x in transfers])
            ),
            "fraction_random_topology_positive": float(
                np.mean([x["random_topology_positive"] for x in transfers])
            ),
            "beats_all_random_topologies": int(
                np.sum([x["beats_all_random_topologies"] for x in transfers])
            ),
            "median_label_shuffle_gain": float(
                np.median([x["label_shuffle_gain"] for x in transfers])
            ),
            "fraction_label_shuffle_positive": float(
                np.mean([x["label_shuffle_positive"] for x in transfers])
            ),
            "beats_all_label_shuffles": int(
                np.sum([x["beats_all_label_shuffles"] for x in transfers])
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--controls", type=int, default=4)
    ap.add_argument("--geometry-controls", type=int, default=32)
    ap.add_argument("--init-seed", type=int, default=20260902)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "real3" / "audit.json",
    )
    args = ap.parse_args()

    trained, fresh = load_trained_and_init(args.model, args.init_seed)

    print("TRAINED")
    trained_result = one_model(
        trained,
        controls=args.controls,
        geometry_controls=args.geometry_controls,
        seed=30000,
    )

    print()
    print("RANDOM INITIALIZATION")
    init_result = one_model(
        fresh,
        controls=args.controls,
        geometry_controls=args.geometry_controls,
        seed=60000,
    )

    ta = trained_result["aggregate"]
    ia = init_result["aggregate"]

    trained_vs_init_label_gain = float(
        ta["median_label_shuffle_gain"] - ia["median_label_shuffle_gain"]
    )
    trained_vs_init_random_gain = float(
        ta["median_random_topology_gain"] - ia["median_random_topology_gain"]
    )

    paired_label_better = []
    paired_random_better = []
    for layer in range(8):
        for direction in ("Q_to_K", "K_to_Q"):
            t = trained_result["layers"][layer][direction]
            i = init_result["layers"][layer][direction]
            paired_label_better.append(
                t["label_shuffle_gain"] > i["label_shuffle_gain"]
            )
            paired_random_better.append(
                t["random_topology_gain"] > i["random_topology_gain"]
            )

    aggregate = {
        "trained": ta,
        "random_initialization": ia,
        "trained_minus_init_median_label_gain": trained_vs_init_label_gain,
        "trained_minus_init_median_random_gain": trained_vs_init_random_gain,
        "fraction_paired_transfers_trained_label_gain_better": float(
            np.mean(paired_label_better)
        ),
        "fraction_paired_transfers_trained_random_gain_better": float(
            np.mean(paired_random_better)
        ),
    }

    transfer_pass = (
        ta["median_random_topology_gain"] >= 0.005
        and ta["fraction_random_topology_positive"] >= 0.75
        and ta["median_label_shuffle_gain"] >= 0.005
        and ta["fraction_label_shuffle_positive"] >= 0.75
    )
    training_specific = (
        trained_vs_init_label_gain >= 0.005
        and aggregate[
            "fraction_paired_transfers_trained_label_gain_better"
        ] >= 0.75
    )

    if transfer_pass and training_specific:
        classification = "TRAINED_QK_TOPOLOGY_TRANSFERS_OUT_OF_SAMPLE"
    elif transfer_pass:
        classification = "QK_TOPOLOGY_TRANSFERS_BUT_NOT_TRAINING_SPECIFIC"
    else:
        classification = "NO_ROBUST_QK_TOPOLOGY_TRANSFER"

    summary = {
        "experiment": "REAL3",
        "model": args.model,
        "protocol": {
            "directions": ["Q_to_K", "K_to_Q"],
            "reason_QK_are_aligned": (
                "Q and K output coordinates are paired by the attention dot product"
            ),
            "source_topology_frozen_before_target_fit": True,
            "target_fit_parameters": "leak + signed diagonal wrappers only",
            "attacker_1": (
                "random binary topology with source branch-length multiset"
            ),
            "attacker_2": (
                "same exact source topology and lengths with leaf identities permuted"
            ),
            "attacker_3": (
                "fresh random-initialized model from the exact same TinyStories config"
            ),
            "controls_per_transfer": int(args.controls),
            "predeclared_transfer_pass": (
                "trained median gain >= .005 vs both random topology and label "
                "shuffle, positive on >=75% of 16 transfers"
            ),
            "predeclared_training_specific": (
                "trained-minus-init median label gain >= .005 and trained label "
                "gain larger in >=75% of paired transfers"
            ),
        },
        "aggregate": aggregate,
        "trained": trained_result,
        "random_initialization": init_result,
        "classification": classification,
        "stopping_line": (
            "Only a training-specific out-of-sample Q/K transfer earns scaling "
            "or functional interpretation of a recovered topology."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("REAL3 aggregate")
    print("TRAINED")
    print(f"  median random-topology gain:    {ta['median_random_topology_gain']:+.5f}")
    print(f"  random gain positive:           {ta['fraction_random_topology_positive']:.3f}")
    print(f"  median label-shuffle gain:      {ta['median_label_shuffle_gain']:+.5f}")
    print(f"  label gain positive:            {ta['fraction_label_shuffle_positive']:.3f}")
    print(f"  geometry median z:              {ta['median_geometry_z']:+.3f}")
    print()
    print("RANDOM INIT")
    print(f"  median random-topology gain:    {ia['median_random_topology_gain']:+.5f}")
    print(f"  median label-shuffle gain:      {ia['median_label_shuffle_gain']:+.5f}")
    print(f"  geometry median z:              {ia['median_geometry_z']:+.3f}")
    print()
    print(f"trained-init label gain:          {trained_vs_init_label_gain:+.5f}")
    print(f"paired trained label better:      {aggregate['fraction_paired_transfers_trained_label_gain_better']:.3f}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
