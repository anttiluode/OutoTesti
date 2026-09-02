from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outotesti.geometry import geometry_null_audit
from outotesti.green import fit_green_operator, random_topology_with_lengths, relabel_leaves
from outotesti.metrics import channel_distance
from outotesti.stability import split_half_tree_stability
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


def head_score_operators(model):
    layers = qk_by_layer(model)
    num_heads = int(
        getattr(model.config, "num_heads", getattr(model.config, "num_attention_heads"))
    )
    hidden = int(model.config.hidden_size)
    if hidden % num_heads:
        raise RuntimeError("hidden size is not divisible by num_heads")
    head_dim = hidden // num_heads

    out = {}
    for layer in sorted(layers):
        Q = layers[layer]["Q"].detach().float().cpu().numpy().astype(float)
        K = layers[layer]["K"].detach().float().cpu().numpy().astype(float)
        heads = {}
        for h in range(num_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            Qh = Q[sl, :]
            Kh = K[sl, :]
            # PyTorch Linear uses y=x W^T. For one attention head:
            # q=x W_Q,h^T, k=y W_K,h^T, so q.k = x M_h y^T.
            M = (Qh.T @ Kh) / math.sqrt(head_dim)
            heads[h] = M
        out[layer] = heads
    return out, num_heads, head_dim


def green_self_audit(
    M: np.ndarray,
    *,
    controls: int,
    seed: int,
    leak_grid: np.ndarray,
) -> dict:
    tree = neighbor_joining(channel_distance(M))
    fit = fit_green_operator(
        M,
        tree,
        leak_grid=leak_grid,
        exponents=(1.0,),
        wrapper_iterations=20,
    )

    lengths = np.asarray([w for _, _, w in tree.edges], dtype=float)
    rng = np.random.default_rng(seed)
    random_errors = []
    label_errors = []

    for _ in range(controls):
        rt = random_topology_with_lengths(M.shape[0], lengths, rng)
        rf = fit_green_operator(
            M,
            rt,
            leak_grid=leak_grid,
            exponents=(1.0,),
            wrapper_iterations=20,
        )
        random_errors.append(float(rf["error"]))

        lt = relabel_leaves(tree, rng.permutation(M.shape[0]))
        lf = fit_green_operator(
            M,
            lt,
            leak_grid=leak_grid,
            exponents=(1.0,),
            wrapper_iterations=20,
        )
        label_errors.append(float(lf["error"]))

    random_errors = np.asarray(random_errors, dtype=float)
    label_errors = np.asarray(label_errors, dtype=float)
    err = float(fit["error"])
    return {
        "error": err,
        "random_topology_median_error": float(np.median(random_errors)),
        "random_topology_gain": float(np.median(random_errors) - err),
        "label_shuffle_median_error": float(np.median(label_errors)),
        "label_shuffle_gain": float(np.median(label_errors) - err),
        "random_positive": bool(err < np.median(random_errors)),
        "label_positive": bool(err < np.median(label_errors)),
        "leak_ratio": float(fit["leak_ratio"]),
    }


def one_model(
    model,
    *,
    geometry_controls: int,
    split_controls: int,
    green_controls: int,
    seed: int,
):
    operators, num_heads, head_dim = head_score_operators(model)
    leak_grid = np.geomspace(1e-3, 1e3, 17)

    detail = {}
    flat = []

    for layer in sorted(operators):
        layer_rows = {}
        for head, M in operators[layer].items():
            key_seed = seed + layer * 1000 + head * 20

            geom = geometry_null_audit(
                M,
                controls=geometry_controls,
                quartets=4096,
                seed=key_seed + 1,
            )
            stable = split_half_tree_stability(
                M,
                splits=4,
                controls=split_controls,
                seed=key_seed + 2,
            )
            green = green_self_audit(
                M,
                controls=green_controls,
                seed=key_seed + 3,
                leak_grid=leak_grid,
            )

            row = {
                "layer": int(layer),
                "head": int(head),
                "geometry_z": float(geom["p95_gap"]["tree_likeness_z"]),
                "geometry_p": float(geom["p95_gap"]["empirical_p_lower"]),
                "heldout_test_error": float(stable["median_test_error"]),
                "heldout_label_gain": float(stable["median_label_shuffle_gain"]),
                "heldout_random_gain": float(stable["median_random_topology_gain"]),
                "heldout_label_positive_fraction": float(
                    stable["fraction_splits_label_positive"]
                ),
                "heldout_random_positive_fraction": float(
                    stable["fraction_splits_random_positive"]
                ),
                "green_error": float(green["error"]),
                "green_label_gain": float(green["label_shuffle_gain"]),
                "green_random_gain": float(green["random_topology_gain"]),
                "green_label_positive": bool(green["label_positive"]),
                "green_random_positive": bool(green["random_positive"]),
            }
            layer_rows[head] = {
                **row,
                "geometry": geom,
                "stability": stable,
                "green": green,
            }
            flat.append(row)

            print(
                f"L{layer} H{head:02d} "
                f"z={row['geometry_z']:+.2f} "
                f"held-label={row['heldout_label_gain']:+.3f} "
                f"held-rand={row['heldout_random_gain']:+.3f} "
                f"green-label={row['green_label_gain']:+.4f} "
                f"green-rand={row['green_random_gain']:+.4f}"
            )
        detail[layer] = layer_rows

    def arr(field):
        return np.asarray([x[field] for x in flat], dtype=float)

    aggregate = {
        "head_count": len(flat),
        "num_heads_per_layer": int(num_heads),
        "head_dim": int(head_dim),
        "geometry": {
            "median_z": float(np.median(arr("geometry_z"))),
            "mean_z": float(np.mean(arr("geometry_z"))),
            "count_p_lt_0p05": int(np.sum(arr("geometry_p") < 0.05)),
            "fraction_p_lt_0p05": float(np.mean(arr("geometry_p") < 0.05)),
        },
        "heldout": {
            "median_label_gain": float(np.median(arr("heldout_label_gain"))),
            "fraction_label_gain_positive": float(
                np.mean(arr("heldout_label_gain") > 0)
            ),
            "median_random_gain": float(np.median(arr("heldout_random_gain"))),
            "fraction_random_gain_positive": float(
                np.mean(arr("heldout_random_gain") > 0)
            ),
        },
        "green": {
            "median_error": float(np.median(arr("green_error"))),
            "median_label_gain": float(np.median(arr("green_label_gain"))),
            "fraction_label_gain_positive": float(
                np.mean(arr("green_label_gain") > 0)
            ),
            "median_random_gain": float(np.median(arr("green_random_gain"))),
            "fraction_random_gain_positive": float(
                np.mean(arr("green_random_gain") > 0)
            ),
        },
    }
    return {"aggregate": aggregate, "heads": detail, "flat": flat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--geometry-controls", type=int, default=32)
    ap.add_argument("--split-controls", type=int, default=4)
    ap.add_argument("--green-controls", type=int, default=2)
    ap.add_argument("--init-seed", type=int, default=20260902)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "real4" / "audit.json",
    )
    args = ap.parse_args()

    trained, fresh = load_trained_and_init(args.model, args.init_seed)

    print("TRAINED HEAD SCORE OPERATORS")
    trained_result = one_model(
        trained,
        geometry_controls=args.geometry_controls,
        split_controls=args.split_controls,
        green_controls=args.green_controls,
        seed=100000,
    )

    print()
    print("RANDOM-INIT HEAD SCORE OPERATORS")
    init_result = one_model(
        fresh,
        geometry_controls=args.geometry_controls,
        split_controls=args.split_controls,
        green_controls=args.green_controls,
        seed=200000,
    )

    ta = trained_result["aggregate"]
    ia = init_result["aggregate"]
    nheads = ta["head_count"]

    tflat = trained_result["flat"]
    iflat = init_result["flat"]
    if len(tflat) != len(iflat):
        raise RuntimeError("trained/init head counts differ")

    paired_heldout_better = np.asarray(
        [
            t["heldout_label_gain"] > i["heldout_label_gain"]
            for t, i in zip(tflat, iflat)
        ],
        dtype=bool,
    )
    paired_green_better = np.asarray(
        [
            t["green_label_gain"] > i["green_label_gain"]
            for t, i in zip(tflat, iflat)
        ],
        dtype=bool,
    )

    aggregate = {
        "trained": ta,
        "random_initialization": ia,
        "trained_minus_init_geometry_median_z": float(
            ta["geometry"]["median_z"] - ia["geometry"]["median_z"]
        ),
        "trained_minus_init_heldout_label_gain": float(
            ta["heldout"]["median_label_gain"]
            - ia["heldout"]["median_label_gain"]
        ),
        "trained_minus_init_green_label_gain": float(
            ta["green"]["median_label_gain"]
            - ia["green"]["median_label_gain"]
        ),
        "fraction_paired_heldout_label_gain_trained_better": float(
            np.mean(paired_heldout_better)
        ),
        "fraction_paired_green_label_gain_trained_better": float(
            np.mean(paired_green_better)
        ),
    }

    geometry_pass = (
        ta["geometry"]["median_z"] >= 1.0
        and ta["geometry"]["count_p_lt_0p05"] >= int(math.ceil(0.25 * nheads))
    )
    heldout_pass = (
        ta["heldout"]["median_label_gain"] >= 0.01
        and ta["heldout"]["fraction_label_gain_positive"] >= 0.75
        and ta["heldout"]["median_random_gain"] >= 0.01
        and ta["heldout"]["fraction_random_gain_positive"] >= 0.75
    )
    training_specific = (
        aggregate["trained_minus_init_heldout_label_gain"] >= 0.01
        and aggregate[
            "fraction_paired_heldout_label_gain_trained_better"
        ] >= 0.75
    )
    green_pass = (
        ta["green"]["median_label_gain"] >= 0.005
        and ta["green"]["fraction_label_gain_positive"] >= 0.75
        and ta["green"]["median_random_gain"] >= 0.005
        and ta["green"]["fraction_random_gain_positive"] >= 0.75
        and aggregate["trained_minus_init_green_label_gain"] >= 0.005
    )

    if geometry_pass and heldout_pass and training_specific and green_pass:
        classification = "GAUGE_INVARIANT_HEAD_GREEN_GEOMETRY_PRESENT"
    elif geometry_pass and heldout_pass and training_specific:
        classification = "GAUGE_INVARIANT_HELDOUT_TREE_GEOMETRY_PRESENT"
    else:
        classification = "NO_ROBUST_GAUGE_INVARIANT_TREE_GEOMETRY"

    summary = {
        "experiment": "REAL4",
        "model": args.model,
        "object": "head-wise gauge-invariant attention score bilinear M_h",
        "definition": "M_h = W_Q,h^T W_K,h / sqrt(head_dim)",
        "protocol": {
            "spectrum_controls_per_head": int(args.geometry_controls),
            "heldout_column_splits_per_head": 4,
            "heldout_controls_per_split": int(args.split_controls),
            "green_controls_per_head": int(args.green_controls),
            "random_initialization_same_config": True,
            "predeclared_geometry_pass": (
                "trained median z>=1 and p<.05 on >=25% of heads"
            ),
            "predeclared_heldout_pass": (
                "median heldout gain >=.01 vs both leaf shuffle and random "
                "topology, positive on >=75% of heads"
            ),
            "predeclared_training_specific": (
                "trained-init heldout label gain >=.01 and trained larger on "
                ">=75% of paired heads"
            ),
            "predeclared_green_pass": (
                "median Green gain >=.005 vs label/random, positive >=75%, "
                "and trained-init label gain >=.005"
            ),
        },
        "aggregate": aggregate,
        "classification": classification,
        "trained_heads": trained_result["heads"],
        "random_init_heads": init_result["heads"],
        "stopping_line": (
            "Only a training-specific held-out signal on the gauge-invariant "
            "score operator earns scaling. Green reconstruction remains a "
            "separate stronger requirement."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("REAL4 aggregate")
    print(f"heads:                                {nheads}")
    print(f"head_dim:                             {ta['head_dim']}")
    print("TRAINED")
    print(f"  geometry median z:                  {ta['geometry']['median_z']:+.3f}")
    print(f"  geometry p<.05:                     {ta['geometry']['count_p_lt_0p05']} / {nheads}")
    print(f"  heldout label gain:                 {ta['heldout']['median_label_gain']:+.4f}")
    print(f"  heldout random gain:                {ta['heldout']['median_random_gain']:+.4f}")
    print(f"  Green label gain:                   {ta['green']['median_label_gain']:+.5f}")
    print(f"  Green random gain:                  {ta['green']['median_random_gain']:+.5f}")
    print("RANDOM INIT")
    print(f"  geometry median z:                  {ia['geometry']['median_z']:+.3f}")
    print(f"  heldout label gain:                 {ia['heldout']['median_label_gain']:+.4f}")
    print(f"  Green label gain:                   {ia['green']['median_label_gain']:+.5f}")
    print()
    print(f"trained-init heldout label gain:      {aggregate['trained_minus_init_heldout_label_gain']:+.4f}")
    print(f"paired heldout trained better:        {aggregate['fraction_paired_heldout_label_gain_trained_better']:.3f}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
