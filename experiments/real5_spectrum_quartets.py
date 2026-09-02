from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outotesti.geometry import spectrum_matched_quartet_stability_audit


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
    head_dim = hidden // num_heads

    out = []
    for layer in sorted(layers):
        Q = layers[layer]["Q"].detach().float().cpu().numpy().astype(float)
        K = layers[layer]["K"].detach().float().cpu().numpy().astype(float)
        for h in range(num_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            M = (Q[sl, :].T @ K[sl, :]) / math.sqrt(head_dim)
            out.append((int(layer), int(h), M))
    return out, num_heads, head_dim


def audit_model(
    model,
    *,
    controls: int,
    splits: int,
    quartets: int,
    seed: int,
):
    operators, num_heads, head_dim = head_score_operators(model)
    rows = []

    for idx, (layer, head, M) in enumerate(operators):
        audit = spectrum_matched_quartet_stability_audit(
            M,
            controls=controls,
            splits=splits,
            quartets=quartets,
            seed=seed + idx * 101,
        )
        row = {
            "layer": layer,
            "head": head,
            "observed_agreement": float(
                audit["observed"]["median_agreement"]
            ),
            "null_median_agreement": float(audit["null_median_agreement"]),
            "agreement_gain": float(audit["agreement_gain"]),
            "stability_z": float(audit["stability_z"]),
            "empirical_p_upper": float(audit["empirical_p_upper"]),
        }
        rows.append(row)

        print(
            f"L{layer} H{head:02d} "
            f"agree={row['observed_agreement']:.3f} "
            f"null={row['null_median_agreement']:.3f} "
            f"gain={row['agreement_gain']:+.3f} "
            f"z={row['stability_z']:+.2f} "
            f"p={row['empirical_p_upper']:.3f}"
        )

    def arr(field):
        return np.asarray([r[field] for r in rows], dtype=float)

    layer_summary = {}
    for layer in range(8):
        sub = [r for r in rows if r["layer"] == layer]
        layer_summary[layer] = {
            "median_gain": float(np.median([r["agreement_gain"] for r in sub])),
            "median_z": float(np.median([r["stability_z"] for r in sub])),
            "count_p_lt_0p05": int(
                np.sum(np.asarray([r["empirical_p_upper"] for r in sub]) < 0.05)
            ),
        }

    aggregate = {
        "head_count": len(rows),
        "num_heads_per_layer": int(num_heads),
        "head_dim": int(head_dim),
        "median_observed_agreement": float(np.median(arr("observed_agreement"))),
        "median_null_agreement": float(np.median(arr("null_median_agreement"))),
        "median_agreement_gain": float(np.median(arr("agreement_gain"))),
        "fraction_gain_positive": float(np.mean(arr("agreement_gain") > 0)),
        "median_stability_z": float(np.median(arr("stability_z"))),
        "mean_stability_z": float(np.mean(arr("stability_z"))),
        "count_p_lt_0p05": int(np.sum(arr("empirical_p_upper") < 0.05)),
        "fraction_p_lt_0p05": float(np.mean(arr("empirical_p_upper") < 0.05)),
        "layers": layer_summary,
    }
    return {"aggregate": aggregate, "heads": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--controls", type=int, default=32)
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--quartets", type=int, default=4096)
    ap.add_argument("--init-seed", type=int, default=20260902)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "real5" / "audit.json",
    )
    args = ap.parse_args()

    trained, fresh = load_trained_and_init(args.model, args.init_seed)

    print("TRAINED")
    trained_result = audit_model(
        trained,
        controls=args.controls,
        splits=args.splits,
        quartets=args.quartets,
        seed=310000,
    )

    print()
    print("RANDOM INIT")
    init_result = audit_model(
        fresh,
        controls=args.controls,
        splits=args.splits,
        quartets=args.quartets,
        seed=620000,
    )

    ta = trained_result["aggregate"]
    ia = init_result["aggregate"]
    nheads = ta["head_count"]

    trows = trained_result["heads"]
    irows = init_result["heads"]
    paired_better = np.asarray(
        [
            t["agreement_gain"] > i["agreement_gain"]
            for t, i in zip(trows, irows)
        ],
        dtype=bool,
    )

    aggregate = {
        "trained": ta,
        "random_initialization": ia,
        "trained_minus_init_median_gain": float(
            ta["median_agreement_gain"] - ia["median_agreement_gain"]
        ),
        "trained_minus_init_median_z": float(
            ta["median_stability_z"] - ia["median_stability_z"]
        ),
        "fraction_paired_gain_trained_better": float(np.mean(paired_better)),
    }

    spectrum_controlled_pass = (
        ta["median_stability_z"] >= 1.0
        and ta["count_p_lt_0p05"] >= int(math.ceil(0.25 * nheads))
        and ta["median_agreement_gain"] >= 0.02
        and ta["fraction_gain_positive"] >= 0.75
    )
    training_specific = (
        aggregate["trained_minus_init_median_gain"] >= 0.02
        and aggregate["fraction_paired_gain_trained_better"] >= 0.75
    )

    if spectrum_controlled_pass and training_specific:
        classification = "SPECTRUM_CONTROLLED_STABLE_QUARTET_TOPOLOGY_PRESENT"
    elif spectrum_controlled_pass:
        classification = "QUARTET_TOPOLOGY_NOT_TRAINING_SPECIFIC"
    else:
        classification = "HELDOUT_TREE_EFFECT_EXPLAINED_BY_SPECTRAL_GEOMETRY"

    summary = {
        "experiment": "REAL5",
        "model": args.model,
        "object": "head-wise gauge-invariant attention score operator M_h",
        "protocol": {
            "controls_per_head": int(args.controls),
            "column_splits_per_head": int(args.splits),
            "quartets_per_split": int(args.quartets),
            "null": (
                "exact singular values preserved, independent left/right "
                "singular-vector orientation randomized"
            ),
            "measurement": (
                "four-point quartet split inferred from train-half columns; "
                "agreement with same quartet split in held-out-half columns"
            ),
            "predeclared_pass": (
                "trained median z>=1, p<.05 on >=25% heads, median agreement "
                "gain>=.02, positive gain>=75%"
            ),
            "predeclared_training_specific": (
                "trained-init median gain>=.02 and trained gain larger on "
                ">=75% paired heads"
            ),
        },
        "aggregate": aggregate,
        "trained": trained_result,
        "random_initialization": init_result,
        "classification": classification,
        "stopping_line": (
            "A negative result closes the tree-topology interpretation of REAL4. "
            "A positive result earns replication on a second independently "
            "trained small model before any size scaling."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("REAL5 aggregate")
    print(f"heads:                              {nheads}")
    print("TRAINED")
    print(f"  observed quartet agreement:       {ta['median_observed_agreement']:.4f}")
    print(f"  exact-spectrum null agreement:    {ta['median_null_agreement']:.4f}")
    print(f"  median agreement gain:            {ta['median_agreement_gain']:+.4f}")
    print(f"  median z:                         {ta['median_stability_z']:+.3f}")
    print(f"  p<.05:                            {ta['count_p_lt_0p05']} / {nheads}")
    print("RANDOM INIT")
    print(f"  median agreement gain:            {ia['median_agreement_gain']:+.4f}")
    print(f"  median z:                         {ia['median_stability_z']:+.3f}")
    print()
    print(f"trained-init median gain:           {aggregate['trained_minus_init_median_gain']:+.4f}")
    print(f"paired trained gain better:         {aggregate['fraction_paired_gain_trained_better']:.3f}")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
