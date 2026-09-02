from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outotesti.geometry import geometry_null_audit
from outotesti.green import fit_green_operator, random_topology_with_lengths
from outotesti.metrics import channel_distance
from outotesti.projection import matched_budget_sparse, matched_budget_svd
from outotesti.tree import neighbor_joining


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model


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


def family(name: str) -> str:
    if ".q_proj." in name:
        return "Q"
    if ".k_proj." in name:
        return "K"
    if ".v_proj." in name:
        return "V"
    if ".out_proj." in name:
        return "OUT"
    return "OTHER"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--geometry-controls", type=int, default=64)
    ap.add_argument("--green-random-trees", type=int, default=4)
    ap.add_argument("--quartets", type=int, default=4096)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "real2")
    args = ap.parse_args()

    model = load_model(args.model)
    params = square_projection_parameters(model)
    if len(params) != 32:
        raise RuntimeError(f"expected 32 square attention projections, got {len(params)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    leak_grid = np.geomspace(1e-3, 1e3, 17)

    rows = []
    detail = {}

    for idx, (name, parameter) in enumerate(params):
        W = parameter.detach().float().cpu().numpy().astype(float)
        D = channel_distance(W)
        inferred = neighbor_joining(D)

        geometry = geometry_null_audit(
            W,
            controls=args.geometry_controls,
            quartets=args.quartets,
            seed=2000 + idx,
        )

        green = fit_green_operator(
            W,
            inferred,
            leak_grid=leak_grid,
            exponents=(1.0,),
            wrapper_iterations=40,
        )
        budget = int(green["parameter_budget"])
        svd = matched_budget_svd(W, budget)
        sparse = matched_budget_sparse(W, budget)

        lengths = np.asarray([w for _, _, w in inferred.edges], dtype=float)
        rng = np.random.default_rng(5000 + idx)
        random_errors = []
        for _ in range(args.green_random_trees):
            rt = random_topology_with_lengths(W.shape[0], lengths, rng)
            rf = fit_green_operator(
                W,
                rt,
                leak_grid=leak_grid,
                exponents=(1.0,),
                wrapper_iterations=40,
            )
            random_errors.append(float(rf["error"]))

        random_errors = np.asarray(random_errors, dtype=float)
        random_median = float(np.median(random_errors))
        topology_gain = float(random_median - green["error"])

        row = {
            "name": name,
            "family": family(name),
            "n": int(W.shape[0]),
            "geometry_p95_observed": geometry["p95_gap"]["observed"],
            "geometry_p95_null_median": geometry["p95_gap"]["null_median"],
            "geometry_p95_z": geometry["p95_gap"]["tree_likeness_z"],
            "geometry_p95_empirical_p": geometry["p95_gap"]["empirical_p_lower"],
            "green_error": float(green["error"]),
            "green_random_median_error": random_median,
            "green_topology_gain": topology_gain,
            "green_beats_all_random": bool(green["error"] < np.min(random_errors)),
            "green_leak_ratio": float(green["leak_ratio"]),
            "green_budget": budget,
            "svd_error": float(svd["error"]),
            "svd_rank": int(svd["rank"]),
            "sparse_error": float(sparse["error"]),
        }
        rows.append(row)
        detail[name] = {
            **row,
            "green_random_errors": random_errors.tolist(),
            "geometry": geometry,
        }

        print(
            f"[{idx+1:02d}/32] {name:55s} "
            f"geom-z={row['geometry_p95_z']:+.2f} "
            f"p={row['geometry_p95_empirical_p']:.3f} "
            f"green={row['green_error']:.3f} "
            f"rand={row['green_random_median_error']:.3f} "
            f"gain={row['green_topology_gain']:+.4f} "
            f"svd={row['svd_error']:.3f} "
            f"sparse={row['sparse_error']:.3f}"
        )

    z = np.asarray([r["geometry_p95_z"] for r in rows], dtype=float)
    p = np.asarray([r["geometry_p95_empirical_p"] for r in rows], dtype=float)
    gains = np.asarray([r["green_topology_gain"] for r in rows], dtype=float)
    green_errors = np.asarray([r["green_error"] for r in rows], dtype=float)
    random_errors = np.asarray([r["green_random_median_error"] for r in rows], dtype=float)

    family_summary = {}
    for fam in ("Q", "K", "V", "OUT"):
        subset = [r for r in rows if r["family"] == fam]
        family_summary[fam] = {
            "count": len(subset),
            "median_geometry_z": float(np.median([r["geometry_p95_z"] for r in subset])),
            "fraction_geometry_p_lt_0p05": float(
                np.mean([r["geometry_p95_empirical_p"] < 0.05 for r in subset])
            ),
            "median_green_error": float(np.median([r["green_error"] for r in subset])),
            "median_green_topology_gain": float(
                np.median([r["green_topology_gain"] for r in subset])
            ),
        }

    aggregate = {
        "matrix_count": len(rows),
        "geometry": {
            "median_tree_likeness_z": float(np.median(z)),
            "mean_tree_likeness_z": float(np.mean(z)),
            "fraction_empirical_p_lt_0p05": float(np.mean(p < 0.05)),
            "count_empirical_p_lt_0p05": int(np.sum(p < 0.05)),
        },
        "green": {
            "median_error": float(np.median(green_errors)),
            "median_random_topology_error": float(np.median(random_errors)),
            "median_topology_gain": float(np.median(gains)),
            "fraction_positive_topology_gain": float(np.mean(gains > 0)),
            "fraction_gain_over_0p005": float(np.mean(gains > 0.005)),
            "count_beats_all_random": int(np.sum([r["green_beats_all_random"] for r in rows])),
        },
        "family": family_summary,
    }

    geometry_signal = (
        aggregate["geometry"]["median_tree_likeness_z"] >= 1.0
        and aggregate["geometry"]["count_empirical_p_lt_0p05"] >= 8
    )
    green_signal = (
        aggregate["green"]["median_topology_gain"] >= 0.005
        and aggregate["green"]["fraction_positive_topology_gain"] >= 0.75
    )

    if geometry_signal and green_signal:
        classification = "CHANNEL_GEOMETRY_AND_GREEN_TOPOLOGY_SIGNAL"
    elif geometry_signal:
        classification = "CHANNEL_GEOMETRY_SIGNAL_ONLY"
    elif green_signal:
        classification = "GREEN_TOPOLOGY_SIGNAL_ONLY"
    else:
        classification = "NO_TOPOLOGY_SENSITIVE_SIGNAL_AT_D64"

    summary = {
        "experiment": "REAL2",
        "model": args.model,
        "protocol": {
            "geometry_null": (
                "preserve exact singular values, independently randomize left/right "
                "singular-vector orientations; lower four-point gap is more tree-like"
            ),
            "geometry_controls_per_matrix": int(args.geometry_controls),
            "geometry_quartets": int(args.quartets),
            "green_operator": "diag(a) [(L_tree + leak I)^-1]_leaves diag(b)",
            "green_random_attacker": (
                "random binary topology carrying exactly the inferred tree's "
                "branch-length multiset"
            ),
            "green_random_trees_per_matrix": int(args.green_random_trees),
            "green_leak_grid": leak_grid.tolist(),
            "green_conductance": "g_e proportional to inverse inferred branch length",
            "predeclared_geometry_signal": (
                "median z >= 1 and at least 8/32 matrices empirical p < .05"
            ),
            "predeclared_green_signal": (
                "median inferred-vs-random error gain >= .005 and positive in >=75%"
            ),
        },
        "aggregate": aggregate,
        "classification": classification,
        "matrices": detail,
        "stopping_line": (
            "If neither geometry nor Green topology is topology-sensitive at d=64, "
            "do not scale to larger language models. If only geometry survives, "
            "treat OutoTesti as a geometry diagnostic rather than a weight translator."
        ),
    }

    (args.out_dir / "audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fields = list(rows[0].keys())
    with (args.out_dir / "audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("REAL2 aggregate")
    print(f"geometry median z:                    {aggregate['geometry']['median_tree_likeness_z']:+.3f}")
    print(f"geometry p<.05:                       {aggregate['geometry']['count_empirical_p_lt_0p05']} / 32")
    print(f"Green median error:                   {aggregate['green']['median_error']:.4f}")
    print(f"Green random-topology median error:   {aggregate['green']['median_random_topology_error']:.4f}")
    print(f"Green median topology gain:           {aggregate['green']['median_topology_gain']:+.5f}")
    print(f"Green positive topology gain:         {aggregate['green']['fraction_positive_topology_gain']:.3f}")
    print(f"Green >.005 gain:                     {aggregate['green']['fraction_gain_over_0p005']:.3f}")
    print(f"Green beats all random controls:      {aggregate['green']['count_beats_all_random']} / 32")
    print(f"classification: {classification}")


if __name__ == "__main__":
    main()
