from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outotesti.audit import audit_matrix
from outotesti.metrics import four_point_score
from outotesti.projection import fit_tree_kernel
from outotesti.tree import leaf_distance_matrix, random_binary_tree


def main() -> None:
    rng = np.random.default_rng(20260902)
    n = 16
    true_tree = random_binary_tree(n, rng)
    D = leaf_distance_matrix(true_tree)
    four_true = four_point_score(D)

    alpha = 1.7
    K = np.exp(-alpha * D / np.median(D[D > 0]))
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    W_tree = a[:, None] * K * b[None, :]

    oracle = fit_tree_kernel(W_tree, true_tree)
    inferred = audit_matrix(W_tree, random_tree_controls=4, seed=4)

    dense = rng.normal(size=(n, n))
    dense_audit = audit_matrix(dense, random_tree_controls=4, seed=3)

    summary = {
        "exact_tree_metric_four_point": four_true,
        "oracle_tree_matrix_error": float(oracle["error"]),
        "inferred_tree_matrix_error": float(inferred["tree"]["error"]),
        "inferred_tree_random_control_median": float(inferred["random_tree"]["median_error"]),
        "dense_tree_error": float(dense_audit["tree"]["error"]),
        "dense_svd_error": float(dense_audit["svd"]["error"]),
        "dense_sparse_error": float(dense_audit["sparse"]["error"]),
        "stopping_line": (
            "Oracle recovery proves the operator family is internally coherent. "
            "Inferring the generating topology from row geometry is a separate "
            "problem and is not guaranteed even for a tree-generated matrix."
        ),
    }

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "sanity.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("OutoTesti synthetic sanity")
    print(f"exact tree four-point max gap: {four_true['max_gap']:.3e}")
    print(f"oracle reconstruction error:   {oracle['error']:.3e}")
    print(f"inferred tree error:           {inferred['tree']['error']:.4f}")
    print(f"inferred random-tree median:   {inferred['random_tree']['median_error']:.4f}")
    print(f"random dense tree error:       {dense_audit['tree']['error']:.4f}")
    print(f"random dense SVD error:        {dense_audit['svd']['error']:.4f}")

    assert four_true["max_gap"] < 1e-10
    assert oracle["error"] < 1e-6
    assert np.isfinite(inferred["tree"]["error"])
    assert np.isfinite(dense_audit["tree"]["error"])


if __name__ == "__main__":
    main()
