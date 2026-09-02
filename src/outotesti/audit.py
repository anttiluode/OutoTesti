from __future__ import annotations

import numpy as np

from .metrics import channel_distance, four_point_score
from .projection import fit_tree_kernel, matched_budget_sparse, matched_budget_svd
from .tree import neighbor_joining, random_binary_tree


def audit_matrix(W: np.ndarray, *, random_tree_controls: int = 8, seed: int = 0) -> dict:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("OutoTesti v0.1 audits square matrices")

    D = channel_distance(W)
    four = four_point_score(D, seed=seed)
    inferred = neighbor_joining(D)
    tree_fit = fit_tree_kernel(W, inferred)
    budget = int(tree_fit["parameter_budget"])

    svd = matched_budget_svd(W, budget)
    sparse = matched_budget_sparse(W, budget)

    rng = np.random.default_rng(seed)
    random_errors = []
    for _ in range(random_tree_controls):
        rt = random_binary_tree(W.shape[0], rng)
        random_errors.append(float(fit_tree_kernel(W, rt)["error"]))

    return {
        "shape": list(W.shape),
        "frobenius_norm": float(np.linalg.norm(W)),
        "channel_four_point": four,
        "tree": {
            "error": float(tree_fit["error"]),
            "alpha": float(tree_fit["alpha"]),
            "parameter_budget": budget,
            "edge_count": int(tree_fit["edge_count"]),
            "W_hat": tree_fit["W_hat"],
            "note": tree_fit["note"],
        },
        "svd": {
            "error": float(svd["error"]),
            "rank": int(svd["rank"]),
            "parameter_budget": int(svd["parameter_budget"]),
            "W_hat": svd["W_hat"],
        },
        "sparse": {
            "error": float(sparse["error"]),
            "nonzeros": int(sparse["nonzeros"]),
            "parameter_budget": int(sparse["parameter_budget"]),
            "W_hat": sparse["W_hat"],
        },
        "random_tree": {
            "controls": int(random_tree_controls),
            "median_error": float(np.median(random_errors)),
            "min_error": float(np.min(random_errors)),
            "max_error": float(np.max(random_errors)),
            "errors": random_errors,
        },
    }


def jsonable_audit(result: dict) -> dict:
    out = {}
    for key, value in result.items():
        if isinstance(value, dict):
            out[key] = jsonable_audit(value)
        elif isinstance(value, np.ndarray):
            continue
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out
