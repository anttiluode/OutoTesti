from __future__ import annotations

import numpy as np

from .metrics import relative_frobenius
from .tree import Tree, leaf_distance_matrix


def _fit_diag_wrappers(W: np.ndarray, K: np.ndarray, *, iterations: int = 80):
    W = np.asarray(W, dtype=float)
    K = np.asarray(K, dtype=float)
    m, n = W.shape
    b = np.ones(n, dtype=float)
    a = np.ones(m, dtype=float)

    for _ in range(iterations):
        KB = K * b[None, :]
        a = np.sum(W * KB, axis=1) / (np.sum(KB * KB, axis=1) + 1e-15)

        AK = a[:, None] * K
        b = np.sum(W * AK, axis=0) / (np.sum(AK * AK, axis=0) + 1e-15)

        scale = np.sqrt(max(np.mean(b * b), 1e-15))
        b /= scale
        a *= scale

    W_hat = a[:, None] * K * b[None, :]
    return a, b, W_hat


def fit_tree_kernel(W: np.ndarray, tree: Tree, *, alpha_grid=None) -> dict:
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("v0.1 audits square matrices only")
    if tree.n_leaves != W.shape[0]:
        raise ValueError("tree leaf count must match W")

    D = leaf_distance_matrix(tree)
    nz = D[D > 1e-12]
    scale = float(np.median(nz)) if len(nz) else 1.0
    Dn = D / max(scale, 1e-12)

    if alpha_grid is None:
        alpha_grid = np.geomspace(0.05, 20.0, 49)

    best = None
    for alpha in alpha_grid:
        K = np.exp(-float(alpha) * Dn)
        a, b, W_hat = _fit_diag_wrappers(W, K)
        err = relative_frobenius(W, W_hat)
        candidate = (err, float(alpha), a, b, W_hat, K)
        if best is None or err < best[0]:
            best = candidate

    err, alpha, a, b, W_hat, K = best
    edge_count = len(tree.edges)
    parameter_budget = edge_count + len(a) + len(b) + 1
    return {
        "error": float(err),
        "alpha": float(alpha),
        "a": a,
        "b": b,
        "W_hat": W_hat,
        "kernel": K,
        "tree_distance": D,
        "parameter_budget": int(parameter_budget),
        "edge_count": int(edge_count),
        "note": (
            "budget counts branch lengths + diagonal wrappers + alpha; "
            "it omits the discrete topology encoding cost"
        ),
    }


def matched_budget_svd(W: np.ndarray, budget: int) -> dict:
    W = np.asarray(W, dtype=float)
    m, n = W.shape
    per_rank = m + n + 1
    rank = max(1, min(min(m, n), int(budget // per_rank)))
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    W_hat = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    return {
        "rank": int(rank),
        "parameter_budget": int(rank * per_rank),
        "error": relative_frobenius(W, W_hat),
        "W_hat": W_hat,
    }


def matched_budget_sparse(W: np.ndarray, budget: int) -> dict:
    W = np.asarray(W, dtype=float)
    k = max(1, min(W.size, int(budget)))
    idx = np.argpartition(np.abs(W).ravel(), -k)[-k:]
    W_hat = np.zeros_like(W)
    W_hat.ravel()[idx] = W.ravel()[idx]
    return {
        "nonzeros": int(k),
        "parameter_budget": int(k),
        "error": relative_frobenius(W, W_hat),
        "W_hat": W_hat,
    }
