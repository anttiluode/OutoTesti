from __future__ import annotations

import numpy as np

from .metrics import relative_frobenius
from .tree import Tree, leaf_distance_matrix


def _fit_diag_wrappers(W: np.ndarray, K: np.ndarray, *, iterations: int = 80):
    W = np.asarray(W, dtype=float)
    K = np.asarray(K, dtype=float)
    m, n = W.shape
    # Signed diagonal wrappers make the objective bilinear. Starting both
    # factors at +1 can converge to a poor stationary point when the true
    # wrappers have mixed signs. Use the leading SVD of the elementwise ratio
    # W/K as a sign-aware initializer, then optimize the actual weighted error.
    kscale = max(float(np.max(np.abs(K))), 1e-15)
    mask = np.abs(K) >= 1e-8 * kscale
    ratio = np.zeros_like(W)
    ratio[mask] = W[mask] / K[mask]

    U0, s0, Vt0 = np.linalg.svd(ratio, full_matrices=False)
    if len(s0) and s0[0] > 1e-15:
        root = np.sqrt(float(s0[0]))
        a = U0[:, 0] * root
        b = Vt0[0, :] * root
    else:
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

    alpha_grid = np.asarray(alpha_grid, dtype=float)
    alpha_grid.sort()

    def evaluate(alpha_value: float):
        K_value = np.exp(-float(alpha_value) * Dn)
        a_value, b_value, W_hat_value = _fit_diag_wrappers(W, K_value)
        err_value = relative_frobenius(W, W_hat_value)
        return (
            float(err_value),
            float(alpha_value),
            a_value,
            b_value,
            W_hat_value,
            K_value,
        )

    grid_results = [evaluate(float(alpha)) for alpha in alpha_grid]
    best_idx = int(np.argmin([x[0] for x in grid_results]))
    best = grid_results[best_idx]

    # Refine continuously in log(alpha) around the best grid point. This is
    # important for the sanity gate: a matrix generated exactly by this family
    # should not retain projection error merely because alpha missed a grid bin.
    lo_idx = max(0, best_idx - 1)
    hi_idx = min(len(alpha_grid) - 1, best_idx + 1)
    lo = float(np.log(alpha_grid[lo_idx]))
    hi = float(np.log(alpha_grid[hi_idx]))

    if hi > lo:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        x1 = hi - (hi - lo) / phi
        x2 = lo + (hi - lo) / phi
        r1 = evaluate(float(np.exp(x1)))
        r2 = evaluate(float(np.exp(x2)))
        for _ in range(36):
            if r1[0] <= r2[0]:
                hi = x2
                x2 = x1
                r2 = r1
                x1 = hi - (hi - lo) / phi
                r1 = evaluate(float(np.exp(x1)))
            else:
                lo = x1
                x1 = x2
                r1 = r2
                x2 = lo + (hi - lo) / phi
                r2 = evaluate(float(np.exp(x2)))
        for candidate in (r1, r2, evaluate(float(np.exp(0.5 * (lo + hi))))):
            if candidate[0] < best[0]:
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
