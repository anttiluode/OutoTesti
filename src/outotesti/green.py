from __future__ import annotations

import numpy as np

from .metrics import relative_frobenius
from .projection import _fit_diag_wrappers
from .tree import Tree


def tree_laplacian(tree: Tree, *, exponent: float = 1.0) -> np.ndarray:
    """Weighted Laplacian using conductance ~ inverse branch length^exponent."""
    n = tree.n_nodes
    L = np.zeros((n, n), dtype=float)
    lengths = np.asarray([w for _, _, w in tree.edges if w > 1e-12], dtype=float)
    scale = float(np.median(lengths)) if len(lengths) else 1.0

    for u, v, length in tree.edges:
        ell = max(float(length), scale * 1e-6, 1e-12)
        g = (scale / ell) ** float(exponent)
        L[u, u] += g
        L[v, v] += g
        L[u, v] -= g
        L[v, u] -= g
    return L


def green_leaf_kernel(
    tree: Tree,
    *,
    leak_ratio: float,
    exponent: float = 1.0,
) -> np.ndarray:
    """Leaf-to-leaf Green matrix of a leaky weighted tree network."""
    L = tree_laplacian(tree, exponent=exponent)
    diag = np.diag(L)
    positive = diag[diag > 1e-12]
    base = float(np.median(positive)) if len(positive) else 1.0
    leak = max(float(leak_ratio), 1e-12) * base
    A = L + leak * np.eye(L.shape[0], dtype=float)
    G = np.linalg.inv(A)
    return G[: tree.n_leaves, : tree.n_leaves]


def fit_green_operator(
    W: np.ndarray,
    tree: Tree,
    *,
    leak_grid: np.ndarray | None = None,
    exponents: tuple[float, ...] = (1.0,),
) -> dict:
    """Fit W ~= diag(a) G_tree(leak) diag(b).

    The tree topology and branch lengths are fixed. Only leak, an optional
    conductance exponent, and signed diagonal input/output wrappers are fit.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("Green audit requires a square matrix")
    if tree.n_leaves != W.shape[0]:
        raise ValueError("tree leaf count must match matrix dimension")

    if leak_grid is None:
        leak_grid = np.geomspace(1e-3, 1e3, 31)

    best = None
    for exponent in exponents:
        for leak_ratio in leak_grid:
            K = green_leaf_kernel(
                tree,
                leak_ratio=float(leak_ratio),
                exponent=float(exponent),
            )
            # Normalize only for conditioning; diagonal wrappers absorb scale.
            K = K / max(float(np.linalg.norm(K)), 1e-15)
            a, b, W_hat = _fit_diag_wrappers(W, K)
            err = relative_frobenius(W, W_hat)
            candidate = (
                float(err),
                float(leak_ratio),
                float(exponent),
                a,
                b,
                W_hat,
                K,
            )
            if best is None or err < best[0]:
                best = candidate

    assert best is not None
    err, leak_ratio, exponent, a, b, W_hat, K = best
    parameter_budget = len(tree.edges) + len(a) + len(b) + 2
    return {
        "error": err,
        "leak_ratio": leak_ratio,
        "conductance_exponent": exponent,
        "a": a,
        "b": b,
        "W_hat": W_hat,
        "kernel": K,
        "parameter_budget": int(parameter_budget),
        "edge_count": int(len(tree.edges)),
        "note": (
            "budget counts branch lengths + two diagonal wrappers + leak + "
            "conductance exponent; discrete topology encoding is omitted"
        ),
    }


def random_topology_with_lengths(
    n_leaves: int,
    lengths: np.ndarray,
    rng: np.random.Generator,
) -> Tree:
    """Random unrooted binary topology carrying the same branch-length multiset."""
    from .tree import random_binary_tree

    prototype = random_binary_tree(n_leaves, rng)
    lengths = np.asarray(lengths, dtype=float).copy()
    if len(lengths) != len(prototype.edges):
        raise ValueError("length multiset must have 2*n_leaves-3 values")
    rng.shuffle(lengths)

    edges = tuple(
        (u, v, float(lengths[i]))
        for i, (u, v, _) in enumerate(prototype.edges)
    )
    return Tree(n_leaves, edges)
