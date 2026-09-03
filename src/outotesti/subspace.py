from __future__ import annotations

import numpy as np


def row_subspace_basis(W: np.ndarray, *, rank: int | None = None) -> np.ndarray:
    """Return an orthonormal basis for the row subspace as ambient-space columns.

    For W with shape (k, d), returns U with shape (d, r), U.T @ U = I.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be 2-D")

    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    if rank is None:
        tol = max(W.shape) * np.finfo(float).eps * max(float(s[0]) if len(s) else 0.0, 1.0)
        r = int(np.sum(s > tol))
    else:
        r = int(rank)

    if r <= 0 or r > min(W.shape):
        raise ValueError("invalid row-subspace rank")
    return Vt[:r, :].T


def chordal_subspace_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Normalized Grassmann chordal distance between equal-rank subspaces."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or U.shape[0] != V.shape[0]:
        raise ValueError("bases must be 2-D in the same ambient space")
    if U.shape[1] != V.shape[1]:
        raise ValueError("subspaces must have equal rank")
    k = U.shape[1]
    overlap_sq = float(np.linalg.norm(U.T @ V, ord="fro") ** 2)
    d2 = max(0.0, k - overlap_sq)
    return float(np.sqrt(d2 / max(k, 1)))


def head_row_subspaces(
    W: np.ndarray,
    *,
    num_heads: int,
) -> list[np.ndarray]:
    """Split a projection matrix by output head and return row-subspace bases."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] % num_heads:
        raise ValueError("projection rows must divide evenly into heads")
    head_dim = W.shape[0] // num_heads
    return [
        row_subspace_basis(
            W[h * head_dim : (h + 1) * head_dim, :],
            rank=head_dim,
        )
        for h in range(num_heads)
    ]


def subspace_distance_matrix(bases: list[np.ndarray]) -> np.ndarray:
    n = len(bases)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = chordal_subspace_distance(bases[i], bases[j])
            D[i, j] = d
            D[j, i] = d
    return D


def head_subspace_distance_matrix(
    W: np.ndarray,
    *,
    num_heads: int,
) -> np.ndarray:
    return subspace_distance_matrix(
        head_row_subspaces(W, num_heads=num_heads)
    )


def upper_triangle_correlation(D1: np.ndarray, D2: np.ndarray) -> float:
    D1 = np.asarray(D1, dtype=float)
    D2 = np.asarray(D2, dtype=float)
    if D1.shape != D2.shape or D1.ndim != 2 or D1.shape[0] != D1.shape[1]:
        raise ValueError("distance matrices must be square and equally shaped")
    tri = np.triu_indices(D1.shape[0], 1)
    x = D1[tri]
    y = D2[tri]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-15 or sy < 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
