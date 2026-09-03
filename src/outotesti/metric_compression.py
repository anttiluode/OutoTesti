from __future__ import annotations

import numpy as np

from .tree import Tree, leaf_distance_matrix, neighbor_joining


def scaled_metric_error(target_D: np.ndarray, predictor_D: np.ndarray) -> float:
    """Relative RMS error after fitting one nonnegative global scale."""
    target_D = np.asarray(target_D, dtype=float)
    predictor_D = np.asarray(predictor_D, dtype=float)
    if target_D.shape != predictor_D.shape:
        raise ValueError("distance matrices must have the same shape")
    n = target_D.shape[0]
    tri = np.triu_indices(n, 1)
    y = target_D[tri]
    x = predictor_D[tri]
    scale = float(np.dot(x, y) / max(np.dot(x, x), 1e-15))
    scale = max(scale, 0.0)
    return float(
        np.linalg.norm(y - scale * x)
        / max(np.linalg.norm(y), 1e-15)
    )


def star_metric_from_distances(D: np.ndarray) -> np.ndarray:
    """Least-squares equal-center star metric D_ij ~= r_i + r_j."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n) or n < 3:
        raise ValueError("D must be square with n>=3")

    s = np.sum(D, axis=1)
    total_pairs = float(np.sum(D[np.triu_indices(n, 1)]))
    R = total_pairs / (n - 1)
    r = (s - R) / (n - 2)
    r = np.maximum(r, 0.0)

    out = r[:, None] + r[None, :]
    np.fill_diagonal(out, 0.0)
    return out


def classical_mds_metric(D: np.ndarray, *, dim: int = 2) -> np.ndarray:
    """Classical MDS reconstruction using the top positive eigenmodes."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("D must be square")
    if dim < 1:
        raise ValueError("dim must be >=1")

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D * D) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    positive = np.maximum(evals[:dim], 0.0)
    X = evecs[:, :dim] * np.sqrt(positive)[None, :]
    diff = X[:, None, :] - X[None, :, :]
    out = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(out, 0.0)
    return out


def nj_tree_metric(D: np.ndarray) -> tuple[Tree, np.ndarray]:
    tree = neighbor_joining(D)
    return tree, leaf_distance_matrix(tree)


def source_predictors(D: np.ndarray) -> dict[str, np.ndarray]:
    """Source-only metric representations used in REAL9."""
    tree, tree_D = nj_tree_metric(D)
    return {
        "raw": np.asarray(D, dtype=float).copy(),
        "star": star_metric_from_distances(D),
        "tree": tree_D,
        "mds2": classical_mds_metric(D, dim=2),
    }


def target_errors(source_D: np.ndarray, target_D: np.ndarray) -> dict[str, float]:
    preds = source_predictors(source_D)
    return {
        name: scaled_metric_error(target_D, pred)
        for name, pred in preds.items()
    }
