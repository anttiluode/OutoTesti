from __future__ import annotations

import numpy as np


def centered_cosine_distance(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pairwise distance between row vectors after per-row centering.

    Returns (1-cosine)/2 in [0,1] up to roundoff.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    Xc = X - np.mean(X, axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    Xn = Xc / np.maximum(norms, eps)
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    D = 0.5 * (1.0 - sim)
    np.fill_diagonal(D, 0.0)
    return D


def upper_triangle_correlation(D1: np.ndarray, D2: np.ndarray) -> float:
    D1 = np.asarray(D1, dtype=float)
    D2 = np.asarray(D2, dtype=float)
    if D1.shape != D2.shape or D1.ndim != 2 or D1.shape[0] != D1.shape[1]:
        raise ValueError("distance matrices must be equally shaped and square")
    tri = np.triu_indices(D1.shape[0], 1)
    x = D1[tri]
    y = D2[tri]
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def label_permutation_correlation_test(
    predictor_D: np.ndarray,
    target_D: np.ndarray,
    *,
    controls: int = 512,
    seed: int = 0,
) -> dict:
    observed = upper_triangle_correlation(predictor_D, target_D)
    rng = np.random.default_rng(seed)
    n = predictor_D.shape[0]
    null = np.empty(controls, dtype=float)

    for i in range(controls):
        p = rng.permutation(n)
        permuted = target_D[np.ix_(p, p)]
        null[i] = upper_triangle_correlation(predictor_D, permuted)

    std = float(np.std(null, ddof=1)) if controls > 1 else 0.0
    return {
        "observed": float(observed),
        "null_median": float(np.median(null)),
        "null_mean": float(np.mean(null)),
        "null_std": std,
        "z": float(
            (observed - np.mean(null)) / max(std, 1e-12)
        ) if controls > 1 else 0.0,
        "empirical_p_upper": float(
            (1 + np.sum(null >= observed)) / (controls + 1)
        ),
    }


def combine_qk_distances(Dq: np.ndarray, Dk: np.ndarray) -> np.ndarray:
    Dq = np.asarray(Dq, dtype=float)
    Dk = np.asarray(Dk, dtype=float)
    if Dq.shape != Dk.shape:
        raise ValueError("Q/K distance matrices must match")
    return 0.5 * (Dq + Dk)
