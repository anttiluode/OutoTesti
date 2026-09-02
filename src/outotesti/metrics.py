from __future__ import annotations

import itertools
import numpy as np


def channel_distance(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Euclidean distance between row directions on the unit sphere."""
    X = np.asarray(W, dtype=float)
    if X.ndim != 2:
        raise ValueError("W must be 2-D")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, eps)
    cosine = np.clip(Xn @ Xn.T, -1.0, 1.0)
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - cosine)))
    np.fill_diagonal(D, 0.0)
    return D


def four_point_score(D: np.ndarray, *, max_quartets: int = 20000, seed: int = 0) -> dict:
    """Measure violation of the additive-tree four-point condition."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("D must be square")
    if n < 4:
        return {"quartets": 0, "median_gap": 0.0, "p95_gap": 0.0, "max_gap": 0.0}

    total = n * (n - 1) * (n - 2) * (n - 3) // 24
    if total <= max_quartets:
        quartets = list(itertools.combinations(range(n), 4))
    else:
        rng = np.random.default_rng(seed)
        seen = set()
        while len(seen) < max_quartets:
            seen.add(tuple(sorted(rng.choice(n, size=4, replace=False).tolist())))
        quartets = list(seen)

    gaps = np.empty(len(quartets), dtype=float)
    for qi, (i, j, k, l) in enumerate(quartets):
        sums = np.asarray([
            D[i, j] + D[k, l],
            D[i, k] + D[j, l],
            D[i, l] + D[j, k],
        ], dtype=float)
        sums.sort()
        gaps[qi] = (sums[-1] - sums[-2]) / max(float(sums[-1]), 1e-12)

    return {
        "quartets": int(len(gaps)),
        "median_gap": float(np.median(gaps)),
        "p95_gap": float(np.quantile(gaps, 0.95)),
        "max_gap": float(np.max(gaps)),
    }


def relative_frobenius(W: np.ndarray, W_hat: np.ndarray) -> float:
    W = np.asarray(W, dtype=float)
    W_hat = np.asarray(W_hat, dtype=float)
    return float(np.linalg.norm(W - W_hat) / max(np.linalg.norm(W), 1e-15))
