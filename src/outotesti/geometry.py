from __future__ import annotations

import numpy as np

from .metrics import channel_distance


def sampled_quartets(n: int, count: int, seed: int = 0) -> np.ndarray:
    if n < 4:
        return np.empty((0, 4), dtype=int)
    rng = np.random.default_rng(seed)
    qs = set()
    target = min(count, n * (n - 1) * (n - 2) * (n - 3) // 24)
    while len(qs) < target:
        qs.add(tuple(sorted(rng.choice(n, size=4, replace=False).tolist())))
    return np.asarray(sorted(qs), dtype=int)


def four_point_gaps_for_quartets(D: np.ndarray, quartets: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    q = np.asarray(quartets, dtype=int)
    if q.size == 0:
        return np.empty(0, dtype=float)
    i, j, k, l = q.T
    sums = np.stack(
        [
            D[i, j] + D[k, l],
            D[i, k] + D[j, l],
            D[i, l] + D[j, k],
        ],
        axis=1,
    )
    sums.sort(axis=1)
    return (sums[:, 2] - sums[:, 1]) / np.maximum(sums[:, 2], 1e-12)


def spectrum_randomized_matrix(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserve singular values exactly, randomize left/right singular vectors."""
    W = np.asarray(W, dtype=float)
    m, n = W.shape
    s = np.linalg.svd(W, compute_uv=False)

    A = rng.normal(size=(m, m))
    B = rng.normal(size=(n, n))
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)

    S = np.zeros((m, n), dtype=float)
    r = min(m, n, len(s))
    S[np.arange(r), np.arange(r)] = s[:r]
    return Qa @ S @ Qb.T


def geometry_null_audit(
    W: np.ndarray,
    *,
    controls: int = 64,
    quartets: int = 4096,
    seed: int = 0,
) -> dict:
    """Ask whether row-channel geometry is more additive-tree-like than a
    dimension- and singular-spectrum-matched random orientation null.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    qs = sampled_quartets(n, quartets, seed=seed)
    observed_gaps = four_point_gaps_for_quartets(channel_distance(W), qs)
    observed = {
        "median_gap": float(np.median(observed_gaps)),
        "p95_gap": float(np.quantile(observed_gaps, 0.95)),
    }

    rng = np.random.default_rng(seed + 1)
    null_median = np.empty(controls, dtype=float)
    null_p95 = np.empty(controls, dtype=float)

    for i in range(controls):
        W0 = spectrum_randomized_matrix(W, rng)
        gaps = four_point_gaps_for_quartets(channel_distance(W0), qs)
        null_median[i] = np.median(gaps)
        null_p95[i] = np.quantile(gaps, 0.95)

    def summarize(obs: float, vals: np.ndarray) -> dict:
        # Lower gap = more tree-like.
        return {
            "observed": float(obs),
            "null_median": float(np.median(vals)),
            "null_mean": float(np.mean(vals)),
            "null_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "empirical_p_lower": float(
                (1 + np.sum(vals <= obs)) / (len(vals) + 1)
            ),
            "tree_likeness_z": float(
                (np.mean(vals) - obs) / max(np.std(vals, ddof=1), 1e-12)
            ) if len(vals) > 1 else 0.0,
        }

    return {
        "controls": int(controls),
        "quartets": int(len(qs)),
        "null": "exact singular spectrum, independent Haar-like left/right orientations",
        "median_gap": summarize(observed["median_gap"], null_median),
        "p95_gap": summarize(observed["p95_gap"], null_p95),
    }
