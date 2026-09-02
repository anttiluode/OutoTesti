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



def quartet_split_ids(D: np.ndarray, quartets: np.ndarray) -> np.ndarray:
    """Return the four-point split selected by each quartet.

    For an additive tree metric the smallest of the three pair-sums identifies
    the bipartition. Values 0/1/2 correspond to ij|kl, ik|jl, il|jk.
    """
    D = np.asarray(D, dtype=float)
    q = np.asarray(quartets, dtype=int)
    if q.size == 0:
        return np.empty(0, dtype=np.int8)
    i, j, k, l = q.T
    sums = np.stack(
        [
            D[i, j] + D[k, l],
            D[i, k] + D[j, l],
            D[i, l] + D[j, k],
        ],
        axis=1,
    )
    return np.argmin(sums, axis=1).astype(np.int8)


def quartet_heldout_stability(
    W: np.ndarray,
    *,
    splits: int = 4,
    quartets: int = 4096,
    seed: int = 0,
) -> dict:
    """Infer quartet splits from half the columns and test them on the other half."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[1] < 4:
        raise ValueError("W must be 2-D with at least four columns")

    rng = np.random.default_rng(seed)
    qs = sampled_quartets(W.shape[0], quartets, seed=seed + 7919)
    agreements = []

    for _ in range(splits):
        cols = rng.permutation(W.shape[1])
        cut = W.shape[1] // 2
        D_train = channel_distance(W[:, cols[:cut]])
        D_test = channel_distance(W[:, cols[cut:]])
        train_split = quartet_split_ids(D_train, qs)
        test_split = quartet_split_ids(D_test, qs)
        agreements.append(float(np.mean(train_split == test_split)))

    return {
        "splits": int(splits),
        "quartets": int(len(qs)),
        "median_agreement": float(np.median(agreements)),
        "mean_agreement": float(np.mean(agreements)),
        "agreements": agreements,
    }


def spectrum_matched_quartet_stability_audit(
    W: np.ndarray,
    *,
    controls: int = 32,
    splits: int = 4,
    quartets: int = 4096,
    seed: int = 0,
) -> dict:
    """Compare held-out quartet topology with exact-spectrum orientation nulls."""
    W = np.asarray(W, dtype=float)
    observed = quartet_heldout_stability(
        W, splits=splits, quartets=quartets, seed=seed
    )

    m, n = W.shape
    s = np.linalg.svd(W, compute_uv=False)
    rng = np.random.default_rng(seed + 104729)
    null = np.empty(controls, dtype=float)

    for c in range(controls):
        Qa, _ = np.linalg.qr(rng.normal(size=(m, m)))
        Qb, _ = np.linalg.qr(rng.normal(size=(n, n)))
        S = np.zeros((m, n), dtype=float)
        r = min(m, n, len(s))
        S[np.arange(r), np.arange(r)] = s[:r]
        W0 = Qa @ S @ Qb.T
        null[c] = quartet_heldout_stability(
            W0, splits=splits, quartets=quartets, seed=seed
        )["median_agreement"]

    obs = float(observed["median_agreement"])
    std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    return {
        "observed": observed,
        "null": "exact singular spectrum, randomized left/right orientation",
        "controls": int(controls),
        "null_median_agreement": float(np.median(null)),
        "null_mean_agreement": float(np.mean(null)),
        "null_std_agreement": std,
        "agreement_gain": float(obs - np.median(null)),
        "stability_z": float(
            (obs - np.mean(null)) / max(std, 1e-12)
        ) if len(null) > 1 else 0.0,
        "empirical_p_upper": float(
            (1 + np.sum(null >= obs)) / (len(null) + 1)
        ),
    }
