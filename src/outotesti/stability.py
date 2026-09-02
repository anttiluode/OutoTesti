from __future__ import annotations

import numpy as np

from .green import random_topology_with_lengths, relabel_leaves
from .metrics import channel_distance
from .tree import Tree, leaf_distance_matrix, neighbor_joining


def scaled_tree_metric_error(D: np.ndarray, tree: Tree) -> float:
    """Relative RMS error after fitting one nonnegative global distance scale."""
    D = np.asarray(D, dtype=float)
    Dt = leaf_distance_matrix(tree)
    n = D.shape[0]
    tri = np.triu_indices(n, 1)
    y = D[tri]
    x = Dt[tri]
    scale = float(np.dot(x, y) / max(np.dot(x, x), 1e-15))
    scale = max(scale, 0.0)
    return float(
        np.linalg.norm(y - scale * x)
        / max(np.linalg.norm(y), 1e-15)
    )


def split_half_tree_stability(
    W: np.ndarray,
    *,
    splits: int = 4,
    controls: int = 4,
    seed: int = 0,
) -> dict:
    """Infer row topology from half the columns and test on held-out columns."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[1] < 4:
        raise ValueError("W must be a 2-D matrix with at least four columns")

    rng = np.random.default_rng(seed)
    per_split = []

    for split in range(splits):
        perm_cols = rng.permutation(W.shape[1])
        cut = W.shape[1] // 2
        train_cols = perm_cols[:cut]
        test_cols = perm_cols[cut:]

        D_train = channel_distance(W[:, train_cols])
        D_test = channel_distance(W[:, test_cols])
        tree = neighbor_joining(D_train)

        observed = scaled_tree_metric_error(D_test, tree)
        lengths = np.asarray([w for _, _, w in tree.edges], dtype=float)

        label_errors = []
        random_errors = []
        for _ in range(controls):
            lt = relabel_leaves(tree, rng.permutation(tree.n_leaves))
            label_errors.append(scaled_tree_metric_error(D_test, lt))

            rt = random_topology_with_lengths(tree.n_leaves, lengths, rng)
            random_errors.append(scaled_tree_metric_error(D_test, rt))

        label_errors = np.asarray(label_errors, dtype=float)
        random_errors = np.asarray(random_errors, dtype=float)
        per_split.append(
            {
                "test_error": float(observed),
                "label_shuffle_median_error": float(np.median(label_errors)),
                "label_shuffle_gain": float(np.median(label_errors) - observed),
                "random_topology_median_error": float(np.median(random_errors)),
                "random_topology_gain": float(np.median(random_errors) - observed),
            }
        )

    return {
        "splits": int(splits),
        "controls_per_split": int(controls),
        "median_test_error": float(np.median([x["test_error"] for x in per_split])),
        "median_label_shuffle_gain": float(
            np.median([x["label_shuffle_gain"] for x in per_split])
        ),
        "fraction_splits_label_positive": float(
            np.mean([x["label_shuffle_gain"] > 0 for x in per_split])
        ),
        "median_random_topology_gain": float(
            np.median([x["random_topology_gain"] for x in per_split])
        ),
        "fraction_splits_random_positive": float(
            np.mean([x["random_topology_gain"] > 0 for x in per_split])
        ),
        "per_split": per_split,
    }
