import numpy as np

from outotesti.projection import fit_tree_kernel, matched_budget_svd
from outotesti.tree import leaf_distance_matrix, random_binary_tree


def test_oracle_tree_kernel_recovers_generated_matrix():
    rng = np.random.default_rng(3)
    tree = random_binary_tree(14, rng)
    D = leaf_distance_matrix(tree)
    K = np.exp(-1.5 * D / np.median(D[D > 0]))
    a = rng.normal(size=14)
    b = rng.normal(size=14)
    W = a[:, None] * K * b[None, :]
    result = fit_tree_kernel(W, tree)
    assert result["error"] < 1e-6


def test_matched_budget_svd_is_finite():
    rng = np.random.default_rng(4)
    W = rng.normal(size=(16, 16))
    out = matched_budget_svd(W, budget=60)
    assert 1 <= out["rank"] <= 16
    assert np.isfinite(out["error"])
