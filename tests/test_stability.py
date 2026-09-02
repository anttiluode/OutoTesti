import numpy as np

from outotesti.stability import scaled_tree_metric_error, split_half_tree_stability
from outotesti.tree import leaf_distance_matrix, random_binary_tree


def test_scaled_tree_metric_error_exact_up_to_scale():
    rng = np.random.default_rng(21)
    tree = random_binary_tree(12, rng)
    D = 3.7 * leaf_distance_matrix(tree)
    assert scaled_tree_metric_error(D, tree) < 1e-12


def test_split_half_stability_finite():
    rng = np.random.default_rng(22)
    W = rng.normal(size=(16, 20))
    out = split_half_tree_stability(W, splits=2, controls=2, seed=3)
    assert np.isfinite(out["median_test_error"])
    assert np.isfinite(out["median_label_shuffle_gain"])
