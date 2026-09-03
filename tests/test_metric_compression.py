import numpy as np

from outotesti.metric_compression import (
    classical_mds_metric,
    scaled_metric_error,
    star_metric_from_distances,
)
from outotesti.tree import leaf_distance_matrix, random_binary_tree


def test_scaled_metric_error_is_scale_invariant():
    rng = np.random.default_rng(301)
    tree = random_binary_tree(10, rng)
    D = leaf_distance_matrix(tree)
    assert scaled_metric_error(3.2 * D, D) < 1e-12


def test_star_metric_recovers_true_star():
    rng = np.random.default_rng(302)
    r = rng.uniform(0.2, 1.2, size=12)
    D = r[:, None] + r[None, :]
    np.fill_diagonal(D, 0.0)
    D2 = star_metric_from_distances(D)
    assert np.allclose(D, D2, atol=1e-10, rtol=1e-10)


def test_mds2_recovers_planar_euclidean_distances():
    rng = np.random.default_rng(303)
    X = rng.normal(size=(12, 2))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    D2 = classical_mds_metric(D, dim=2)
    assert np.allclose(D, D2, atol=1e-8, rtol=1e-8)
