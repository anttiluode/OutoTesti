import numpy as np

from outotesti.behavior import (
    centered_cosine_distance,
    label_permutation_correlation_test,
    upper_triangle_correlation,
)


def test_centered_cosine_distance_identical_rows_zero():
    X = np.tile(np.arange(10, dtype=float), (4, 1))
    D = centered_cosine_distance(X)
    assert np.max(np.abs(D)) < 1e-12


def test_permutation_test_detects_same_labeled_metric():
    rng = np.random.default_rng(401)
    X = rng.normal(size=(12, 5))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    out = label_permutation_correlation_test(D, D, controls=64, seed=2)
    assert out["observed"] > 0.99
    assert out["empirical_p_upper"] < 0.05


def test_upper_triangle_correlation_is_scale_invariant():
    rng = np.random.default_rng(402)
    X = rng.normal(size=(10, 3))
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    assert upper_triangle_correlation(D, 7.0 * D) > 0.999999
