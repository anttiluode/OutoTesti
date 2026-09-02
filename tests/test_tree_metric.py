import numpy as np

from outotesti.metrics import four_point_score
from outotesti.tree import leaf_distance_matrix, neighbor_joining, random_binary_tree


def test_exact_tree_metric_passes_four_point():
    rng = np.random.default_rng(1)
    tree = random_binary_tree(12, rng)
    D = leaf_distance_matrix(tree)
    score = four_point_score(D)
    assert score["max_gap"] < 1e-10


def test_neighbor_joining_reconstructs_additive_distances():
    rng = np.random.default_rng(2)
    tree = random_binary_tree(10, rng)
    D = leaf_distance_matrix(tree)
    inferred = neighbor_joining(D)
    D2 = leaf_distance_matrix(inferred)
    assert np.allclose(D, D2, atol=1e-8, rtol=1e-8)
