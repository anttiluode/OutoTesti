import numpy as np

from outotesti.geometry import (
    quartet_heldout_stability,
    quartet_split_ids,
    sampled_quartets,
    spectrum_matched_quartet_stability_audit,
)
from outotesti.tree import leaf_distance_matrix, random_binary_tree


def test_quartet_split_ids_on_exact_tree_are_self_consistent():
    rng = np.random.default_rng(31)
    tree = random_binary_tree(12, rng)
    D = leaf_distance_matrix(tree)
    qs = sampled_quartets(12, 100, seed=1)
    a = quartet_split_ids(D, qs)
    b = quartet_split_ids(2.7 * D, qs)
    assert np.array_equal(a, b)


def test_quartet_stability_audit_is_finite():
    rng = np.random.default_rng(32)
    W = rng.normal(size=(16, 16))
    out = quartet_heldout_stability(W, splits=2, quartets=100, seed=2)
    assert 0 <= out["median_agreement"] <= 1

    audit = spectrum_matched_quartet_stability_audit(
        W, controls=4, splits=2, quartets=100, seed=3
    )
    assert np.isfinite(audit["stability_z"])
    assert 0 < audit["empirical_p_upper"] <= 1
