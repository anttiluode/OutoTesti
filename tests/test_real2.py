import numpy as np

from outotesti.geometry import geometry_null_audit
from outotesti.green import fit_green_operator, green_leaf_kernel, random_topology_with_lengths, relabel_leaves
from outotesti.tree import random_binary_tree


def test_green_operator_oracle_recovery():
    rng = np.random.default_rng(10)
    tree = random_binary_tree(12, rng)
    K = green_leaf_kernel(tree, leak_ratio=0.7)
    a = rng.normal(size=12)
    b = rng.normal(size=12)
    W = a[:, None] * K * b[None, :]
    result = fit_green_operator(
        W,
        tree,
        leak_grid=np.asarray([0.7]),
        exponents=(1.0,),
    )
    assert result["error"] < 1e-8


def test_random_topology_preserves_length_multiset():
    rng = np.random.default_rng(11)
    tree = random_binary_tree(15, rng)
    lengths = np.asarray([w for _, _, w in tree.edges])
    rt = random_topology_with_lengths(15, lengths, rng)
    assert np.allclose(
        np.sort(lengths),
        np.sort([w for _, _, w in rt.edges]),
    )


def test_geometry_null_is_finite():
    rng = np.random.default_rng(12)
    W = rng.normal(size=(16, 16))
    result = geometry_null_audit(W, controls=8, quartets=200, seed=1)
    assert np.isfinite(result["p95_gap"]["tree_likeness_z"])
    assert 0 < result["p95_gap"]["empirical_p_lower"] <= 1



def test_relabel_leaves_preserves_unlabeled_lengths():
    rng = np.random.default_rng(13)
    tree = random_binary_tree(10, rng)
    perm = rng.permutation(10)
    relabeled = relabel_leaves(tree, perm)
    assert np.allclose(
        np.sort([w for _, _, w in tree.edges]),
        np.sort([w for _, _, w in relabeled.edges]),
    )
    assert relabeled.n_nodes == tree.n_nodes
