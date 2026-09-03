import numpy as np

from outotesti.gauge import joint_qk_head_gauge
from outotesti.subspace import (
    head_subspace_distance_matrix,
    row_subspace_basis,
    chordal_subspace_distance,
)


def test_row_subspace_is_invariant_to_left_orthogonal_basis_change():
    rng = np.random.default_rng(201)
    W = rng.normal(size=(4, 64))
    R, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    U = row_subspace_basis(W, rank=4)
    V = row_subspace_basis(R.T @ W, rank=4)
    assert chordal_subspace_distance(U, V) < 1e-7


def test_head_subspace_geometry_survives_qk_gauge():
    rng = np.random.default_rng(202)
    Wq = rng.normal(size=(64, 64))
    Wk = rng.normal(size=(64, 64))
    Dq = head_subspace_distance_matrix(Wq, num_heads=16)
    Dk = head_subspace_distance_matrix(Wk, num_heads=16)

    q2, k2, _ = joint_qk_head_gauge(
        Wq, Wk, num_heads=16, rng=np.random.default_rng(203)
    )
    Dq2 = head_subspace_distance_matrix(q2, num_heads=16)
    Dk2 = head_subspace_distance_matrix(k2, num_heads=16)

    assert np.allclose(Dq, Dq2, atol=1e-10, rtol=1e-10)
    assert np.allclose(Dk, Dk2, atol=1e-10, rtol=1e-10)
