import numpy as np

from outotesti.gauge import (
    joint_qk_head_gauge,
    max_head_score_relative_error,
)


def test_joint_qk_gauge_preserves_head_score_operators():
    rng = np.random.default_rng(101)
    Wq = rng.normal(size=(64, 64))
    Wk = rng.normal(size=(64, 64))
    q2, k2, _ = joint_qk_head_gauge(
        Wq, Wk, num_heads=16, rng=np.random.default_rng(102)
    )
    err = max_head_score_relative_error(
        Wq, Wk, q2, k2, num_heads=16
    )
    assert err < 1e-12


def test_joint_qk_gauge_preserves_singular_values():
    rng = np.random.default_rng(103)
    Wq = rng.normal(size=(64, 64))
    Wk = rng.normal(size=(64, 64))
    q2, k2, _ = joint_qk_head_gauge(
        Wq, Wk, num_heads=16, rng=np.random.default_rng(104)
    )
    assert np.allclose(
        np.linalg.svd(Wq, compute_uv=False),
        np.linalg.svd(q2, compute_uv=False),
        atol=1e-10,
        rtol=1e-10,
    )
    assert np.allclose(
        np.linalg.svd(Wk, compute_uv=False),
        np.linalg.svd(k2, compute_uv=False),
        atol=1e-10,
        rtol=1e-10,
    )
