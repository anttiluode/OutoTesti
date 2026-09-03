from __future__ import annotations

import numpy as np


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic-sign Haar-like orthogonal matrix from a Gaussian QR draw."""
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs[None, :]


def joint_qk_head_gauge(
    Wq: np.ndarray,
    Wk: np.ndarray,
    *,
    num_heads: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Apply independent orthogonal gauges to Q/K output coordinates per head.

    PyTorch Linear uses q = x Wq.T and k = x Wk.T. For each head h,

        Wq_h' = R_h.T @ Wq_h
        Wk_h' = R_h.T @ Wk_h

    with orthogonal R_h. Then

        Wq_h'.T @ Wk_h' = Wq_h.T @ Wk_h

    exactly in real arithmetic, so all pre-softmax attention scores are unchanged.
    """
    Wq = np.asarray(Wq, dtype=float)
    Wk = np.asarray(Wk, dtype=float)
    if Wq.shape != Wk.shape or Wq.ndim != 2:
        raise ValueError("Wq and Wk must have the same 2-D shape")
    if Wq.shape[0] % num_heads:
        raise ValueError("output dimension must be divisible by num_heads")

    head_dim = Wq.shape[0] // num_heads
    q2 = Wq.copy()
    k2 = Wk.copy()
    rotations = []

    for h in range(num_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        R = random_orthogonal(head_dim, rng)
        q2[sl, :] = R.T @ Wq[sl, :]
        k2[sl, :] = R.T @ Wk[sl, :]
        rotations.append(R)

    return q2, k2, rotations


def max_head_score_relative_error(
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wq2: np.ndarray,
    Wk2: np.ndarray,
    *,
    num_heads: int,
) -> float:
    Wq = np.asarray(Wq, dtype=float)
    Wk = np.asarray(Wk, dtype=float)
    Wq2 = np.asarray(Wq2, dtype=float)
    Wk2 = np.asarray(Wk2, dtype=float)
    head_dim = Wq.shape[0] // num_heads

    errors = []
    for h in range(num_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        M = Wq[sl, :].T @ Wk[sl, :]
        M2 = Wq2[sl, :].T @ Wk2[sl, :]
        errors.append(
            np.linalg.norm(M2 - M) / max(np.linalg.norm(M), 1e-15)
        )
    return float(max(errors, default=0.0))
