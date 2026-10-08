"""Symmetric positive definite *block*-banded solves with square blocks (JAX, functional).

A matrix with block half-bandwidth ``m`` is stored in lower block-banded form
``band[d, j] = G[j + d, j]`` (a bs x bs block), shape ``(m + 1, nj, bs, bs)``; ``band[0, j]`` is the
full symmetric diagonal block. The Cholesky factor ``L`` is computed with a right-looking
algorithm in a ``fori_loop`` over block columns (static trip count, so reverse-mode
differentiable). For a chain with joints numbered along the chain, ``m = 1`` and the solve is O(nj).
"""

from __future__ import annotations

import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

UNROLL = 1


def block_band_to_dense(band):
    """Expand block-banded storage into the full symmetric matrix of size bs nj."""
    m1, nj, bs = band.shape[:3]
    dense = jnp.zeros((nj, nj, bs, bs), band.dtype)
    cols = jnp.arange(nj)
    for d in range(m1):
        idx = cols[: nj - d]
        dense = dense.at[idx + d, idx].set(band[d, : nj - d])
        if d:
            dense = dense.at[idx, idx + d].set(jnp.swapaxes(band[d, : nj - d], -1, -2))
    return dense.transpose(0, 2, 1, 3).reshape(bs * nj, bs * nj)


def _pad(band):
    """Pad with m identity columns so that updates to columns j + d never go out of bounds."""
    m1, nj, bs = band.shape[:3]
    pad = jnp.zeros((m1, m1 - 1, bs, bs), band.dtype).at[0].set(jnp.eye(bs, dtype=band.dtype))
    return jnp.concatenate((band, pad), axis=1)


def _chol3(a):
    """Closed-form Cholesky factor of a 3x3 SPD matrix (elementwise ops, no LAPACK call)."""
    l00 = jnp.sqrt(a[0, 0])
    l10 = a[1, 0] / l00
    l20 = a[2, 0] / l00
    l11 = jnp.sqrt(a[1, 1] - l10 * l10)
    l21 = (a[2, 1] - l20 * l10) / l11
    l22 = jnp.sqrt(a[2, 2] - l20 * l20 - l21 * l21)
    z = jnp.zeros_like(l00)
    return jnp.stack((jnp.stack((l00, z, z)), jnp.stack((l10, l11, z)), jnp.stack((l20, l21, l22))))


def _tri_inv3(low):
    """Closed-form inverse of a 3x3 lower-triangular matrix."""
    i00 = 1.0 / low[0, 0]
    i11 = 1.0 / low[1, 1]
    i22 = 1.0 / low[2, 2]
    i10 = -low[1, 0] * i00 * i11
    i21 = -low[2, 1] * i11 * i22
    i20 = -(low[2, 0] * i00 + low[2, 1] * i10) * i22
    z = jnp.zeros_like(i00)
    return jnp.stack((jnp.stack((i00, z, z)), jnp.stack((i10, i11, z)), jnp.stack((i20, i21, i22))))


def _chol(a):
    return _chol3(a) if a.shape[-1] == 3 else jnp.linalg.cholesky(a)


def _tri_inv(low):
    if low.shape[-1] == 3:
        return _tri_inv3(low)
    return solve_triangular(low, jnp.eye(low.shape[-1], dtype=low.dtype), lower=True)


def cholesky_block_banded(band):
    """Block-banded Cholesky factor ``L`` (same storage as ``band``), G = L L^T."""
    m1, nj = band.shape[:2]
    m = m1 - 1
    pairs = [(d1, d2) for d1 in range(1, m + 1) for d2 in range(1, d1 + 1)]
    d1_idx = np.array([p[0] for p in pairs], dtype=np.intp)
    d2_idx = np.array([p[1] for p in pairs], dtype=np.intp)

    def column(j, a):
        ljj = _chol(a[0, j])
        a = a.at[0, j].set(ljj)
        if m:
            below = a[1:, j] @ _tri_inv(ljj).T
            a = a.at[1:, j].set(below)
            outer = below[d1_idx - 1] @ jnp.swapaxes(below[d2_idx - 1], -1, -2)
            a = a.at[d1_idx - d2_idx, j + d2_idx].add(-outer)
        return a

    return lax.fori_loop(0, nj, column, _pad(band), unroll=UNROLL)[:, :nj]


def solve_cholesky_block_banded(chol, rhs):
    """Solve ``L L^T x = rhs`` given the factor from ``cholesky_block_banded``; rhs is (bs nj,)."""
    m1, nj, bs = chol.shape[:3]
    m = m1 - 1
    pad = jnp.zeros((m1, m1 - 1, bs, bs), chol.dtype)
    lower = jnp.concatenate((chol, pad), axis=1)
    b = jnp.concatenate((rhs.reshape(nj, bs), jnp.zeros((m, bs), rhs.dtype)))

    def forward(j, b):
        yj = _tri_inv(lower[0, j]) @ b[j]
        b = b.at[j].set(yj)
        if m:
            update = -jnp.einsum("dab,b->da", lower[1:, j], yj)
            tail = lax.dynamic_slice_in_dim(b, j + 1, m, axis=0)
            b = lax.dynamic_update_slice_in_dim(b, tail + update, j + 1, axis=0)
        return b

    y = lax.fori_loop(0, nj, forward, b, unroll=UNROLL)

    def backward(i, x):
        j = nj - 1 - i
        tail = lax.dynamic_slice_in_dim(x, j + 1, m, axis=0) if m else x[:0]
        acc = y[j] - jnp.einsum("dab,da->b", lower[1:, j], tail) if m else y[j]
        return x.at[j].set(_tri_inv(lower[0, j]).T @ acc)

    x = lax.fori_loop(0, nj, backward, jnp.zeros_like(b), unroll=UNROLL)
    return x[:nj].reshape(-1)


def solve_spd_block_banded(band, rhs):
    """Solve ``G x = rhs`` for symmetric positive definite G in lower block-banded storage."""
    return solve_cholesky_block_banded(cholesky_block_banded(band), rhs)
