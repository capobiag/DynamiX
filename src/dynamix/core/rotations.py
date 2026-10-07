"""Backend-agnostic rotation math, [x, y, z, w] quaternions, batched over leading axes.

Every function takes the array namespace ``xp`` (``numpy`` or ``jax.numpy``) as first argument and
is free of in-place operations, so the same code runs eagerly under NumPy and traced under JAX.
This module imports neither JAX nor Warp. The backends' ``quaternion`` modules bind ``xp``.
"""

from __future__ import annotations


def multiply(xp, a, b):
    """Hamilton product a * b."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return xp.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def from_rotvec(xp, rotvec):
    """Quaternion of the rotation vector (axis * angle); finite derivatives at zero rotation."""
    angle2 = xp.einsum("...i,...i->...", rotvec, rotvec)[..., None]
    small = angle2 < 1e-16
    # double-where keeps sqrt and the division out of the unused branch (safe gradients)
    safe2 = xp.where(small, 1.0, angle2)
    angle = xp.sqrt(safe2)
    scale = xp.where(small, 0.5 - angle2 / 48.0, xp.sin(0.5 * angle) / angle)
    w = xp.where(small, 1.0 - angle2 / 8.0, xp.cos(0.5 * angle))
    return xp.concatenate((rotvec * scale, w), axis=-1)


def to_matrix(xp, q):
    """Rotation matrices (..., 3, 3) of unit quaternions (..., 4)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    entries = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    flat = xp.stack([e for row in entries for e in row], axis=-1)
    return flat.reshape(q.shape[:-1] + (3, 3))


def integrate(xp, q, omega, h):
    """Advance q by the body-frame angular velocity omega over time h (exact for constant omega)."""
    out = multiply(xp, q, from_rotvec(xp, omega * h))
    return out / xp.linalg.norm(out, axis=-1, keepdims=True)


def skew(xp, v):
    """Cross-product matrices (..., 3, 3) such that skew(v) @ x = v x x."""
    zero = xp.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return xp.stack(
        (
            xp.stack((zero, -z, y), axis=-1),
            xp.stack((z, zero, -x), axis=-1),
            xp.stack((-y, x, zero), axis=-1),
        ),
        axis=-2,
    )
