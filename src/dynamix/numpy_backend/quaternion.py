"""Unit quaternion utilities, [x, y, z, w] order. All functions are batched over leading axes."""

from __future__ import annotations

import numpy as np


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Quaternion of the rotation vector (axis * angle)."""
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    small = angle < 1e-8
    # sin(half)/angle with a series fallback near zero
    scale = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / np.where(small, 1.0, angle))
    return np.concatenate((rotvec * scale, np.cos(half)), axis=-1)


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) of unit quaternions (..., 4)."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def integrate(q: np.ndarray, omega: np.ndarray, h: float) -> np.ndarray:
    """Advance q by the body-frame angular velocity omega over time h (exact for constant omega)."""
    out = multiply(q, from_rotvec(omega * h))
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrices (..., 3, 3) such that skew(v) @ x = v x x."""
    s = np.zeros(v.shape[:-1] + (3, 3))
    s[..., 0, 1] = -v[..., 2]
    s[..., 0, 2] = v[..., 1]
    s[..., 1, 0] = v[..., 2]
    s[..., 1, 2] = -v[..., 0]
    s[..., 2, 0] = -v[..., 1]
    s[..., 2, 1] = v[..., 0]
    return s
