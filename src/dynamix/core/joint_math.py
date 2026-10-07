"""Backend-agnostic helpers shared by the joint systems (free of in-place operations)."""

from __future__ import annotations

from dynamix.core.joints import POSITION_ROWS


def side_rate(xp, jw, sign, v_side, w_side, lin=None):
    """Per-side contribution J_side u to the joint velocity, shape (n, rows).

    ``lin`` is the (n, rows, 3) linear block of the b side (None: E, the first 3 rows).
    """
    rows = jw.shape[1]
    if lin is not None:
        part = sign[:, None] * (lin @ v_side[:, :, None])[:, :, 0]
    else:
        part = sign[:, None] * v_side
        if rows > POSITION_ROWS:
            pad = xp.zeros((part.shape[0], rows - POSITION_ROWS), part.dtype)
            part = xp.concatenate((part, pad), 1)
    return part + (jw @ w_side[:, :, None])[:, :, 0]


def gyroscopic_acceleration(xp, inertia_body, inertia_inv, w):
    """I^-1 (w x I w), the explicit gyroscopic term of the Euler equations."""
    gyro = xp.cross(w, xp.einsum("nij,nj->ni", inertia_body, w))
    return xp.einsum("nij,nj->ni", inertia_inv, gyro)
