"""Backend-agnostic helpers shared by the joint systems (free of in-place operations)."""

from __future__ import annotations

from dynamix.core.joints import POSITION_ROWS


def side_rate(xp, jw, sign, v_side, w_side):
    """Per-side contribution J_side u to the joint velocity: [s v; 0] + jw w, shape (n, rows)."""
    rows = jw.shape[1]
    lin = sign[:, None] * v_side
    if rows > POSITION_ROWS:
        lin = xp.concatenate((lin, xp.zeros((lin.shape[0], rows - POSITION_ROWS), lin.dtype)), 1)
    return lin + (jw @ w_side[:, :, None])[:, :, 0]


def gyroscopic_acceleration(xp, inertia_body, inertia_inv, w):
    """I^-1 (w x I w), the explicit gyroscopic term of the Euler equations."""
    gyro = xp.cross(w, xp.einsum("nij,nj->ni", inertia_body, w))
    return xp.einsum("nij,nj->ni", inertia_inv, gyro)
