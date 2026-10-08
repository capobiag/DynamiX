"""Ball joint constraint g(q) = 0 and its velocity-level Jacobian.

Joint a-b: g = (x_b + R_b r_b) - (x_a + R_a r_a), with body a optionally the world (index -1),
in which case ``anchor_a`` is a fixed world point. The Jacobian maps the generalized velocity
u = [v, w] (world-frame linear, body-frame angular velocity) to g_dot: g_dot = J u.
With d(R r)/dt = R (w x r) = -R [r]_x w:
    dg/dv_b = I,   dg/dw_b = -R_b [r_b]_x   (opposite sign for body a).
"""

from __future__ import annotations

import numpy as np

from dynamix.numpy_backend import quaternion as quat


def _anchor(q, rot, idx, anchor):
    """World anchor point and lever arm R r for body indices (-1 = world)."""
    is_world = idx < 0
    safe = np.where(is_world, 0, idx)
    arm = np.einsum("jab,jb->ja", rot[safe], anchor)
    arm[is_world] = 0.0
    point = np.where(is_world[:, None], anchor, q[safe, :3] + arm)
    return point, arm, is_world


def ball_joint_residual(q, joint_a, joint_b, anchor_a, anchor_b, rot=None) -> np.ndarray:
    """g(q), shape (nj, 3)."""
    if rot is None:
        rot = quat.to_matrix(q[:, 3:])
    pa, _, _ = _anchor(q, rot, joint_a, anchor_a)
    pb, _, _ = _anchor(q, rot, joint_b, anchor_b)
    return pb - pa


def ball_joint_jacobian(q, joint_a, joint_b, anchor_a, anchor_b, out=None, rot=None) -> np.ndarray:
    """Dense Jacobian of shape (3 nj, 6 n). ``out`` is overwritten if given."""
    n, nj = q.shape[0], joint_a.shape[0]
    if out is None:
        out = np.zeros((3 * nj, 6 * n))
    else:
        out.fill(0.0)
    if rot is None:
        rot = quat.to_matrix(q[:, 3:])
    a_is_world = joint_a < 0
    rot_a = rot[np.where(a_is_world, 0, joint_a)]
    rot_b = rot[joint_b]

    rows = 3 * np.arange(nj)[:, None] + np.arange(3)[None, :]  # (nj, 3)
    eye = np.broadcast_to(np.eye(3), (nj, 3, 3))

    block_b = np.concatenate((eye, -rot_b @ quat.skew(anchor_b)), axis=2)  # (nj, 3, 6)
    cols_b = 6 * joint_b[:, None] + np.arange(6)[None, :]
    out[rows[:, :, None], cols_b[:, None, :]] = block_b

    block_a = -np.concatenate((eye, -rot_a @ quat.skew(anchor_a)), axis=2)
    cols_a = 6 * np.where(a_is_world, 0, joint_a)[:, None] + np.arange(6)[None, :]
    keep = ~a_is_world
    out[rows[keep][:, :, None], cols_a[keep][:, None, :]] = block_a[keep]
    return out
