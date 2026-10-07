"""Backend-agnostic joint constraints: residual g(q) and angular Jacobian blocks per joint kind.

Every joint kind contributes ``rows`` scalar equations g = 0 between body a (possibly the world)
and body b. With u = [v, w] (world-frame linear, body-frame angular velocity), g_dot = J u where

    J_a = [ -E, ang_a ],    J_b = [ +E, ang_b ],      E = [I; 0]  (the first 3 rows are positional)

so only the angular blocks ``ang_a``, ``ang_b`` (rows x 3) depend on the configuration. All
functions take the array namespace ``xp`` (``numpy`` or ``jax.numpy``) and are free of in-place
operations. The world is handled by appending an identity frame at index -1.

Ball joint (3 rows):  g = (x_b + R_b r_b) - (x_a + R_a r_a)
    ang_b = -R_b [r_b]_x,    ang_a = +R_a [r_a]_x
Hinge joint (5 rows): the same point rows plus two orientation rows keeping the hinge axes
parallel, g_i = (R_a p_i) . (R_b n_b) for p_1, p_2 perpendicular to the axis n_a in frame a:
    ang_a,i = p_i x (R_a^T R_b n_b),    ang_b,i = n_b x (R_b^T R_a p_i)
Joints with fewer rows than the system maximum are padded with zero rows (see ``joint_topology``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dynamix.core import rotations

BALL, HINGE = 0, 1
ROWS = {BALL: 3, HINGE: 5}
POSITION_ROWS = 3


class BallData(NamedTuple):
    joint_a: np.ndarray  # (m,) body index, -1 for the world
    joint_b: np.ndarray  # (m,)
    anchor_a: np.ndarray  # (m, 3)
    anchor_b: np.ndarray  # (m, 3)
    jw_a: np.ndarray  # (m, 3, 3) = +[r_a]_x
    jw_b: np.ndarray  # (m, 3, 3) = -[r_b]_x


class HingeData(NamedTuple):
    joint_a: np.ndarray
    joint_b: np.ndarray
    anchor_a: np.ndarray
    anchor_b: np.ndarray
    jw_a: np.ndarray
    jw_b: np.ndarray
    axis_b: np.ndarray  # (m, 3) unit hinge axis in frame b
    perp_a: np.ndarray  # (m, 2, 3) unit vectors perpendicular to the hinge axis in frame a


class JointSet(NamedTuple):
    """Per-kind constant joint data; ``order`` restores joint order after grouping by kind."""

    ball: BallData | None
    hinge: HingeData | None
    order: np.ndarray | None

    @property
    def n_joints(self) -> int:
        return sum(d.joint_b.shape[0] for d in (self.ball, self.hinge) if d is not None)

    @property
    def rows(self) -> int:
        return ROWS[HINGE] if self.hinge is not None else ROWS[BALL]


def perpendicular_pair(axis: np.ndarray) -> np.ndarray:
    """Two unit vectors perpendicular to each unit axis (m, 3) -> (m, 2, 3)."""
    helper = np.zeros_like(axis)
    helper[np.arange(axis.shape[0]), np.argmin(np.abs(axis), axis=1)] = 1.0
    p1 = np.cross(axis, helper)
    p1 /= np.linalg.norm(p1, axis=1, keepdims=True)
    return np.stack((p1, np.cross(axis, p1)), axis=1)


def build_joint_set(kind, joint_a, joint_b, anchor_a, anchor_b, axis_a, axis_b) -> JointSet:
    """Group joints by kind on the host (NumPy arrays)."""
    skew_a, skew_b = rotations.skew(np, anchor_a), rotations.skew(np, anchor_b)
    groups, indices = {}, {}
    for k in (BALL, HINGE):
        idx = np.flatnonzero(kind == k)
        if idx.size == 0:
            continue
        indices[k] = idx
        common = (
            joint_a[idx],
            joint_b[idx],
            anchor_a[idx],
            anchor_b[idx],
            skew_a[idx],
            -skew_b[idx],
        )
        if k == BALL:
            groups[k] = BallData(*common)
        else:
            unit_a = axis_a[idx] / np.linalg.norm(axis_a[idx], axis=1, keepdims=True)
            unit_b = axis_b[idx] / np.linalg.norm(axis_b[idx], axis=1, keepdims=True)
            groups[k] = HingeData(*common, unit_b, perpendicular_pair(unit_a))
    grouped = np.concatenate(list(indices.values()))
    order = None if np.array_equal(grouped, np.arange(grouped.size)) else np.argsort(grouped)
    return JointSet(groups.get(BALL), groups.get(HINGE), order)


def _pad_rows(xp, a, rows):
    k = a.shape[1]
    if k == rows:
        return a
    return xp.concatenate((a, xp.zeros((a.shape[0], rows - k) + a.shape[2:], a.dtype)), axis=1)


def _points(xp, d, rot, pos):
    """Rotations of both bodies and the point residual rows."""
    ra, rb = rot[d.joint_a], rot[d.joint_b]
    lever_a = (ra @ d.anchor_a[:, :, None])[:, :, 0]
    lever_b = (rb @ d.anchor_b[:, :, None])[:, :, 0]
    return ra, rb, (pos[d.joint_b] + lever_b) - (pos[d.joint_a] + lever_a)


def _ball(xp, d, rot, pos, rows):
    ra, rb, g = _points(xp, d, rot, pos)
    return (
        _pad_rows(xp, g, rows),
        _pad_rows(xp, ra @ d.jw_a, rows),
        _pad_rows(xp, rb @ d.jw_b, rows),
    )


def _hinge(xp, d, rot, pos, rows):
    ra, rb, g_pos = _points(xp, d, rot, pos)
    c = (rb @ d.axis_b[:, :, None])[:, :, 0]  # hinge axis of b in the world frame
    p_world = xp.einsum("mab,mib->mia", ra, d.perp_a)  # R_a p_i
    g = xp.concatenate((g_pos, xp.einsum("mia,ma->mi", p_world, c)), axis=1)
    ang_a = xp.cross(d.perp_a, xp.einsum("mba,mb->ma", ra, c)[:, None, :])
    ang_b = xp.cross(d.axis_b[:, None, :], xp.einsum("mba,mib->mia", rb, p_world))
    ang_a = xp.concatenate((ra @ d.jw_a, ang_a), axis=1)
    ang_b = xp.concatenate((rb @ d.jw_b, ang_b), axis=1)
    return _pad_rows(xp, g, rows), _pad_rows(xp, ang_a, rows), _pad_rows(xp, ang_b, rows)


def evaluate(xp, joints: JointSet, q, rot):
    """Return ``g`` (nj, rows) and ``ang`` (2 nj, rows, 3): ang_a blocks, then ang_b blocks."""
    rows = joints.rows
    rot_ext = xp.concatenate((rot, xp.eye(3, dtype=rot.dtype)[None]))
    pos_ext = xp.concatenate((q[:, :3], xp.zeros((1, 3), q.dtype)))
    parts = []
    if joints.ball is not None:
        parts.append(_ball(xp, joints.ball, rot_ext, pos_ext, rows))
    if joints.hinge is not None:
        parts.append(_hinge(xp, joints.hinge, rot_ext, pos_ext, rows))
    if len(parts) == 1:
        g, ang_a, ang_b = parts[0]
    else:
        g, ang_a, ang_b = (xp.concatenate(x) for x in zip(*parts, strict=True))
    if joints.order is not None:
        g, ang_a, ang_b = g[joints.order], ang_a[joints.order], ang_b[joints.order]
    return g, xp.concatenate((ang_a, ang_b))
