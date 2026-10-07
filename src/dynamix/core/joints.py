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
Fixed joint (6 rows): the point rows plus three orientation rows keeping the relative rotation
at its initial value C (R_b = R_a C). With M = C^T R_a^T R_b (identity at rest),
    g_rot = vee(skew(M)),  ang_b = (tr(M) I - M^T) / 2,  ang_a = -(tr(M) I - M) C^T / 2
Prismatic joint (5 rows): the three orientation rows above plus two translation rows keeping the
displacement d = (x_b + R_b r_b) - (x_a + R_a r_a) along the axis n_a in frame a, with p_1, p_2
perpendicular to n_a: g_i = (R_a p_i) . d. Their linear Jacobian is no longer E but
    lin_b,i = R_a p_i,  lin_a = -lin_b
    ang_a,i = p_i x (r_a + R_a^T d),  ang_b,i = r_b x (R_b^T R_a p_i)
so a system containing a prismatic joint also carries the linear blocks ``lin`` (nj, rows, 3);
for the other joint kinds ``lin`` is E padded to ``rows``.
Joints with fewer rows than the system maximum are padded with zero rows (see ``joint_topology``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dynamix.core import rotations

BALL, HINGE, FIXED, PRISMATIC = 0, 1, 2, 3
ROWS = {BALL: 3, HINGE: 5, FIXED: 6, PRISMATIC: 5}
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


class FixedData(NamedTuple):
    joint_a: np.ndarray
    joint_b: np.ndarray
    anchor_a: np.ndarray
    anchor_b: np.ndarray
    jw_a: np.ndarray
    jw_b: np.ndarray
    rel_rot: np.ndarray  # (m, 3, 3) C with R_b = R_a C at rest


class PrismaticData(NamedTuple):
    joint_a: np.ndarray
    joint_b: np.ndarray
    anchor_a: np.ndarray
    anchor_b: np.ndarray
    rel_rot: np.ndarray
    perp_a: np.ndarray  # (m, 2, 3) unit vectors perpendicular to the slide axis in frame a


class JointSet(NamedTuple):
    """Per-kind constant joint data; ``order`` restores joint order after grouping by kind."""

    ball: BallData | None
    hinge: HingeData | None
    fixed: FixedData | None
    prismatic: PrismaticData | None
    order: np.ndarray | None

    @property
    def _present(self):
        return (
            (BALL, self.ball),
            (HINGE, self.hinge),
            (FIXED, self.fixed),
            (PRISMATIC, self.prismatic),
        )

    @property
    def n_joints(self) -> int:
        return sum(d.joint_b.shape[0] for _, d in self._present if d is not None)

    @property
    def rows(self) -> int:
        return max(ROWS[k] for k, d in self._present if d is not None)

    @property
    def general_lin(self) -> bool:
        """True if some joint has a configuration dependent linear Jacobian (not just E)."""
        return self.prismatic is not None


def perpendicular_pair(axis: np.ndarray) -> np.ndarray:
    """Two unit vectors perpendicular to each unit axis (m, 3) -> (m, 2, 3)."""
    helper = np.zeros_like(axis)
    helper[np.arange(axis.shape[0]), np.argmin(np.abs(axis), axis=1)] = 1.0
    p1 = np.cross(axis, helper)
    p1 /= np.linalg.norm(p1, axis=1, keepdims=True)
    return np.stack((p1, np.cross(axis, p1)), axis=1)


def build_joint_set(
    kind, joint_a, joint_b, anchor_a, anchor_b, axis_a, axis_b, rel_rot=None
) -> JointSet:
    """Group joints by kind on the host (NumPy arrays)."""
    if rel_rot is None:
        rel_rot = np.tile(np.eye(3), (kind.shape[0], 1, 1))
    skew_a, skew_b = rotations.skew(np, anchor_a), rotations.skew(np, anchor_b)
    groups, indices = {}, []
    for k in (BALL, HINGE, FIXED, PRISMATIC):
        idx = np.flatnonzero(kind == k)
        if idx.size == 0:
            continue
        indices.append(idx)
        ends = (joint_a[idx], joint_b[idx], anchor_a[idx], anchor_b[idx])
        jw = (skew_a[idx], -skew_b[idx])
        if k == BALL:
            groups[k] = BallData(*ends, *jw)
        elif k == HINGE:
            unit_a = axis_a[idx] / np.linalg.norm(axis_a[idx], axis=1, keepdims=True)
            unit_b = axis_b[idx] / np.linalg.norm(axis_b[idx], axis=1, keepdims=True)
            groups[k] = HingeData(*ends, *jw, unit_b, perpendicular_pair(unit_a))
        elif k == FIXED:
            groups[k] = FixedData(*ends, *jw, rel_rot[idx])
        else:
            unit_a = axis_a[idx] / np.linalg.norm(axis_a[idx], axis=1, keepdims=True)
            groups[k] = PrismaticData(*ends, rel_rot[idx], perpendicular_pair(unit_a))
    grouped = np.concatenate(indices)
    order = None if np.array_equal(grouped, np.arange(grouped.size)) else np.argsort(grouped)
    return JointSet(
        groups.get(BALL), groups.get(HINGE), groups.get(FIXED), groups.get(PRISMATIC), order
    )


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


def _orientation(xp, ra, rb, rel_rot):
    """Rotation rows of the fixed and prismatic joints: g (m, 3), ang_a, ang_b (m, 3, 3)."""
    c_t = xp.swapaxes(rel_rot, 1, 2)
    mat = c_t @ (xp.swapaxes(ra, 1, 2) @ rb)
    trace = (mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2])[:, None, None]
    eye = xp.eye(3, dtype=mat.dtype)
    g = 0.5 * xp.stack(
        (mat[:, 2, 1] - mat[:, 1, 2], mat[:, 0, 2] - mat[:, 2, 0], mat[:, 1, 0] - mat[:, 0, 1]),
        axis=1,
    )
    ang_b = 0.5 * (trace * eye - xp.swapaxes(mat, 1, 2))
    ang_a = -0.5 * (trace * eye - mat) @ c_t
    return g, ang_a, ang_b


def _fixed(xp, d, rot, pos, rows):
    ra, rb, g_pos = _points(xp, d, rot, pos)
    g_rot, rot_a, rot_b = _orientation(xp, ra, rb, d.rel_rot)
    g = xp.concatenate((g_pos, g_rot), axis=1)
    ang_a = xp.concatenate((ra @ d.jw_a, rot_a), axis=1)
    ang_b = xp.concatenate((rb @ d.jw_b, rot_b), axis=1)
    return _pad_rows(xp, g, rows), _pad_rows(xp, ang_a, rows), _pad_rows(xp, ang_b, rows)


def _prismatic(xp, d, rot, pos, rows):
    ra, rb, disp = _points(xp, d, rot, pos)
    g_rot, rot_a, rot_b = _orientation(xp, ra, rb, d.rel_rot)
    p_world = xp.einsum("mab,mib->mia", ra, d.perp_a)  # R_a p_i
    g = xp.concatenate((g_rot, xp.einsum("mia,ma->mi", p_world, disp)), axis=1)
    local = d.anchor_a + xp.einsum("mba,mb->ma", ra, disp)  # r_a + R_a^T d
    ang_a = xp.concatenate((rot_a, xp.cross(d.perp_a, local[:, None, :])), axis=1)
    rb_p = xp.einsum("mba,mib->mia", rb, p_world)
    ang_b = xp.concatenate((rot_b, xp.cross(d.anchor_b[:, None, :], rb_p)), axis=1)
    lin = xp.concatenate((xp.zeros_like(rot_a), p_world), axis=1)
    return (
        _pad_rows(xp, g, rows),
        _pad_rows(xp, ang_a, rows),
        _pad_rows(xp, ang_b, rows),
        (_pad_rows(xp, lin, rows)),
    )


def evaluate(xp, joints: JointSet, q, rot):
    """Return ``(g, ang, lin)``.

    ``g`` is (nj, rows); ``ang`` is (2 nj, rows, 3) with the ang_a blocks, then the ang_b blocks;
    ``lin`` is the (nj, rows, 3) linear block of the b side (the a side is -lin), or None when it
    is E for every joint (see ``JointSet.general_lin``).
    """
    rows = joints.rows
    rot_ext = xp.concatenate((rot, xp.eye(3, dtype=rot.dtype)[None]))
    pos_ext = xp.concatenate((q[:, :3], xp.zeros((1, 3), q.dtype)))
    general = joints.general_lin
    parts = []
    for kind, data in joints._present:
        if data is None:
            continue
        func = {BALL: _ball, HINGE: _hinge, FIXED: _fixed, PRISMATIC: _prismatic}[kind]
        out = func(xp, data, rot_ext, pos_ext, rows)
        if general and len(out) == 3:
            lin = xp.zeros((data.joint_b.shape[0], rows, 3), q.dtype)
            lin = lin + xp.eye(rows, 3, dtype=q.dtype)
            out = (*out, lin)
        parts.append(out if general else out[:3])
    if len(parts) == 1:
        merged = parts[0]
    else:
        merged = tuple(xp.concatenate(x) for x in zip(*parts, strict=True))
    if joints.order is not None:
        merged = tuple(x[joints.order] for x in merged)
    g, ang_a, ang_b = merged[:3]
    return g, xp.concatenate((ang_a, ang_b)), (merged[3] if general else None)
