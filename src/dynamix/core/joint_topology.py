"""Host-side topology of a joint system as NumPy index arrays (shared by all backends).

Each joint has one *side* per attached body (the world has none). ``side_*`` arrays describe the
sides, ``pair_*`` the ordered side pairs (s, t) on the same body with joint(s) >= joint(t), which
are the non-zero blocks of the lower triangle of G = J M^-1 J^T. Every joint occupies ``rows``
consecutive equations, where ``rows`` is the largest row count of any joint kind present; joints
with fewer equations are padded with zero Jacobian rows and a unit diagonal in G (``pad_diag``),
which keeps G positive definite and gives zero multipliers for the padding. Joints are renumbered
(reverse Cuthill-McKee on the joints sharing a body) when that narrows the band of G; ``joint_perm``
maps the internal joint index to the input joint index. Backends convert these arrays to their own
array types once, at setup.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dynamix.core import joints as jt


class JointTopology(NamedTuple):
    side_joint: np.ndarray  # (n_sides,)
    side_body: np.ndarray  # (n_sides,)
    side_ang: np.ndarray  # (n_sides,) index into the (2 nj) angular blocks of ``joints.evaluate``
    sign: np.ndarray  # (n_sides,)
    inertia_inv_side: np.ndarray  # (n_sides, 3, 3)
    inv_mass_signed: np.ndarray  # (n_sides,)
    pair_s: np.ndarray  # (n_pairs,)
    pair_t: np.ndarray  # (n_pairs,)
    pair_coeff: np.ndarray  # (n_pairs,) s_s s_t / m
    pair_band: np.ndarray  # (n_pairs,) joint(s) - joint(t)
    pair_col: np.ndarray  # (n_pairs,) joint(t)
    pad_diag: np.ndarray | None  # (n_joints, rows) 1 for padded equations; None without padding
    joints: jt.JointSet
    joint_perm: np.ndarray  # (n_joints,) input index of each internal joint

    @property
    def n_joints(self) -> int:
        return self.joints.n_joints

    @property
    def rows(self) -> int:
        return self.joints.rows

    @property
    def n_sides(self) -> int:
        return self.side_joint.shape[0]

    @property
    def block_bandwidth(self) -> int:
        """Half bandwidth of G in blocks of ``rows`` x ``rows``."""
        return int(self.pair_band.max())


def _groups(joint_a, joint_b):
    by_body: dict[int, list[int]] = {}
    for j in range(joint_b.shape[0]):
        for body in (joint_a[j], joint_b[j]):
            if body >= 0:
                by_body.setdefault(int(body), []).append(j)
    return by_body


def _band(by_body, position):
    return max(
        (max(position[j] for j in js) - min(position[j] for j in js)) for js in by_body.values()
    )


def band_reducing_order(joint_a, joint_b) -> np.ndarray:
    """Joint permutation (new -> old) that narrows the block band of G, or the identity."""
    n = joint_b.shape[0]
    by_body = _groups(joint_a, joint_b)
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for js in by_body.values():
        for j in js:
            neighbors[j].update(js)
            neighbors[j].discard(j)
    order, seen = [], [False] * n
    for start in sorted(range(n), key=lambda j: len(neighbors[j])):
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        for j in queue:
            order.append(j)
            for k in sorted(neighbors[j], key=lambda k: len(neighbors[k])):
                if not seen[k]:
                    seen[k] = True
                    queue.append(k)
    perm = np.array(order[::-1], dtype=np.intp)
    new_pos = np.empty(n, dtype=np.intp)
    new_pos[perm] = np.arange(n)
    if _band(by_body, new_pos) < _band(by_body, np.arange(n)):
        return perm
    return np.arange(n, dtype=np.intp)


def build_joint_topology(
    joint_a,
    joint_b,
    anchor_a,
    anchor_b,
    mass,
    inertia_inv,
    kind=None,
    axis_a=None,
    axis_b=None,
    rel_rot=None,
) -> JointTopology:
    joint_a = np.asarray(joint_a, dtype=np.intp)
    joint_b = np.asarray(joint_b, dtype=np.intp)
    anchor_a = np.asarray(anchor_a, dtype=np.float64)
    anchor_b = np.asarray(anchor_b, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    inertia_inv = np.asarray(inertia_inv, dtype=np.float64)
    n_joints = joint_b.shape[0]
    if n_joints == 0:
        raise ValueError("at least one joint is required")
    if np.any(joint_a == joint_b) or np.any(joint_b < 0):
        raise ValueError("a joint must connect two distinct bodies (only body a may be world)")
    kind = np.full(n_joints, jt.BALL) if kind is None else np.asarray(kind, dtype=np.intp)
    if np.any(~np.isin(kind, list(jt.ROWS))):
        raise ValueError("unknown joint kind")
    axis_a = np.zeros((n_joints, 3)) if axis_a is None else np.asarray(axis_a, dtype=np.float64)
    axis_b = np.zeros((n_joints, 3)) if axis_b is None else np.asarray(axis_b, dtype=np.float64)
    hinge = kind == jt.HINGE
    if np.any(np.linalg.norm(axis_a[hinge], axis=1) == 0.0) or np.any(
        np.linalg.norm(axis_b[hinge], axis=1) == 0.0
    ):
        raise ValueError("hinge axes must be non-zero")

    uses_axis = (kind == jt.HINGE) | (kind == jt.PRISMATIC)
    if np.any(np.linalg.norm(axis_a[uses_axis], axis=1) == 0.0):
        raise ValueError("joint axes must be non-zero")
    rel_rot = (
        np.tile(np.eye(3), (n_joints, 1, 1))
        if rel_rot is None
        else np.asarray(rel_rot, dtype=np.float64)
    )

    perm = band_reducing_order(joint_a, joint_b)
    joint_a, joint_b, anchor_a, anchor_b = (x[perm] for x in (joint_a, joint_b, anchor_a, anchor_b))
    kind, axis_a, axis_b, rel_rot = (x[perm] for x in (kind, axis_a, axis_b, rel_rot))

    joint_set = jt.build_joint_set(
        kind, joint_a, joint_b, anchor_a, anchor_b, axis_a, axis_b, rel_rot
    )
    rows = joint_set.rows

    has_a = joint_a >= 0
    ids = np.arange(n_joints)
    side_joint = np.concatenate((ids[has_a], ids))
    side_body = np.concatenate((joint_a[has_a], joint_b))
    side_ang = np.concatenate((ids[has_a], n_joints + ids))
    sign = np.concatenate((-np.ones(int(has_a.sum())), np.ones(n_joints)))

    by_body: dict[int, list[int]] = {}
    for s, body in enumerate(side_body):
        by_body.setdefault(int(body), []).append(s)
    first, second = [], []
    for sides in by_body.values():
        for s in sides:
            for t in sides:
                if side_joint[s] >= side_joint[t]:
                    first.append(s)
                    second.append(t)
    pair_s = np.array(first, dtype=np.intp)
    pair_t = np.array(second, dtype=np.intp)

    joint_rows = np.array([jt.ROWS[int(k)] for k in kind])
    pad_diag = (np.arange(rows)[None, :] >= joint_rows[:, None]).astype(np.float64)

    return JointTopology(
        side_joint=side_joint,
        side_body=side_body,
        side_ang=side_ang,
        sign=sign,
        inertia_inv_side=inertia_inv[side_body],
        inv_mass_signed=sign / mass[side_body],
        pair_s=pair_s,
        pair_t=pair_t,
        pair_coeff=sign[pair_s] * sign[pair_t] / mass[side_body[pair_s]],
        pair_band=side_joint[pair_s] - side_joint[pair_t],
        pair_col=side_joint[pair_t],
        pad_diag=pad_diag if pad_diag.any() else None,
        joints=joint_set,
        joint_perm=perm,
    )
