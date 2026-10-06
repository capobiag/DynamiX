"""Block-sparse ball joint system with a banded Delassus matrix.

Each joint has at most two *sides* (one per attached body; the world has none). For the side of
body i with sign s (+1 for body b, -1 for body a) and body-frame anchor r:

    J_side = [ s 1,  -s R_i [r]_x ]        (3 x 6; only the 3 x 3 rotational block varies)

The Delassus matrix G = J M^-1 J^T is assembled from per-side blocks into lower banded storage.
Its block (j, k) is non-zero only when joints j and k share a body, so for a chain with joints
numbered along the chain the half-bandwidth is constant (5) and the solve is O(n). Topology
dependent index arrays are built once; per-step work arrays are reused.

The dense functions in ``constraints.py`` serve as the reference for these blocks.
"""

from __future__ import annotations

import numpy as np

from dynamix.numpy_backend import quaternion as quat


class BallJointSystem:
    def __init__(self, joint_a, joint_b, anchor_a, anchor_b, mass, inertia_inv):
        joint_a = np.asarray(joint_a, dtype=np.intp)
        joint_b = np.asarray(joint_b, dtype=np.intp)
        anchor_a = np.asarray(anchor_a, dtype=np.float64)
        anchor_b = np.asarray(anchor_b, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        n_joints = joint_b.shape[0]
        if n_joints == 0:
            raise ValueError("at least one joint is required")
        if np.any(joint_a == joint_b) or np.any(joint_b < 0):
            raise ValueError("a joint must connect two distinct bodies (only body a may be world)")
        has_a = joint_a >= 0
        ids = np.arange(n_joints)

        self.n_joints = n_joints
        self.side_joint = np.concatenate((ids[has_a], ids))
        self.side_body = np.concatenate((joint_a[has_a], joint_b))
        self.sign = np.concatenate((-np.ones(int(has_a.sum())), np.ones(n_joints)))
        self.anchor = np.concatenate((anchor_a[has_a], anchor_b))
        self.world_offset = np.zeros((n_joints, 3))
        self.world_offset[~has_a] = -anchor_a[~has_a]

        # d(g)/d(w) = R @ (-s [r]_x)
        self._jw_factor = -self.sign[:, None, None] * quat.skew(self.anchor)
        self._inertia_inv_side = np.asarray(inertia_inv)[self.side_body]
        self._inv_mass_signed = self.sign / mass[self.side_body]

        pair_s, pair_t = self._side_pairs()
        self._pair_s, self._pair_t = pair_s, pair_t
        self._pair_coeff = self.sign[pair_s] * self.sign[pair_t] / mass[self.side_body[pair_s]]
        self._build_band_index(pair_s, pair_t)

        self.jw = np.empty((self.n_sides, 3, 3))
        self._ww = np.empty((self.n_sides, 3, 3))
        self._blocks = np.empty((pair_s.shape[0], 3, 3))
        self.g = np.empty((n_joints, 3))
        self._rate = np.empty((n_joints, 3))

    @property
    def n_sides(self) -> int:
        return self.side_joint.shape[0]

    def _side_pairs(self):
        """Ordered side pairs (s, t) on the same body with joint(s) >= joint(t)."""
        by_body: dict[int, list[int]] = {}
        for s, body in enumerate(self.side_body):
            by_body.setdefault(int(body), []).append(s)
        first, second = [], []
        for sides in by_body.values():
            for s in sides:
                for t in sides:
                    if self.side_joint[s] >= self.side_joint[t]:
                        first.append(s)
                        second.append(t)
        return np.array(first, dtype=np.intp), np.array(second, dtype=np.intp)

    def _build_band_index(self, pair_s, pair_t) -> None:
        size = 3 * self.n_joints
        rows = 3 * self.side_joint[pair_s][:, None] + (np.arange(9) // 3)[None, :]
        cols = 3 * self.side_joint[pair_t][:, None] + (np.arange(9) % 3)[None, :]
        lower = rows >= cols
        self._keep = np.flatnonzero(lower.ravel())
        offset = (rows - cols)[lower]
        self.half_bandwidth = int(offset.max())
        self._target = (offset * size + cols[lower]).astype(np.intp)
        self._band_shape = (self.half_bandwidth + 1, size)

    def update(self, q, rot) -> np.ndarray:
        """Evaluate g(q) into ``self.g`` and the rotational Jacobian blocks into ``self.jw``."""
        rot_side = rot[self.side_body]
        np.matmul(rot_side, self._jw_factor, out=self.jw)
        lever = np.matmul(rot_side, self.anchor[:, :, None])[:, :, 0]
        points = self.sign[:, None] * (q[self.side_body, :3] + lever)
        self.g[:] = self.world_offset
        np.add.at(self.g, self.side_joint, points)
        return self.g

    def delassus_band(self) -> np.ndarray:
        """G = J M^-1 J^T in lower banded storage; requires a prior ``update``."""
        np.matmul(self._inertia_inv_side, self.jw.transpose(0, 2, 1), out=self._ww)
        np.matmul(self.jw[self._pair_s], self._ww[self._pair_t], out=self._blocks)
        values = self._blocks.reshape(-1, 9)
        values[:, ::4] += self._pair_coeff[:, None]
        flat = np.bincount(
            self._target,
            weights=values.ravel()[self._keep],
            minlength=self._band_shape[0] * self._band_shape[1],
        )
        return flat.reshape(self._band_shape)

    def jacobian_times(self, u) -> np.ndarray:
        """J u as a flat vector of length 3 n_joints; requires a prior ``update``."""
        body = self.side_body
        rate = self.sign[:, None] * u[body, :3] + np.matmul(self.jw, u[body, 3:, None])[:, :, 0]
        self._rate.fill(0.0)
        np.add.at(self._rate, self.side_joint, rate)
        return self._rate.reshape(-1)

    def add_impulse(self, impulse, u) -> None:
        """u += M^-1 J^T impulse in place; needs ``delassus_band`` first (it fills ``_ww``)."""
        local = impulse[self.side_joint]
        d_lin = self._inv_mass_signed[:, None] * local
        d_ang = np.matmul(self._ww, local[:, :, None])[:, :, 0]
        np.add.at(u[:, :3], self.side_body, d_lin)
        np.add.at(u[:, 3:], self.side_body, d_ang)
