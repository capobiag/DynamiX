"""Block-sparse joint system with a banded Delassus matrix (NumPy).

Each joint has at most two *sides* (one per attached body; the world has none) and ``rows``
equations (see ``dynamix.core.joints``). For a side of body i with sign s (+1 for body b, -1 for
body a)

    J_side = [ s E,  ang ]        (rows x 6; only the rows x 3 angular block varies)

The Delassus matrix G = J M^-1 J^T is assembled from per-side blocks into lower banded storage.
Its block (j, k) is non-zero only when joints j and k share a body, so for a chain with joints
numbered along the chain the half-bandwidth is constant and the solve is O(n). Topology
dependent index arrays are built once; per-step work arrays are reused.
"""

from __future__ import annotations

import numpy as np

from dynamix.core import joint_math as jm
from dynamix.core import joints as jt
from dynamix.core.joint_topology import build_joint_topology


class JointSystem:
    def __init__(
        self, joint_a, joint_b, anchor_a, anchor_b, mass, inertia_inv, kind=None, axis_a=None,
        axis_b=None,
    ):  # fmt: skip
        topo = build_joint_topology(
            joint_a, joint_b, anchor_a, anchor_b, mass, inertia_inv, kind, axis_a, axis_b
        )
        self.n_joints, self.rows = topo.n_joints, topo.rows
        self.side_joint, self.side_body, self.sign = topo.side_joint, topo.side_body, topo.sign
        self._joints, self._side_ang = topo.joints, topo.side_ang
        self._inertia_inv_side = topo.inertia_inv_side
        self._inv_mass_signed = topo.inv_mass_signed
        self._pair_s, self._pair_t, self._pair_coeff = topo.pair_s, topo.pair_t, topo.pair_coeff
        self._pad = None if topo.pad_diag is None else topo.pad_diag.reshape(-1)
        self._build_band_index(topo.pair_s, topo.pair_t)

        r = self.rows
        self.jw = np.empty((self.n_sides, r, 3))
        self._ww = np.empty((self.n_sides, 3, r))
        self._blocks = np.empty((topo.pair_s.shape[0], r, r))
        self.g = np.empty((self.n_joints, r))
        self._rate = np.empty((self.n_joints, r))

    @property
    def n_sides(self) -> int:
        return self.side_joint.shape[0]

    def _build_band_index(self, pair_s, pair_t) -> None:
        r = self.rows
        size = r * self.n_joints
        rows = r * self.side_joint[pair_s][:, None] + (np.arange(r * r) // r)[None, :]
        cols = r * self.side_joint[pair_t][:, None] + (np.arange(r * r) % r)[None, :]
        lower = rows >= cols
        self._keep = np.flatnonzero(lower.ravel())
        offset = (rows - cols)[lower]
        self.half_bandwidth = int(offset.max())
        self._target = (offset * size + cols[lower]).astype(np.intp)
        self._band_shape = (self.half_bandwidth + 1, size)

    def update(self, q, rot) -> np.ndarray:
        """Evaluate g(q) into ``self.g`` and the angular Jacobian blocks into ``self.jw``."""
        g, ang = jt.evaluate(np, self._joints, q, rot)
        self.g = g
        self.jw = ang[self._side_ang]
        return g

    def delassus_band(self) -> np.ndarray:
        """G = J M^-1 J^T in lower banded storage; requires a prior ``update``."""
        r = self.rows
        np.matmul(self._inertia_inv_side, self.jw.transpose(0, 2, 1), out=self._ww)
        np.matmul(self.jw[self._pair_s], self._ww[self._pair_t], out=self._blocks)
        values = self._blocks.reshape(-1, r * r)
        values[:, : jt.POSITION_ROWS * (r + 1) : r + 1] += self._pair_coeff[:, None]
        flat = np.bincount(
            self._target,
            weights=values.ravel()[self._keep],
            minlength=self._band_shape[0] * self._band_shape[1],
        )
        band = flat.reshape(self._band_shape)
        if self._pad is not None:
            band[0] += self._pad
        return band

    def jacobian_times(self, u) -> np.ndarray:
        """J u as a flat vector of length rows * n_joints; requires a prior ``update``."""
        body = self.side_body
        rate = jm.side_rate(np, self.jw, self.sign, u[body, :3], u[body, 3:])
        self._rate.fill(0.0)
        np.add.at(self._rate, self.side_joint, rate)
        return self._rate.reshape(-1)

    def add_impulse(self, impulse, u) -> None:
        """u += M^-1 J^T impulse in place; needs ``delassus_band`` first (it fills ``_ww``)."""
        local = impulse[self.side_joint]
        d_lin = self._inv_mass_signed[:, None] * local[:, : jt.POSITION_ROWS]
        d_ang = np.matmul(self._ww, local[:, :, None])[:, :, 0]
        np.add.at(u[:, :3], self.side_body, d_lin)
        np.add.at(u[:, 3:], self.side_body, d_ang)
