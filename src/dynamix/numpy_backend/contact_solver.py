"""Normal-contact impulses by block iteration (NumPy).

Contacts are frictionless sphere contacts with the complementarity law (see docs/theory.md)

    0 <= lam  _|_  J_N u+ - b >= 0,    b = max(-e u_N^-, -beta gap / dt)

with ``e`` applied only above an approach-speed threshold.

Each iteration is a projected Jacobi sweep over all contacts at once (scatter by ``bincount``),
followed by a joint correction ``G dLam = -(stab / dt) g - J u`` that re-imposes the bilateral
constraints with the factorised joint Delassus matrix. The final velocities therefore satisfy
the joint constraints exactly; contacts are met up to ``contact_tolerance``.
Sphere contacts have no angular Jacobian blocks (``r x n = 0``), so only linear velocities enter.
"""

from __future__ import annotations

import numpy as np

from dynamix.core.buffers import SystemBuffers
from dynamix.core.contacts import ContactBuffer
from dynamix.numpy_backend.banded import factor_spd_banded, solve_factored
from dynamix.numpy_backend.joint_system import JointSystem


class ContactSolver:
    def __init__(self, buffers: SystemBuffers, joints: JointSystem | None):
        self.n = buffers.q.shape[0]
        self.joints = joints
        self.iterations = buffers.contact_iterations
        self.tolerance = buffers.contact_tolerance
        self.omega = buffers.contact_omega
        self.beta = buffers.contact_stabilization
        self.threshold = buffers.restitution_threshold
        self.stabilization = buffers.stabilization
        self.dt = buffers.dt
        # inverse mass with a trailing zero entry for the world (index -1 maps to n)
        self.inv_mass = np.append(1.0 / buffers.mass, 0.0)
        self.u_ext = np.zeros((self.n + 1, 6))
        self.last_iterations = 0

    def solve(self, contacts: ContactBuffer, u0: np.ndarray, u_free: np.ndarray, g, band):
        """Overwrite ``u_free`` with the post-contact velocities.

        ``u0`` is the start-of-step velocity (for restitution). ``u_free`` must already satisfy
        the joint constraints; ``g`` and ``band`` are the joint gap and Delassus band at the
        current configuration (None without joints). Returns the contact impulses ``(count,)``.
        """
        n, c = self.n, contacts.count
        a = np.where(contacts.body_a[:c] < 0, n, contacts.body_a[:c])
        bb = contacts.body_b[:c]
        normal = contacts.normal[:c]
        comps = (normal[:, 0], normal[:, 1], normal[:, 2])

        u0_ext = np.zeros((n + 1, 3))
        u0_ext[:n] = u0[:, :3]
        approach = np.einsum("ci,ci->c", normal, u0_ext[bb] - u0_ext[a])
        rhs = np.where(approach < -self.threshold, -contacts.restitution[:c] * approach, 0.0)
        if self.beta > 0.0:
            rhs = np.maximum(rhs, -self.beta * contacts.gap[:c] / self.dt)

        inv_a, inv_b = self.inv_mass[a], self.inv_mass[bb]
        idx = np.concatenate([a, bb])
        count = np.bincount(idx, minlength=n + 1)
        count[n] = 0
        scale = self.omega / (np.maximum(count[a], count[bb]) * (inv_a + inv_b))

        u = self.u_ext
        u[:n] = u_free
        view = u[:n]
        factor = bias = None
        if self.joints is not None:
            factor = factor_spd_banded(band)
            bias = -(self.stabilization / self.dt) * g.reshape(-1)
        lam = np.zeros(c)
        self.last_iterations = 0
        for _ in range(self.iterations):
            self.last_iterations += 1
            rel = sum(k * (u[bb, i] - u[a, i]) for i, k in enumerate(comps))
            new = np.maximum(0.0, lam - scale * (rel - rhs))
            delta = new - lam
            lam = new
            change = float(np.abs(delta).max())
            for i, k in enumerate(comps):
                weights = np.concatenate([-delta * inv_a * k, delta * inv_b * k])
                u[:n, i] += np.bincount(idx, weights, n + 1)[:n]
            if factor is not None:
                corr = solve_factored(factor, bias - self.joints.jacobian_times(view))
                self.joints.add_impulse(corr.reshape(-1, self.joints.rows), view)
                change = max(change, float(np.abs(corr).max()))
            if change < self.tolerance:
                break
        u_free[:] = view
        return lam
