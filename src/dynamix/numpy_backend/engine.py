"""Moreau theta-method time stepping with velocity-level bilateral constraints.

Per step (state at t_k = (q, u)):
    q_th   = q + theta dt u                      (midpoint position; q_rot * exp(theta dt w))
    M(u+ - u) = dt h(u) + J(q_th)^T lam           (h: gravity and gyroscopic -w x I w, explicit)
    M = diag(m 1, I_body) is constant because the angular velocity is expressed in the body frame.
    J(q_th) u+ = -(stabilization / dt) g(q_th)    (velocity-level constraint with drift correction)
    q+     = q_th + (1 - theta) dt u+
Constraints are enforced on velocities; the bias term only counteracts drift.
Contacts are not part of this stage.
"""

from __future__ import annotations

import numpy as np

from dynamix.core.buffers import SystemBuffers
from dynamix.numpy_backend import quaternion as quat
from dynamix.numpy_backend.constraints import ball_joint_jacobian, ball_joint_residual


class Engine:
    def __init__(self, buffers: SystemBuffers):
        self.buf = buffers
        n, nj = buffers.n_bodies, buffers.n_joints
        self.time = 0.0
        self._jac = np.zeros((3 * nj, 6 * n))
        self._inertia_inv = np.linalg.inv(buffers.inertia_body)
        self._q_th = np.empty_like(buffers.q)
        self._u_free = np.empty_like(buffers.u)

    def step(self) -> None:
        b = self.buf
        dt, theta = b.dt, b.theta
        q, u, q_th, u_free = b.q, b.u, self._q_th, self._u_free
        v, w = u[:, :3], u[:, 3:]

        q_th[:, :3] = q[:, :3] + theta * dt * v
        q_th[:, 3:] = quat.integrate(q[:, 3:], w, theta * dt)
        rot = quat.to_matrix(q_th[:, 3:])
        inertia_inv = self._inertia_inv

        gyro = np.cross(w, np.einsum("nij,nj->ni", b.inertia_body, w))
        u_free[:, :3] = v + dt * b.gravity
        u_free[:, 3:] = w - dt * np.einsum("nij,nj->ni", inertia_inv, gyro)

        u_new = u_free
        if b.n_joints:
            jac = ball_joint_jacobian(
                q_th, b.joint_a, b.joint_b, b.joint_anchor_a, b.joint_anchor_b, self._jac, rot
            )
            g = ball_joint_residual(
                q_th, b.joint_a, b.joint_b, b.joint_anchor_a, b.joint_anchor_b, rot
            )
            n = b.n_bodies
            # M^-1 J^T, stored transposed as (3 nj, 6 n)
            jm = jac.reshape(-1, n, 2, 3).copy()
            jm[:, :, 0, :] /= b.mass[None, :, None]
            jm[:, :, 1, :] = np.einsum("rnk,nkl->rnl", jm[:, :, 1, :], inertia_inv)
            jm = jm.reshape(-1, 6 * n)
            schur = jm @ jac.T
            rhs = -(b.stabilization / dt) * g.reshape(-1) - jac @ u_free.reshape(-1)
            lam = np.linalg.solve(schur, rhs)
            u_new = u_free + (lam @ jm).reshape(n, 6)

        u[:] = u_new
        q[:, :3] = q_th[:, :3] + (1.0 - theta) * dt * v
        q[:, 3:] = quat.integrate(q_th[:, 3:], w, (1.0 - theta) * dt)
        self.time += dt

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()


def total_energy(b: SystemBuffers) -> float:
    """Kinetic plus gravitational potential energy."""
    v, w = b.u[:, :3], b.u[:, 3:]
    kinetic = 0.5 * np.sum(b.mass * np.sum(v * v, axis=1))
    kinetic += 0.5 * np.einsum("ni,nij,nj->", w, b.inertia_body, w)
    potential = -np.sum(b.mass * (b.q[:, :3] @ b.gravity))
    return float(kinetic + potential)
