"""Compile an ECS scene into flat, preallocated NumPy buffers.

This is the only place where ECS components are read. Simulation loops only see
``SystemBuffers``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamix.ecs.scene import WORLD, Scene


@dataclass
class SystemBuffers:
    """Numerical state and constant data (float64, C-contiguous).

    q: (n, 7) [x, y, z, qx, qy, qz, qw]
    u: (n, 6) [vx, vy, vz, wx, wy, wz], linear velocity in the world frame,
    angular velocity in the body frame
    """

    q: np.ndarray
    u: np.ndarray
    mass: np.ndarray  # (n,)
    inertia_body: np.ndarray  # (n, 3, 3)
    gravity: np.ndarray  # (3,)
    joint_a: np.ndarray  # (nj,) body index, -1 for world
    joint_b: np.ndarray  # (nj,) body index
    joint_anchor_a: np.ndarray  # (nj, 3)
    joint_anchor_b: np.ndarray  # (nj, 3)
    dt: float
    theta: float
    stabilization: float
    body_entities: tuple[int, ...] = ()

    @property
    def n_bodies(self) -> int:
        return self.q.shape[0]

    @property
    def n_joints(self) -> int:
        return self.joint_a.shape[0]


def compile_scene(scene: Scene) -> SystemBuffers:
    bodies = scene.bodies()
    joints = scene.ball_joints()
    index = {ent: i for i, (ent, _) in enumerate(bodies)}
    n, nj = len(bodies), len(joints)

    q = np.zeros((n, 7))
    u = np.zeros((n, 6))
    mass = np.zeros(n)
    inertia = np.zeros((n, 3, 3))
    for i, (_, b) in enumerate(bodies):
        norm = np.linalg.norm(b.orientation)
        if norm == 0.0:
            raise ValueError("zero quaternion")
        q[i, :3] = b.position
        q[i, 3:] = b.orientation / norm
        u[i, :3] = b.linear_velocity
        u[i, 3:] = b.angular_velocity
        mass[i] = b.mass
        inertia[i] = b.inertia

    joint_a = np.full(nj, WORLD, dtype=np.intp)
    joint_b = np.zeros(nj, dtype=np.intp)
    anchor_a = np.zeros((nj, 3))
    anchor_b = np.zeros((nj, 3))
    for j, (_, jt) in enumerate(joints):
        joint_a[j] = WORLD if jt.body_a == WORLD else index[jt.body_a]
        joint_b[j] = index[jt.body_b]
        anchor_a[j] = jt.anchor_a
        anchor_b[j] = jt.anchor_b

    cfg = scene.config
    return SystemBuffers(
        q=q,
        u=u,
        mass=mass,
        inertia_body=inertia,
        gravity=scene.gravity.acceleration.copy(),
        joint_a=joint_a,
        joint_b=joint_b,
        joint_anchor_a=anchor_a,
        joint_anchor_b=anchor_b,
        dt=cfg.dt,
        theta=cfg.theta,
        stabilization=cfg.stabilization,
        body_entities=tuple(ent for ent, _ in bodies),
    )
