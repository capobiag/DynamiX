"""Compile an ECS scene into flat, preallocated NumPy buffers.

This is the only place where ECS components are read. Simulation loops only see
``SystemBuffers``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamix.core import rotations
from dynamix.core.joints import BALL, HINGE
from dynamix.ecs.components import HingeJoint
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
    joint_kind: np.ndarray | None = None  # (nj,) dynamix.core.joints.BALL / HINGE
    joint_axis_a: np.ndarray | None = None  # (nj, 3) hinge axes (zero for other kinds)
    joint_axis_b: np.ndarray | None = None

    @property
    def n_bodies(self) -> int:
        return self.q.shape[0]

    @property
    def n_joints(self) -> int:
        return self.joint_a.shape[0]


def compile_scene(scene: Scene) -> SystemBuffers:
    bodies = scene.bodies()
    joints = scene.joints()
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
    kind = np.full(nj, BALL, dtype=np.intp)
    axis_a = np.zeros((nj, 3))
    axis_b = np.zeros((nj, 3))
    for j, (_, jt) in enumerate(joints):
        if isinstance(jt, HingeJoint):
            kind[j], axis_a[j], axis_b[j] = HINGE, jt.axis_a, jt.axis_b
        joint_a[j] = WORLD if jt.body_a == WORLD else index[jt.body_a]
        joint_b[j] = index[jt.body_b]
        anchor_a[j] = jt.anchor_a
        anchor_b[j] = jt.anchor_b

    rot = rotations.to_matrix(np, q[:, 3:])
    for j in np.flatnonzero(kind == HINGE):
        ax_a = axis_a[j] / np.linalg.norm(axis_a[j])
        world_a = ax_a if joint_a[j] == WORLD else rot[joint_a[j]] @ ax_a
        world_b = rot[joint_b[j]] @ (axis_b[j] / np.linalg.norm(axis_b[j]))
        if np.linalg.norm(world_a - world_b) > 1e-6:
            raise ValueError(f"hinge joint {j}: axes are not aligned in the initial configuration")

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
        joint_kind=kind,
        joint_axis_a=axis_a,
        joint_axis_b=axis_b,
    )
