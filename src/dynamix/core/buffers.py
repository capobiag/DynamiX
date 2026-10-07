"""Compile an ECS scene into flat, preallocated NumPy buffers.

This is the only place where ECS components are read. Simulation loops only see
``SystemBuffers``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamix.core import joints as jt
from dynamix.core import rotations
from dynamix.core.joints import BALL, FIXED, HINGE, PRISMATIC
from dynamix.ecs.components import FixedJoint, HingeJoint, PrismaticJoint
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
    joint_kind: np.ndarray | None = None  # (nj,) dynamix.core.joints kind
    joint_axis_a: np.ndarray | None = None  # (nj, 3) hinge / slide axes (zero otherwise)
    joint_axis_b: np.ndarray | None = None
    joint_rel_rot: np.ndarray | None = None  # (nj, 3, 3) initial R_a^T R_b (fixed, prismatic)

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
    for j, (_, joint) in enumerate(joints):
        if isinstance(joint, HingeJoint):
            kind[j], axis_a[j], axis_b[j] = HINGE, joint.axis_a, joint.axis_b
        elif isinstance(joint, PrismaticJoint):
            kind[j], axis_a[j] = PRISMATIC, joint.axis_a
        elif isinstance(joint, FixedJoint):
            kind[j] = FIXED
        joint_a[j] = WORLD if joint.body_a == WORLD else index[joint.body_a]
        joint_b[j] = index[joint.body_b]
        anchor_a[j] = joint.anchor_a
        anchor_b[j] = joint.anchor_b

    rot = rotations.to_matrix(np, q[:, 3:])
    rot_a = np.where((joint_a >= 0)[:, None, None], rot[joint_a], np.eye(3))
    rel_rot = np.swapaxes(rot_a, 1, 2) @ rot[joint_b]
    for j in np.flatnonzero(kind == HINGE):
        ax_a = axis_a[j] / np.linalg.norm(axis_a[j])
        world_a = ax_a if joint_a[j] == WORLD else rot[joint_a[j]] @ ax_a
        world_b = rot[joint_b[j]] @ (axis_b[j] / np.linalg.norm(axis_b[j]))
        if np.linalg.norm(world_a - world_b) > 1e-6:
            raise ValueError(f"hinge joint {j}: axes are not aligned in the initial configuration")

    if nj:
        joint_set = jt.build_joint_set(
            kind, joint_a, joint_b, anchor_a, anchor_b, axis_a, axis_b, rel_rot
        )
        residual, _, _ = jt.evaluate(np, joint_set, q, rot)
        for j in np.flatnonzero((kind == FIXED) | (kind == PRISMATIC)):
            if np.abs(residual[j, : jt.ROWS[int(kind[j])]]).max() > 1e-6:
                raise ValueError(f"joint {j}: not satisfied in the initial configuration")

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
        joint_rel_rot=rel_rot,
    )
