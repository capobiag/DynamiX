"""Backend-agnostic normal contacts: SDF primitives, the contact buffer and the contact rows.

Detection is separate from the solver: a detector fills a flat, fixed-capacity ``ContactBuffer``
and the solver consumes nothing else. Conventions:

- A contact belongs to a *shape* body a (a sphere, or the static plane = world, index -1) and a
  *probe* sphere b. With the signed distance ``sdf_a`` of the shape (negative inside),

      gap = sdf_a(x_b) - r_b,        n = grad sdf_a(x_b)        (points from a towards b)

  so ``gap > 0`` is separated and ``gap <= 0`` touching or penetrating.
- The contact point is the midpoint between the two surfaces, ``p = x_b - (r_b + gap / 2) n``.
- Normal row of the Jacobian, with ``r = p - x`` and u = [v (world), w (body)]:

      J_a = [ -n^T, -(R_a^T (r_a x n))^T ],      J_b = [ +n^T, +(R_b^T (r_b x n))^T ]

  For spheres the contact point lies on the line through both centres, so ``r x n = 0`` and the
  angular blocks vanish; the general form is kept for friction and other shapes.

Entries ``0 .. count - 1`` of the buffer are valid; the rest is unused (fixed shapes, no dynamic
allocation). If more contacts are found than the capacity allows, the ones with the smallest gap
are kept and ``overflow`` is set.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ContactBuffer(NamedTuple):
    body_a: np.ndarray  # (C,) integer body index, -1 for the world (plane)
    body_b: np.ndarray  # (C,) integer body index
    point: np.ndarray  # (C, 3) world-frame contact point
    normal: np.ndarray  # (C, 3) unit normal, from a towards b
    gap: np.ndarray  # (C,) signed distance
    restitution: np.ndarray  # (C,) Newton coefficient of the pair
    count: int  # number of valid entries
    overflow: bool  # True if contacts were dropped for lack of capacity


def empty_buffer(capacity: int) -> ContactBuffer:
    return ContactBuffer(
        np.full(capacity, -1, dtype=np.intp),
        np.zeros(capacity, dtype=np.intp),
        np.zeros((capacity, 3)),
        np.zeros((capacity, 3)),
        np.zeros(capacity),
        np.zeros(capacity),
        0,
        False,
    )


def sphere_sdf(xp, x, centre, radius):
    """Signed distance of points ``x`` (m, 3) to spheres (m, 3), (m,); returns it and its gradient.

    The gradient at a coincident centre is the (arbitrary) unit z axis.
    """
    delta = x - centre
    dist = xp.sqrt(xp.sum(delta * delta, axis=-1))
    safe = xp.where(dist > 1e-12, dist, 1.0)
    unit = xp.where(
        (dist > 1e-12)[..., None], delta / safe[..., None], xp.asarray([0.0, 0.0, 1.0], x.dtype)
    )
    return dist - radius, unit


def plane_sdf(xp, x, normal, offset):
    """Signed distance of points ``x`` (m, 3) to planes ``normal . x = offset``; plus gradient."""
    return xp.sum(x * normal, axis=-1) - offset, xp.broadcast_to(normal, x.shape)


def sphere_vs_sdf(xp, x_b, radius_b, sdf, grad):
    """Gap, normal and contact point of probe spheres (centre ``x_b``) against SDF values.

    ``sdf`` and ``grad`` are the SDF of the shape body and its gradient at ``x_b``.
    """
    gap = sdf - radius_b
    point = x_b - (radius_b + 0.5 * gap)[..., None] * grad
    return gap, grad, point


def contact_arms(xp, buf_point, buf_normal, pos_a, pos_b, rot_a, rot_b):
    """Angular Jacobian blocks ``(ang_a, ang_b)`` (C, 3): ``+-R^T (r x n)`` (see module docs).

    ``pos_*`` (C, 3) and ``rot_*`` (C, 3, 3) are the centre and orientation of each body; use the
    identity and the origin for the world.
    """
    arm_a = xp.cross(buf_point - pos_a, buf_normal)
    arm_b = xp.cross(buf_point - pos_b, buf_normal)
    ang_a = -xp.einsum("cji,cj->ci", rot_a, arm_a)
    ang_b = xp.einsum("cji,cj->ci", rot_b, arm_b)
    return ang_a, ang_b
